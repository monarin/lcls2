"""Offline DAQ writer equivalence, failure cleanup, and rate-sequence checks."""
import ast
from contextlib import contextmanager, ExitStack, redirect_stderr
import io
import json
import logging
import os
from pathlib import Path
import runpy
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import numpy as np
from psdaq.configdb.epixquad_layout import RAW_ASIC_LAYOUT, RAW_SHAPE
from psdaq.debugtools.epixquad1kfps import capture_gain_writer as writers
from psdaq.debugtools.epixquad1kfps import capture_standalone as cap
from psdaq.debugtools.epixquad1kfps import standalone_hardware as hw
from psdaq.debugtools.epixquad1kfps import write_gain_mode_standalone as gain


class RegisterCamera:
    """Model only SACI addressing/writes, independently of map conversion."""
    def __init__(self):
        self.image = np.zeros((16, 178, 192), dtype=np.uint8)
        self.events = []
        self.rows = [0] * 16
        self.cols = [0] * 16
        self.Epix10kaSaci = {}
        for asic in range(16):
            variables = {}
            for name in ('enable', 'IsEn', 'trbit', 'PrepareMultiConfig',
                         'WriteMatrixData', 'RowCounter', 'ColCounter', 'WritePixelData'):
                variables[name] = SimpleNamespace(set=lambda value, a=asic, n=name: self.write(a, n, value))
            self.Epix10kaSaci[asic] = SimpleNamespace(**variables)

    def write(self, asic, name, value):
        self.events.append((asic, name, value))
        if name == 'WriteMatrixData':
            self.image[asic].fill(value)
        elif name == 'RowCounter':
            self.rows[asic] = value
        elif name == 'ColCounter':
            bank = (0xe, 0xd, 0xb, 0x7).index(value >> 7)
            self.cols[asic] = bank * 48 + (value & 0x7f)
        elif name == 'WritePixelData':
            self.image[asic, self.rows[asic], self.cols[asic]] = value

    def raw(self):
        # Explicit geometry, independent of the writer's conversion helper.
        out = np.empty(RAW_SHAPE, dtype=np.uint8)
        for segment in range(4):
            a = self.image[4*segment:4*segment+4, :176, :]
            out[segment, :176, :192] = a[2, ::-1, ::-1]
            out[segment, :176, 192:] = a[1, ::-1, ::-1]
            out[segment, 176:, :192] = a[3]
            out[segment, 176:, 192:] = a[0]
        return out


def offline_daq_helper():
    """Compile the unchanged checkout function, excluding hardware imports.

    Production uses a normal module import. AST extraction is test-only so
    these register-model checks can also run without Rogue/PCIe packages.
    """
    path = writers.describe_writer('daq')['source_path']
    tree = ast.parse(Path(path).read_text(), filename=path)
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ('mode', '_apply_raw_pixel_map_program')]
    env = dict(np=np, time=time, logging=logging, RAW_SHAPE=RAW_SHAPE,
               RAW_ASIC_LAYOUT=RAW_ASIC_LAYOUT, BANK_OFFSETS=gain.BANK_OFFSETS)
    exec(compile(tree, path, 'exec'), env)
    return env['_apply_raw_pixel_map_program']


class GainWriterTests(unittest.TestCase):
    def test_both_writers_produce_same_map_all_modes_edges_banks_and_rotations(self):
        helper = offline_daq_helper()
        # All ASICs, both edges and every bank; raw rows 175 and 176 included.
        mask = np.zeros(RAW_SHAPE, dtype=bool)
        mask[:, (0, 175, 176, 351), :] = True
        mask[:, (12, 210), ::47] = True
        selected = [gain._raw_to_direct(*map(int, pixel)) for pixel in np.argwhere(mask)]
        for mode in gain.GAIN_MODES.values():
            with self.subTest(mode=mode.name):
                pixels = selected if mode.selected_value is not None else []
                args = SimpleNamespace(mode=mode, asics=list(range(16)),
                                       injection_delay=1, verbose=False)
                expected = np.where(mask, 8, 12).astype(np.uint8) if pixels else np.full(RAW_SHAPE, mode.background_value, dtype=np.uint8)
                cameras = {}
                for name, function in (('daq', helper), ('standalone', gain._write_gain_mode)):
                    camera = RegisterCamera()
                    writers.program(camera, args, pixels, writers.describe_writer(name), function)
                    np.testing.assert_array_equal(camera.raw(), expected)
                    self.assertEqual([v for _, n, v in camera.events if n == 'trbit'], [mode.trbit]*16)
                    cameras[name] = camera
                if pixels:
                    daq = cameras['daq'].events
                    standalone = cameras['standalone'].events
                    # DAQ updates ASIC 0 before ASIC 1's background fill.
                    self.assertLess(next(i for i,e in enumerate(daq) if e[1]=='WritePixelData'),
                                    next(i for i,e in enumerate(daq) if e[:2]==(1,'WriteMatrixData')))
                    # Standalone fills every background before any pixel update.
                    self.assertLess(max(i for i,e in enumerate(standalone) if e[1]=='WriteMatrixData'),
                                    next(i for i,e in enumerate(standalone) if e[1]=='WritePixelData'))

    def test_daq_uses_low_background_when_it_is_majority(self):
        mode = gain.GAIN_MODES['mapfhl']
        selected = [(0, row, col) for row in range(120) for col in range(192)]
        camera = RegisterCamera()
        writers.program(camera, SimpleNamespace(mode=mode, asics=list(range(16))), selected,
                        writers.describe_writer('daq'), offline_daq_helper())
        np.testing.assert_array_equal(camera.raw(), writers.pixel_config_map(mode, selected))
        self.assertIn((0, 'WriteMatrixData', 8), camera.events)
        self.assertIn((1, 'WriteMatrixData', 12), camera.events)

    def test_write_failure_and_partial_enable_cleanup(self):
        for fail_enable in (False, True):
            with self.subTest(fail_enable=fail_enable):
                asics = {i: Mock() for i in range(16)}
                writer = Mock(side_effect=RuntimeError('write failed'))
                if fail_enable:
                    asics[2].IsEn.set.side_effect = [RuntimeError('enable failed'), None]
                else:
                    asics[0].IsEn.set.side_effect = [None, RuntimeError('cleanup failed')]
                with redirect_stderr(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'enable failed' if fail_enable else 'write failed'):
                    writers.program(SimpleNamespace(Epix10kaSaci=asics),
                                    SimpleNamespace(mode=gain.GAIN_MODES['fm'], asics=list(range(16))),
                                    [], writers.describe_writer('daq'), writer)
                for i in range(3 if fail_enable else 16):
                    asics[i].IsEn.set.assert_called_with(False)
                    asics[i].enable.set.assert_called_with(False)
                if fail_enable:
                    writer.assert_not_called()

    def test_ghost_rows_rejected_and_wrong_source_refused(self):
        with self.assertRaisesRegex(ValueError, 'ghost rows'):
            writers.pixel_config_map(gain.GAIN_MODES['mapfml'], [(0, 176, 0)])
        with patch.object(writers.importlib, 'import_module', return_value=SimpleNamespace(__file__=gain.__file__)):
            with self.assertRaisesRegex(RuntimeError, 'recorded source'):
                writers.resolve_writer(writers.describe_writer('daq'))


class RateSequenceTests(unittest.TestCase):
    def test_phase_finalization_preserves_partial_failure_and_detects_missing_data(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            expected = np.zeros(RAW_SHAPE, dtype=bool)
            np.save(folder/'expected-map.npy', expected)
            record = {'status': 'completed'}
            args = SimpleNamespace(frames=51)
            profile = {'image_payload_bytes': [1095248]}
            self.assertEqual(cap.finalize_capture(folder, record, expected, profile, args), 1)
            self.assertEqual(record['status'], 'failed')
            # Partial failed phases must still be analyzed and exported.
            (folder/'capture.dat').write_bytes(b'partial-data')
            record = {'status': 'failed', 'error': 'phase timeout',
                      'gain_writer': writers.describe_writer('daq')}
            with patch.object(cap, 'analyze', return_value={'valid_images': 5}) as analyze, patch.object(cap, 'export_mismatches', return_value={}) as export:
                self.assertEqual(cap.finalize_capture(folder, record, expected, profile, args), 0)
            analyze.assert_called_once()
            export.assert_called_once()
            saved = json.loads((folder/'test.json').read_text())
            self.assertEqual(saved['status'], 'failed')
            self.assertEqual(saved['error'], 'phase timeout')
            self.assertEqual(saved['analysis']['valid_images'], 5)

    def test_commissioning_wrapper_dry_run_skips_restoration(self):
        source = Path(cap.__file__).resolve().parents[4]/'notes/ued-commissioning-20260928/run_once.py'
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)/'notes/commissioning'
            folder.mkdir(parents=True)
            script = folder/'run_once.py'
            script.write_text(source.read_text())
            child = Mock()
            child.wait.return_value = child.poll.return_value = 0
            with patch('sys.argv', [str(script),'--attempt','7','--gain-writer','daq','--rate-sequence-hz','10','20','10','--dry-run']), patch('socket.gethostname', return_value='offline-host'), patch('subprocess.Popen', return_value=child) as launch, patch('subprocess.run', side_effect=AssertionError('restoration prohibited')) as restore:
                with self.assertRaises(SystemExit) as exit:
                    runpy.run_path(str(script), run_name='__main__')
            self.assertEqual(exit.exception.code, 0)
            command = launch.call_args.args[0]
            self.assertIn('--dry-run', command)
            self.assertEqual(command[command.index('--gain-writer')+1], 'daq')
            restore.assert_not_called()
            record = json.loads((folder/'execution-attempt07.json').read_text())
            self.assertIn('no hardware', record['restore_skipped'])

    def test_one_map_write_and_baseline_across_three_phases_and_abort_on_failure(self):
        for failing_phase in (None, 2):
            with self.subTest(failing_phase=failing_phase), tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
                out = Path(temp)
                np.save(out/'expected-map.npy', np.zeros(RAW_SHAPE, dtype=bool))
                camera, pcie = Mock(), Mock()
                @contextmanager
                def session(*a, **kw):
                    yield pcie, camera
                stack.enter_context(patch.object(hw, 'session', session))
                stack.enter_context(patch.object(hw, 'runtime_snapshot', return_value={}))
                stack.enter_context(patch.object(hw, 'camera_identity', return_value={'adc_done': True, 'adc_failed': False}))
                baseline = stack.enter_context(patch.object(hw, 'load_camera_baseline'))
                pause = stack.enter_context(patch.object(hw, 'pause_triggers'))
                program = stack.enter_context(patch.object(gain, '_write_gain_mode'))
                stack.enter_context(patch.object(cap, 'trigger_snapshot', return_value={'auto_frequency_hz': 10}))
                streams = []
                def make_stream(*a):
                    pair = (Mock(), Mock())
                    streams.append(pair)
                    return pair
                stack.enter_context(patch.object(cap, 'make_stream_writer', side_effect=make_stream))
                wait = stack.enter_context(patch.object(cap, 'wait_for_images', side_effect=[None, TimeoutError('phase timeout')] if failing_phase else None))
                args = SimpleNamespace(gain_writer='standalone', rate_sequence_hz=[10, 20, 10],
                                       max_bytes=10000000, mode=gain.GAIN_MODES['fm'], asics=list(range(16)),
                                       injection_delay=1, verbose=False, frames=51, timeout=30,
                                       drain_timeout=1, drain_seconds=.01)
                profile = dict(image_payload_bytes=[1095248], camera_yaml='baseline',
                               camera_trigger_source=3, camera_auto_trigger=True, rate_hz=10)
                record = dict(test_id='rate-test', status='running')
                if failing_phase:
                    with self.assertRaisesRegex(TimeoutError, 'phase timeout'):
                        cap.capture(args, profile, [], out, record)
                else:
                    cap.capture(args, profile, [], out, record)
                n = failing_phase or 3
                baseline.assert_called_once()
                program.assert_called_once()
                self.assertEqual(wait.call_count, n)
                self.assertEqual(pause.call_count, n)
                self.assertEqual(pcie.StopRun.call_count, n)
                self.assertEqual(camera.SystemRegs.AutoTrigFreqHz.set.call_args_list,
                                 [call(r) for r in [10,20,10][:n]])
                self.assertEqual(len(record['phases']), n)
                for i, phase in enumerate(record['phases']):
                    self.assertEqual(phase['map_written_this_phase'], i == 0)
                    saved = json.loads((out/phase['path']/'test.json').read_text())
                    self.assertEqual(saved['status'], 'failed' if i+1==failing_phase else 'completed')
                for writer, sink in streams:
                    writer.close.assert_called_once()
                    sink.disarm.assert_called_once()

    def test_dry_run_never_resolves_writer_or_enters_hardware_and_rejects_bad_rates(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            folder = Path(temp)
            yaml = folder/'pcie.yml'
            yaml.write_text('DevRoot: {}\n')
            profile = json.loads((Path(cap.__file__).parent/'profiles/drp-ued-cmp003.json').read_text())
            profile.update(pcie_yaml=str(yaml), camera_yaml=str(yaml))
            path = folder/'profile.json'
            path.write_text(json.dumps(profile))
            stack.enter_context(patch.dict(os.environ, SUBMODULEDIR=temp))
            stack.enter_context(patch.object(cap, 'software_snapshot', return_value={}))
            hardware = stack.enter_context(patch.object(hw, 'session', side_effect=AssertionError('hardware touched')))
            resolve = stack.enter_context(patch.object(writers, 'resolve_writer', side_effect=AssertionError('writer imported')))
            argv = ['--profile',str(path),'--mode','MapFML','--no-default-pixels',
                    '--raw-pixel','2,175,30','--gain-writer','daq','--label','offline',
                    '--outdir',str(folder/'dry'),'--dry-run']
            self.assertEqual(cap.main(argv+['--rate-sequence-hz','10','20','10']), 0)
            record = json.loads((folder/'dry/test.json').read_text())
            self.assertEqual(record['status'], 'dry_run')
            self.assertEqual(record['gain_writer']['function'], '_apply_raw_pixel_map_program')
            self.assertEqual(record['rate_sequence_hz'], [10,20,10])
            self.assertFalse((folder/'dry/capture.dat').exists())
            for rates in (['--rate-hz','nan'], ['--rate-hz','0'], ['--rate-sequence-hz','10'],
                          ['--rate-hz','1e-30'], ['--rate-hz','1e30'],
                          ['--rate-sequence-hz','10','-2'], ['--rate-hz','10','--rate-sequence-hz','10','20']):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    cap.main(argv+rates)
            profile['camera_auto_trigger'] = False
            path.write_text(json.dumps(profile))
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cap.main(argv+['--rate-hz','10'])
            hardware.assert_not_called()
            resolve.assert_not_called()


if __name__ == '__main__':
    unittest.main()
