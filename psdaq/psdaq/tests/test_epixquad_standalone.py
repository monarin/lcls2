"""Offline checks for profiled capture; no detector or PCIe roots are opened."""
from contextlib import contextmanager, nullcontext
import csv
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from psdaq.debugtools.epixquad1kfps import capture_standalone as cap
from psdaq.debugtools.epixquad1kfps import capture_diagnostics as diagnostics
from psdaq.debugtools.epixquad1kfps import standalone_hardware as hw
from psdaq.debugtools.epixquad1kfps import standalone_profile as profiles
from psdaq.debugtools.epixquad1kfps import write_gain_mode_standalone as gain
from psdaq.debugtools.epixquad1kfps.export_mismatches import export_mismatches
from psdaq.configdb.epixquad_layout import RAW_SHAPE

PROFILE_DIR = Path(cap.__file__).parent / 'profiles'
SIZES = [1095248, 1095288]


def payload(extra=0):
    data = bytearray(SIZES[0] + extra)
    pixels = np.full(RAW_SHAPE, 0x4000 | 1234, dtype='<u2').tobytes()
    data[40+extra:40+extra+len(pixels)] = pixels
    return data


class ReadoutClockTests(unittest.TestCase):
    def test_clock_override_and_later_phase_verification(self):
        camera = Mock()
        camera.AcqCore.AsicRoClkT.get.return_value = 3
        camera.AcqCore.AsicRoClkHalfT.get.return_value = 1
        camera.RdoutCore.AdcPipelineDelay.get.return_value = 68
        actual = diagnostics.configure_clock(camera, 3, write=True)
        camera.AcqCore.AsicRoClkT.set.assert_called_once_with(0xAAAA0003)
        self.assertEqual(actual['adc_pipeline_delay'], 68)
        camera.AcqCore.AsicRoClkT.set.reset_mock()
        diagnostics.configure_clock(camera, 3, write=False)
        camera.AcqCore.AsicRoClkT.set.assert_not_called()
        camera.AcqCore.AsicRoClkT.get.return_value = 6
        camera.AcqCore.AsicRoClkHalfT.get.return_value = 3
        with self.assertRaisesRegex(RuntimeError, 'did not match'):
            diagnostics.configure_clock(camera, 3, write=False)

    def test_unavailable_diagnostics_are_not_zero(self):
        result = diagnostics.snapshot(SimpleNamespace(), SimpleNamespace(), 0)
        self.assertIn('error', result['camera']['RdoutCore.LineBufErr[0]'])
        self.assertIn('error', result['event_builder']['TimeoutDropCnt[2]'])


class ProfileTests(unittest.TestCase):
    def bench(self):
        return profiles.load_profile(PROFILE_DIR/'rdsrv421.json', bundle='/unused')

    def test_unvalidated_and_wrong_host_never_import_hardware(self):
        unvalidated = profiles.load_profile(PROFILE_DIR/'drp-ued-cmp003.json', bundle='/unused')
        unvalidated['status'] = 'unvalidated'
        candidates = [unvalidated, self.bench()]
        with patch.object(hw, 'load_modules') as load, patch.object(profiles.socket, 'gethostname', return_value='other'):
            for profile in candidates:
                with self.assertRaises(ValueError), hw.session(profile):
                    self.fail('must not yield')
            load.assert_not_called()

    def test_override_resolution_and_no_cross_host_defaults(self):
        profile = profiles.load_profile(PROFILE_DIR/'rdsrv421.json',
                                        {'lane': 2, 'prom_bypass': False}, bundle='/bundle')
        self.assertEqual(profile['lane'], 2)
        self.assertFalse(profile['prom_bypass'])
        self.assertTrue(profile['camera_yaml'].startswith('/bundle/'))
        args = gain._parse_args(['--mode', 'FH', '--profile', str(PROFILE_DIR/'drp-ued-cmp003.json')])
        self.assertEqual(args.lane, 0)
        self.assertFalse(args.prom_bypass)
        self.assertEqual(args.profile_data['board_type'], 'XilinxKcu1500')
        self.assertEqual(Path(args.profile_data['camera_yaml']), PROFILE_DIR.parent/'baselines/ued_beam_20260928.yml')

    def test_card_mismatch_prevents_root_construction(self):
        profile = self.bench()
        camera, pcie = Mock(), Mock()
        with patch.object(hw, 'validate_hardware'), patch.object(hw, 'load_modules', return_value=(camera, pcie)), patch.object(hw, 'probe_pcie', return_value={'board_type': 'other'}):
            with self.assertRaises(ValueError), hw.session(profile):
                self.fail('must not yield')
        pcie.DevRoot.assert_not_called()
        camera.Top.assert_not_called()

    def test_partial_startup_cleanup(self):
        for failure in ('pcie', 'camera'):
            with self.subTest(failure=failure):
                camera, pcie = Mock(), Mock()
                root = pcie.DevRoot.return_value if failure == 'pcie' else camera.Top.return_value
                root.start.side_effect = RuntimeError('startup failed')
                with patch.object(hw, 'validate_hardware'), patch.object(hw, 'load_modules', return_value=(camera, pcie)), patch.object(hw, 'probe_pcie'), patch.object(hw, 'check_identity'), patch.object(hw, 'configure_lane'):
                    with self.assertRaisesRegex(RuntimeError, 'startup failed'), hw.session(self.bench()):
                        self.fail('must not yield')
                pcie.DevRoot.return_value.stop.assert_called_once()
                if failure == 'camera':
                    camera.Top.return_value.stop.assert_called_once()
                    pcie.DevRoot.return_value.StopRun.assert_called_once()
                else:
                    camera.Top.assert_not_called()

    def test_five_expected_gain_maps(self):
        for mode in ('FH', 'FM', 'FL', 'MapFHL', 'MapFML'):
            argv = ['--mode', mode, '--dry-run']
            if mode.startswith('Map'):
                argv += ['--no-default-pixels', '--raw-pixel', '0,76,115']
            args = gain._parse_args(argv)
            selected, _ = gain._selected_pixels(args)
            expected = gain._expected_raw_map(args.mode, selected)
            self.assertEqual(expected.shape, RAW_SHAPE)
            self.assertEqual(int(expected.sum()), np.prod(RAW_SHAPE) if mode == 'FL' else 1 if mode.startswith('Map') else 0)


class BaselineTests(unittest.TestCase):
    def test_saci_enable_precedes_load_and_failure_disables_all(self):
        events = []
        def variable(label):
            return SimpleNamespace(set=lambda value: events.append((label, value)))
        asics = {i: SimpleNamespace(enable=variable('enable%d'%i),
                                   IsEn=variable('IsEn%d'%i)) for i in range(2)}
        def load(path):
            events.append(('load', path))
            raise RuntimeError('YAML failed')
        camera = SimpleNamespace(Epix10kaSaci=asics, LoadConfig=load)
        with self.assertRaisesRegex(RuntimeError, 'YAML failed'):
            hw.load_camera_baseline(camera, 'baseline', [0, 1])
        self.assertEqual(events, [('enable0',True), ('IsEn0',True),
                                 ('enable1',True), ('IsEn1',True), ('load','baseline'),
                                 ('IsEn0',False), ('enable0',False),
                                 ('IsEn1',False), ('enable1',False)])

    def test_partial_saci_enable_failure_does_not_load_yaml(self):
        asics = {0: Mock(), 1: Mock()}
        asics[1].IsEn.set.side_effect = [RuntimeError('enable failed'), None]
        camera = SimpleNamespace(Epix10kaSaci=asics, LoadConfig=Mock())
        with self.assertRaisesRegex(RuntimeError, 'enable failed'):
            hw.load_camera_baseline(camera, 'baseline', [0, 1])
        camera.LoadConfig.assert_not_called()
        for asic in asics.values():
            asic.IsEn.set.assert_called_with(False)
            asic.enable.set.assert_called_with(False)


class CaptureTests(unittest.TestCase):
    def test_pause_attempts_every_buffer_even_if_one_fails(self):
        buffers = [Mock(), Mock()]
        buffers[0].MasterEnable.set.side_effect = RuntimeError('bus failure')
        pcie = Mock()
        pcie.find.return_value = buffers
        with patch.dict(sys.modules, {'l2si_core': SimpleNamespace(TriggerEventBuffer=object)}):
            with self.assertRaisesRegex(RuntimeError, 'Could not pause'):
                hw.pause_triggers(pcie)
        for buffer in buffers:
            buffer.MasterEnable.set.assert_called_once_with(False)

    def test_only_full_error_free_images_satisfy_target(self):
        counts = cap.ImageCounts(SIZES, 10000000)
        counts.observed(bytes(1600), 0)
        counts.observed(payload(), 1)
        self.assertEqual(counts.snapshot()['valid_images'], 0)
        ticks = iter([0, 2])
        with self.assertRaises(TimeoutError):
            cap.wait_for_images(counts, 1, 1, clock=lambda: next(ticks))
        counts.observed(payload(), 0)
        cap.wait_for_images(counts, 1, 1)
        self.assertFalse(counts.room(10000000))
        with self.assertRaisesRegex(RuntimeError, 'byte limit'):
            cap.wait_for_images(counts, 1, 1)

    def test_timeout_closes_writer_and_quiesces_even_if_one_disable_fails(self):
        camera, pcie, writer, sink = Mock(), Mock(), Mock(), Mock()
        args = SimpleNamespace(max_bytes=10000000, mode=gain._normalize_mode('FH'),
                               asics=list(range(16)), injection_delay=1, verbose=False,
                               gain_writer='standalone', rate_sequence_hz=None,
                               frames=1, timeout=1, drain_timeout=0.1, drain_seconds=0.01)
        profile = {'image_payload_bytes': SIZES, 'camera_yaml': 'baseline',
                   'camera_trigger_source': 3, 'camera_auto_trigger': True, 'rate_hz': 10}
        @contextmanager
        def session(*a, **k):
            yield pcie, camera
        def fail_wait(*a):
            camera.SystemRegs.TrigEn.set.side_effect = RuntimeError('disable failed')
            raise TimeoutError('no frames')
        with tempfile.TemporaryDirectory() as temp, patch.object(hw, 'session', session), patch.object(hw, 'pause_triggers') as pause, patch.object(hw, 'load_camera_baseline'), patch.object(hw, 'runtime_snapshot', return_value={}), patch.object(hw, 'camera_identity', return_value={'adc_done': True, 'adc_failed': False}), patch.object(gain, '_write_gain_mode'), patch.object(cap, 'trigger_snapshot', return_value={}), patch.object(cap, 'make_stream_writer', return_value=(writer, sink)), patch.object(cap, 'wait_for_images', side_effect=fail_wait):
            record = {}
            with self.assertRaisesRegex(TimeoutError, 'no frames'):
                cap.capture(args, profile, [], Path(temp), record)
        writer.close.assert_called_once()
        sink.disarm.assert_called_once()
        pcie.StopRun.assert_called_once()
        pause.assert_called_once_with(pcie)
        camera.SystemRegs.AutoTrigEn.set.assert_called_with(False)
        camera.RdoutCore.RdoutEn.set.assert_called_with(False)
        self.assertTrue(record['cleanup_errors'])

    def test_records_preserve_failed_attempt_and_prevent_overwrite(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(cap, 'software_snapshot', return_value={'test': True}), patch.object(cap, 'output_log', return_value=nullcontext()), patch.dict(os.environ, {'SUBMODULEDIR': temp}), patch.object(hw, 'load_modules') as load:
            dest = Path(temp)/'attempt'
            baseline = Path(temp)/'lcls2-pgp-pcie-apps/software/config/defaults_LCLS-II.yml'
            baseline.parent.mkdir(parents=True)
            baseline.write_text('DevRoot: {}\n')
            profile_data = json.loads((PROFILE_DIR/'drp-ued-cmp003.json').read_text())
            profile_data.update(status='unvalidated', camera_yaml=None)
            profile_path = Path(temp)/'unvalidated.json'
            profile_path.write_text(json.dumps(profile_data))
            argv = ['--profile', str(profile_path), '--mode', 'FH',
                    '--label', 'blocked-profile', '--outdir', str(dest)]
            self.assertEqual(cap.main(argv), 1)
            record = json.loads((dest/'test.json').read_text())
            self.assertEqual(record['status'], 'failed')
            self.assertIn('unvalidated', record['error'])
            self.assertTrue((dest/'expected-map.npy').is_file())
            before = (dest/'test.json').read_bytes()
            with self.assertRaises(FileExistsError):
                cap.main(argv)
            self.assertEqual((dest/'test.json').read_bytes(), before)
            load.assert_not_called()

    def test_truncated_file_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'bad.dat'
            for data in (b'x', struct.pack('<IHBB', 100, 0, 0, 1)+b'x', struct.pack('<IHBB', 0, 0, 0, 1)):
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    cap.check_file_integrity(path)


class MismatchExportTests(unittest.TestCase):
    def test_coordinates_polarity_frequency_and_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            attempt = Path(temp)
            expected = np.zeros(RAW_SHAPE, dtype=bool)
            fp = np.zeros(RAW_SHAPE, dtype=np.uint32)
            fn = np.zeros_like(fp)
            # Exercise all ASICs, including the rotated tiles and both edges.
            pixels = [gain._direct_to_raw(asic, 0, 0) for asic in range(16)]
            pixels += [gain._direct_to_raw(asic, 175, 191) for asic in range(16)]
            for i, coord in enumerate(pixels):
                expected[coord] = i % 2
                (fn if i % 2 else fp)[coord] = 3 if i % 2 else 1
            np.save(attempt/'expected-map.npy', expected)
            np.savez_compressed(attempt/'mismatches.npz', fp=fp, fn=fn)
            record = {'mode': {'name': 'FL'}, 'status': 'failed',
                      'camera_identity': {'carrier_ids': [[123, 456]]},
                      'data': {'sha256': 'source-capture-hash'},
                      'analysis': {'valid_images': 4}}
            result = export_mismatches(attempt, attempt/'export', record)
            self.assertEqual(result['candidate_pixels'], 32)
            with (attempt/'export/mismatches.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 32)
            for row in rows:
                coord = tuple(int(row[k]) for k in ('segment', 'raw_row', 'raw_col'))
                direct = tuple(int(row[k]) for k in ('asic', 'asic_row', 'asic_col'))
                self.assertEqual(gain._direct_to_raw(*direct), coord)
                is_fn = bool(expected[coord])
                self.assertEqual(row['kind'], 'FN' if is_fn else 'FP')
                self.assertEqual(int(row['expected_gainbit']), int(is_fn))
                self.assertEqual(int(row['observed_gainbit']), int(not is_fn))
                self.assertEqual(float(row['fraction']), .75 if is_fn else .25)
            with np.load(attempt/'export/known-bad-candidates.npz') as masks:
                np.testing.assert_array_equal(masks['mask'], (fp != 0) | (fn != 0))
            meta = json.loads((attempt/'export/metadata.json').read_text())
            self.assertEqual(meta['source']['camera_identity'], record['camera_identity'])
            self.assertEqual(meta['source']['status'], 'failed')
            self.assertEqual(meta['source']['data'], record['data'])
            self.assertEqual(meta['source_files_sha256']['mismatches.npz'],
                             profiles.file_sha256(attempt/'mismatches.npz'))
            with self.assertRaises(FileExistsError):
                export_mismatches(attempt, attempt/'export', record)
            fp[pixels[0]] = 5
            np.savez_compressed(attempt/'mismatches.npz', fp=fp, fn=fn)
            with self.assertRaisesRegex(ValueError, 'frame count'):
                export_mismatches(attempt, attempt/'invalid', record)
            self.assertFalse((attempt/'invalid').exists())

    def test_no_frames_is_distinct_from_zero_mismatches(self):
        with tempfile.TemporaryDirectory() as temp:
            attempt = Path(temp)
            zeros = np.zeros(RAW_SHAPE, dtype=np.uint32)
            np.save(attempt/'expected-map.npy', zeros.astype(bool))
            np.savez_compressed(attempt/'mismatches.npz', fp=zeros, fn=zeros)
            export_mismatches(attempt, attempt/'empty', {'analysis': {'valid_images': 0}})
            meta = json.loads((attempt/'empty/metadata.json').read_text())
            self.assertFalse(meta['has_image_evidence'])
            self.assertEqual(meta['candidate_pixels'], 0)


class RogueIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        try:
            import rogue.interfaces.stream as ris
            import pyrogue.utilities.fileio
        except ImportError as exc:
            raise unittest.SkipTest('Rogue runtime unavailable: '+str(exc))
        cls.ris = ris

    def test_completed_failed_and_interrupted_records_keep_data_and_analysis(self):
        for status, code, error in [('completed', 0, None),
                                    ('failed', 1, TimeoutError('no more frames')),
                                    ('interrupted', 130, KeyboardInterrupt())]:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temp:
                out = Path(temp)/'attempt'
                profile = json.loads((PROFILE_DIR/'rdsrv421.json').read_text())
                profile['pcie_yaml'] = profile['camera_yaml'] = None
                source = Path(temp)/'profile.json'
                source.write_text(json.dumps(profile))

                def synthetic_capture(args, resolved, selected, outdir, record):
                    data = payload()
                    (outdir/'capture.dat').write_bytes(struct.pack('<IHBB', len(data)+4, 0, 0, 1)+data)
                    if error is not None:
                        raise error

                with patch.object(cap, 'software_snapshot', return_value={}), patch.object(cap, 'validate_hardware'), patch.object(cap, 'capture', side_effect=synthetic_capture), patch.dict(os.environ, {'SUBMODULEDIR': temp}):
                    result = cap.main(['--profile', str(source), '--mode', 'FL',
                                       '--frames', '1', '--label', 'synthetic', '--outdir', str(out)])
                self.assertEqual(result, code)
                record = json.loads((out/'test.json').read_text())
                self.assertEqual(record['status'], status)
                self.assertEqual(record['analysis']['valid_images'], 1)
                self.assertEqual(record['analysis']['false_negative_occurrences'], 0)
                self.assertEqual(record['data']['sha256'], profiles.file_sha256(out/'capture.dat'))
                self.assertTrue((out/'console.log').exists())
                self.assertEqual(record['mismatch_artifacts']['candidate_pixels'], 0)
                meta = json.loads((out/'mismatch-records/metadata.json').read_text())
                self.assertEqual(meta['source']['status'], status)
                self.assertTrue(meta['has_image_evidence'])

    def test_native_writer_and_analysis(self):
        master = self.ris.Master()
        camera = SimpleNamespace(pgpVc0=master)
        counts = cap.ImageCounts(SIZES, 10000000)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            writer, sink = cap.make_stream_writer(camera, out/'capture.dat', counts)
            try:
                for data, error in [(bytes(48), 0), (payload(), 1), (payload(), 0), (payload(40), 0)]:
                    frame = master._reqFrame(len(data), True)
                    frame.write(data, 0)
                    frame.setError(error)
                    master._sendFrame(frame)
            finally:
                sink.disarm()
                writer.close()
            self.assertEqual(counts.snapshot()['valid_images'], 2)
            self.assertEqual(counts.snapshot()['records'], 4)
            summary = cap.analyze(out/'capture.dat', np.ones(RAW_SHAPE, dtype=bool), SIZES, out)
            self.assertEqual(summary['valid_images'], 2)
            self.assertEqual(summary['invalid_records'], 2)
            self.assertEqual(summary['false_positive_occurrences'], 0)
            self.assertEqual(summary['false_negative_occurrences'], 0)
            self.assertEqual((out/'capture.dat').stat().st_size, counts.snapshot()['bytes'])
            self.assertTrue(np.all(np.load(out/'first_raw.npy') == (0x4000 | 1234)))


if __name__ == '__main__':
    unittest.main()
