"""Offline checks for profiled capture; no detector or PCIe roots are opened."""
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from psdaq.debugtools.epixquad1kfps import capture_standalone as cap
from psdaq.debugtools.epixquad1kfps import standalone_hardware as hw
from psdaq.debugtools.epixquad1kfps import standalone_profile as profiles
from psdaq.debugtools.epixquad1kfps import write_gain_mode_standalone as gain
from psdaq.configdb.epixquad_layout import RAW_SHAPE

PROFILE_DIR = Path(cap.__file__).parent / 'profiles'
SIZES = [1095248, 1095288]


def payload(extra=0):
    data = bytearray(SIZES[0] + extra)
    pixels = np.full(RAW_SHAPE, 0x4000 | 1234, dtype='<u2').tobytes()
    data[40+extra:40+extra+len(pixels)] = pixels
    return data


class ProfileTests(unittest.TestCase):
    def bench(self):
        return profiles.load_profile(PROFILE_DIR/'rdsrv421.json', bundle='/unused')

    def test_unvalidated_and_wrong_host_never_import_hardware(self):
        candidates = [profiles.load_profile(PROFILE_DIR/'drp-ued-cmp003.json'), self.bench()]
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
        self.assertIsNone(args.lane)
        self.assertIsNone(args.profile_data['camera_yaml'])

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


class CaptureTests(unittest.TestCase):
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
                               frames=1, timeout=1, drain_timeout=0.1, drain_seconds=0.01)
        profile = {'image_payload_bytes': SIZES, 'camera_yaml': 'baseline',
                   'camera_trigger_source': 3, 'camera_auto_trigger': True, 'rate_hz': 10}
        @contextmanager
        def session(*a, **k):
            yield pcie, camera
        def fail_wait(*a):
            camera.SystemRegs.TrigEn.set.side_effect = RuntimeError('disable failed')
            raise TimeoutError('no frames')
        with tempfile.TemporaryDirectory() as temp, patch.object(hw, 'session', session), patch.object(hw, 'runtime_snapshot', return_value={}), patch.object(hw, 'camera_identity', return_value={'adc_done': True, 'adc_failed': False}), patch.object(gain, '_write_gain_mode'), patch.object(cap, 'make_stream_writer', return_value=(writer, sink)), patch.object(cap, 'wait_for_images', side_effect=fail_wait):
            record = {}
            with self.assertRaisesRegex(TimeoutError, 'no frames'):
                cap.capture(args, profile, [], Path(temp), record)
        writer.close.assert_called_once()
        sink.disarm.assert_called_once()
        pcie.StopRun.assert_called_once()
        camera.SystemRegs.AutoTrigEn.set.assert_called_with(False)
        camera.RdoutCore.RdoutEn.set.assert_called_with(False)
        self.assertTrue(record['cleanup_errors'])

    def test_records_preserve_failed_attempt_and_prevent_overwrite(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(cap, 'software_snapshot', return_value={'test': True}), patch.object(cap, 'output_log', return_value=nullcontext()), patch.dict(os.environ, {'SUBMODULEDIR': temp}), patch.object(hw, 'load_modules') as load:
            dest = Path(temp)/'attempt'
            argv = ['--profile', str(PROFILE_DIR/'drp-ued-cmp003.json'), '--mode', 'FH',
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
