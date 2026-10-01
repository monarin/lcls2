"""Offline UED external timing, operator wait, and failure cleanup checks."""
from contextlib import ExitStack, nullcontext
import copy
import json
import os
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from psdaq.debugtools.epixquad1kfps import capture_standalone as cap
from psdaq.debugtools.epixquad1kfps import standalone_hardware as hw
from psdaq.debugtools.epixquad1kfps import standalone_profile as profiles
from psdaq.debugtools.epixquad1kfps import ued_external_timing as ext

PROFILE = Path(cap.__file__).parent / 'profiles/drp-ued-cmp003-external.json'
PREFIX = 'DevPcie.Hsio.TimingRx.'


class Tree:
    """Register model with explicit read storage; missing RO values fail."""
    def __init__(self, values, events, path=''):
        self.values, self.events, self.path = values, events, path

    def __getattr__(self, key):
        return Tree(self.values, self.events, self.path + ('.' if self.path else '') + key)

    def set(self, value):
        self.events.append(('set', self.path, value))
        self.values[self.path] = int(value)

    def get(self, read=False):
        if not read:
            raise AssertionError('Timing checks must use hardware readback')
        return self.values[self.path]

    def __call__(self):
        self.events.append(('command', self.path))


def tree():
    values = {PREFIX+'TimingFrameRx.RxLinkUp': 1,
              PREFIX+'TriggerEventManager.XpmMessageAligner.RxId': 0xff000001,
              PREFIX+'TriggerEventManager.XpmMessageAligner.PartitionDelay[0]': 11}
    events = []
    return Tree(values, events), values, events


class ExternalTimingTests(unittest.TestCase):
    def test_matches_daq_delay_values_and_enables_only_two_buffers(self):
        profile = profiles.load_profile(PROFILE)
        pcie, values, events = tree()
        with patch('psdaq.cas.xpm_utils.timTxId', return_value=0xf700244f):
            result = ext.configure(pcie, profile, sleep=lambda _: None)
        manager = PREFIX+'TriggerEventManager.'
        self.assertEqual(values[manager+'TriggerEventBuffer[0].TriggerDelay'], 8577)
        self.assertEqual(values[manager+'EvrV2CoreTriggers.EvrV2TriggerReg[4].Delay'], 11186)
        self.assertEqual(values[manager+'TriggerEventBuffer[4].TriggerDelay'], 42)
        self.assertEqual(values[PREFIX+'TimingFrameRx.ClkSel'], 0)
        self.assertEqual(values[PREFIX+'TimingFrameRx.ModeSelEn'], 1)
        self.assertEqual(values[PREFIX+'TimingFrameRx.ModeSel'], 1)
        self.assertEqual(values[PREFIX+'TimingPhyMonitor.UseMiniTpg'], 0)
        self.assertEqual(result['partition_delay'], 11)
        # No master enabled during configuration; the operator may leave groupca off.
        self.assertEqual([values[manager+'TriggerEventBuffer[%d].MasterEnable'%i] for i in range(8)], [0]*8)
        ext.arm(pcie)
        self.assertEqual([values[manager+'TriggerEventBuffer[%d].MasterEnable'%i] for i in range(8)], [1,0,0,0,1,0,0,0])
        self.assertNotIn(('command', 'StartRun'), events)
        self.assertEqual(values['DevPcie.Application.AppLane[0].EventBuilder.Blowoff'], 0)

    def test_bad_link_partition_and_write_readback_fail_closed(self):
        for field, value, message in (('TimingFrameRx.RxLinkUp', 0, 'link is not up'),
                ('TriggerEventManager.XpmMessageAligner.PartitionDelay[0]', 12, 'PartitionDelay changed'),
                ('TriggerEventManager.XpmMessageAligner.RxId', 0xffffffff, 'Invalid upstream')):
            with self.subTest(field=field), patch('psdaq.cas.xpm_utils.timTxId', return_value=0xf700244f):
                pcie, values, events = tree()
                values[PREFIX+field] = value
                with self.assertRaisesRegex(RuntimeError, message):
                    ext.configure(pcie, profiles.load_profile(PROFILE), sleep=lambda _: None)
                self.assertFalse(any(e[0]=='set' and e[1].endswith('MasterEnable') and e[2] for e in events))
        bad = SimpleNamespace(register=Mock())
        bad.register.get.return_value = 0
        with self.assertRaisesRegex(RuntimeError, 'read back'):
            ext.put(bad, 'register', 1, {})

    def test_profile_rejects_internal_settings_and_wrong_route(self):
        original = profiles.load_profile(PROFILE)
        for key, value in (('timing','xpmmini'), ('lane',1), ('pcie_yaml','defaults.yml'),
                           ('camera_auto_trigger',True), ('camera_trigger_source',3)):
            profile = copy.deepcopy(original)
            profile[key] = value
            with self.assertRaises(ValueError):
                ext.validate(profile)
        profile = copy.deepcopy(original)
        profile['ued_external_timing']['readout_group'] = 1
        with self.assertRaises(ValueError):
            ext.validate(profile)

    def test_timing_setup_failure_closes_pcie_before_camera_construction(self):
        camera, pcie = Mock(), Mock()
        with patch.object(hw, 'validate_hardware'), patch.object(hw, 'probe_pcie'), \
             patch.object(hw, 'check_identity'), patch.object(hw, 'pause_triggers'), \
             patch.object(hw, 'load_modules', return_value=(camera, pcie)), \
             patch.object(ext, 'configure', side_effect=RuntimeError('bad timing')):
            with self.assertRaisesRegex(RuntimeError, 'bad timing'), hw.session(profiles.load_profile(PROFILE)):
                self.fail('must not yield')
        pcie.DevRoot.return_value.stop.assert_called_once()
        self.assertGreaterEqual(pcie.DevRoot.return_value.StopRun.call_count, 1)
        camera.Top.assert_not_called()
        self.assertIsNone(pcie.DevRoot.call_args.kwargs['yamlFileLclsII'])
        self.assertFalse(pcie.DevRoot.call_args.kwargs['standAloneMode'])


class ExternalWaitTests(unittest.TestCase):
    def test_operator_wait_is_separate_and_partial_capture_times_out(self):
        counts = Mock()
        counts.snapshot.side_effect = [dict(valid_images=n, fault=None) for n in (0,0,1,51)]
        ticks = iter((0,0,119,119.5,120.5))
        result = cap.wait_for_external_images(counts, 51, 2, 120, clock=lambda: next(ticks), sleep=lambda _: None)
        self.assertEqual(result['first_image_wait_seconds'], 119.5)
        counts.snapshot.side_effect = [dict(valid_images=n, fault=None) for n in (1,2)]
        ticks = iter((0,0,3))
        with self.assertRaisesRegex(TimeoutError, '2/51'):
            cap.wait_for_external_images(counts, 51, 2, 120, clock=lambda: next(ticks), sleep=lambda _: None)

    def test_no_full_images_and_byte_faults_cannot_wait_forever(self):
        counts = Mock()
        counts.snapshot.return_value = dict(valid_images=0, fault=None, invalid_records=100)
        ticks = iter((0,121))
        with self.assertRaisesRegex(TimeoutError, 'groupca Run'):
            cap.wait_for_external_images(counts, 51, 30, 120, clock=lambda: next(ticks))
        counts.snapshot.return_value = dict(valid_images=0, fault='Capture byte limit reached')
        with self.assertRaisesRegex(RuntimeError, 'byte limit'):
            cap.wait_for_external_images(counts, 51, 30, 120)

    def test_ready_is_visible_then_invalidated_and_cleanup_runs_for_all_outcomes(self):
        for outcome in (None, TimeoutError('no triggers'), KeyboardInterrupt()):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
                out = Path(temp)
                read_fd, write_fd = os.pipe()
                os.set_blocking(read_fd, False)
                stack.callback(os.close, read_fd)
                stack.callback(os.close, write_fd)
                camera, pcie, writer, sink = Mock(), Mock(), Mock(), Mock()
                camera.SystemRegs.TrigSrcSel.get.return_value = 0
                camera.SystemRegs.AutoTrigEn.get.return_value = False
                camera.SystemRegs.TrigEn.get.return_value = True
                camera.RdoutCore.RdoutEn.get.return_value = True
                stack.enter_context(patch.object(hw, 'runtime_snapshot', return_value={}))
                stack.enter_context(patch.object(hw, 'camera_identity', return_value={'adc_done':True, 'adc_failed':False}))
                stack.enter_context(patch.object(hw, 'load_camera_baseline'))
                pause = stack.enter_context(patch.object(hw, 'pause_triggers'))
                stack.enter_context(patch.object(cap.gain_writer, 'program'))
                stack.enter_context(patch.object(cap, 'trigger_snapshot', return_value={}))
                stack.enter_context(patch.object(ext, 'arm', return_value={'verified':True}))
                stack.enter_context(patch.object(ext, 'snapshot', return_value={'link':1}))
                stack.enter_context(patch.object(cap, 'make_stream_writer', return_value=(writer,sink)))
                def wait(*args):
                    self.assertEqual(json.loads((out/'ready.json').read_text())['state'], 'ready')
                    if outcome is not None:
                        raise outcome
                    return {'first_image_wait_seconds':1}
                stack.enter_context(patch.object(cap, 'wait_for_external_images', side_effect=wait))
                args = SimpleNamespace(max_bytes=1000000, asics=list(range(16)), frames=51,
                    timeout=30, wait_for_trigger_seconds=120, drain_timeout=1, drain_seconds=.01,
                    progress_fd=write_fd)
                record = {'gain_writer':{}}
                if outcome is None:
                    cap.acquire_phase(args, profiles.load_profile(PROFILE), [], out, record, pcie, camera, None, 360, True)
                else:
                    with self.assertRaises(type(outcome)):
                        cap.acquire_phase(args, profiles.load_profile(PROFILE), [], out, record, pcie, camera, None, 360, True)
                pcie.StartRun.assert_not_called()
                pcie.StopRun.assert_called_once()
                pause.assert_called_once()
                writer.close.assert_called_once()
                sink.disarm.assert_called_once()
                camera.SystemRegs.AutoTrigFreqHz.set.assert_not_called()
                camera.RdoutCore.RdoutEn.set.assert_called_with(False)
                self.assertEqual(json.loads((out/'ready.json').read_text())['state'], 'stopped')
                self.assertIn(b'READY FOR GROUPCA RUN', os.read(read_fd, 4096))


class ExternalCliTests(unittest.TestCase):
    def test_external_dry_run_never_opens_hardware_and_rejects_internal_rate(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, SUBMODULEDIR=temp))
            stack.enter_context(patch.object(cap, 'software_snapshot', return_value={}))
            stack.enter_context(patch.object(cap, 'output_log', return_value=nullcontext()))
            hardware = stack.enter_context(patch.object(hw, 'session', side_effect=AssertionError('hardware touched')))
            argv = ['--profile',str(PROFILE),'--mode','FH','--label','external-dry',
                    '--outdir',temp+'/dry','--dry-run','--wait-for-trigger-seconds','120']
            self.assertEqual(cap.main(argv), 0)
            record = json.loads((Path(temp)/'dry/test.json').read_text())
            self.assertEqual(record['status'], 'dry_run')
            self.assertIsNone(record['requested_rate_hz'])
            self.assertEqual(record['expected_external_rate_hz'], 360)
            self.assertFalse((Path(temp)/'dry/ready.json').exists())
            for extra in (['--rate-hz','360'], ['--wait-for-trigger-seconds','nan'],
                          ['--rate-sequence-hz','10','360']):
                with self.assertRaises(SystemExit):
                    cap.main(argv+extra)
            hardware.assert_not_called()

    def test_wrapper_selects_external_profile_and_extends_watchdog_in_dry_run(self):
        source = Path(cap.__file__).parents[4] / 'notes/ued-commissioning-20260928/run_once.py'
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            folder = Path(temp)/'notes/commissioning'
            folder.mkdir(parents=True)
            script = folder/'run_once.py'
            script.write_text(source.read_text())
            stack.enter_context(patch('sys.argv', [str(script),'--attempt','1', '--timing-mode','ued-external','--dry-run']))
            stack.enter_context(patch('socket.gethostname',return_value='offline-host'))
            launch = stack.enter_context(patch('subprocess.Popen',return_value=Mock(wait=Mock(return_value=0),poll=Mock(return_value=0))))
            restore = stack.enter_context(patch('subprocess.run',side_effect=AssertionError('no restore during dry run')))
            with self.assertRaises(SystemExit) as status:
                runpy.run_path(str(script),run_name='__main__')
            self.assertEqual(status.exception.code,0)
            command = launch.call_args.args[0]
            self.assertTrue(command[command.index('--profile')+1].endswith('drp-ued-cmp003-external.json'))
            self.assertNotIn('--rate-hz',command)
            audit = json.loads((folder/'execution-attempt01.json').read_text())
            self.assertEqual(audit['watchdog_seconds'],300)
            restore.assert_not_called()


class RogueSchemaTests(unittest.TestCase):
    def test_real_timing_register_schema_without_starting_any_root(self):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        try:
            import pyrogue
        except ImportError as exc:
            self.skipTest('Rogue runtime unavailable: '+str(exc))
        # Import register descriptions only. Never construct DevRoot, DMA, or a memory map.
        from psdaq.utils import enable_lcls2_pgp_pcie_apps  # noqa: F401
        from lcls2_pgp_fw_lib.shared import TimingRx
        schema = TimingRx(enLclsI=False, enLclsII=True, numDetectors=8)
        pcie, values, events = tree()
        with patch('psdaq.cas.xpm_utils.timTxId', return_value=0xf700244f):
            result = ext.configure(pcie, profiles.load_profile(PROFILE), sleep=lambda _: None)
        for path, value in result['registers'].items():
            register = ext.node(schema, path)
            self.assertEqual(register.mode, 'RW', path)
            if register.enum:
                self.assertIn(value, register.enum, path)
        for path in ('TimingFrameRx.RxLinkUp', 'TimingFrameRx.RxDown', 'TimingFrameRx.sofCount',
                     'TimingFrameRx.CrcErrCount', 'TimingFrameRx.RxDecErrCount',
                     'TriggerEventManager.XpmMessageAligner.RxId',
                     'TriggerEventManager.XpmMessageAligner.PartitionDelay[0]',
                     'TriggerEventManager.TriggerEventBuffer[0].TriggerCount',
                     'TriggerEventManager.TriggerEventBuffer[4].TriggerCount'):
            self.assertIn(ext.node(schema,path).mode, ('RW','RO'), path)


if __name__ == '__main__':
    unittest.main()
