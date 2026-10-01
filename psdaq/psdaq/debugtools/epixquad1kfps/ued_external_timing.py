"""UED lane-0 external timing for headless recording; no EPICS writes.

Mirrors the 119M path and dual trigger buffers in epixquad1kfps_config.py.
The operator owns XPM groupca Run. Importing this module opens no devices.
"""
import math
import time


def enabled(profile):
    return profile.get('ued_external_timing') is not None


def validate(profile):
    if not enabled(profile):
        return
    cfg = profile['ued_external_timing']
    if (profile['timing'] != 'external' or profile['lane'] != 0 or profile['data_vc'] != 0 or
            profile['board_type'] != 'XilinxKcu1500' or not profile['pgp4'] or
            profile['device'] != '/dev/datadev_0' or
            profile['hosts'] != ['drp-ued-cmp003'] or profile['pcie_yaml'] is not None or
            profile['camera_trigger_source'] != 0 or profile['camera_auto_trigger'] is not False):
        raise ValueError('UED external timing requires the dedicated cmp003 lane-0 profile without PCIe YAML')
    # These are the reviewed UED route and run-buffer values, not generic defaults.
    if cfg.get('readout_group') != 0 or cfg.get('run_buffer_delay') != 42:
        raise ValueError('UED external timing requires group 0 and run buffer delay 42')
    for key in ('start_ns', 'expected_partition_delay'):
        value = cfg.get(key)
        if type(value) is not int or value < 0:
            raise ValueError('Invalid UED external timing field: ' + key)
    if not math.isfinite(float(cfg['start_ns'])) or cfg['start_ns'] > 1e9:
        raise ValueError('Invalid UED start_ns')


def node(root, path):
    for part in path.split('.'):
        root = getattr(root, part)
    return root


def put(root, path, value, readback):
    register = node(root, path)
    register.set(value)
    actual = int(register.get(read=True))
    readback[path] = actual
    if actual != int(value):
        raise RuntimeError('%s: requested %s, read back %s' % (path, value, actual))


def configure(pcie, profile, sleep=time.sleep):
    """Configure while all buffers are disabled, before opening the camera."""
    from psdaq.cas.xpm_utils import timTxId
    validate(profile)
    cfg = profile['ued_external_timing']
    rx = pcie.DevPcie.Hsio.TimingRx
    readback = {}
    for index in range(8):
        put(rx, 'TriggerEventManager.TriggerEventBuffer[%d].MasterEnable' % index, 0, readback)
    # DevRoot.start selects external LCLS-II timing first, just as DAQ init does.
    # Apply DAQ's subsequent 119M clock override and link reset here.
    for path, value in (('TimingPhyMonitor.UseMiniTpg', 0),
                        ('XpmMiniWrapper.XpmMini.HwEnable', 0),
                        ('TimingFrameRx.ModeSelEn', 1), ('TimingFrameRx.ModeSel', 1),
                        ('TimingFrameRx.ClkSel', 0)):
        put(rx, path, value, readback)
    rx.TimingPhyMonitor.TxPhyPllReset()
    sleep(1)
    rx.TimingFrameRx.C_RxReset()
    sleep(2)
    rx.TimingFrameRx.RxDown.set(0)
    if int(rx.TimingFrameRx.RxLinkUp.get(read=True)) != 1:
        raise RuntimeError('UED external timing link is not up')
    aligner = rx.TriggerEventManager.XpmMessageAligner
    rx_id = int(aligner.RxId.get(read=True))
    if rx_id in (0, 0xffffffff):
        raise RuntimeError('Invalid upstream XPM timing ID: %#x' % rx_id)
    tx_id = timTxId('epixquad')
    put(rx, 'TriggerEventManager.XpmMessageAligner.TxId', tx_id, readback)
    partition_delay = int(getattr(aligner, 'PartitionDelay[%d]' % cfg['readout_group']).get(read=True))
    if partition_delay != cfg['expected_partition_delay']:
        raise RuntimeError('PartitionDelay changed from reviewed UED value: %d (expected %d)' %
                           (partition_delay, cfg['expected_partition_delay']))
    # Same clock period, message period and +9 correction as DAQ Configure.
    run_delay = int(cfg['start_ns'] / (1000 / 119.))
    daq_delay = int(cfg['start_ns'] / (1000 / 119.) - partition_delay * 238 + 9)
    if not 0 <= daq_delay <= 0xffffffff or not 0 <= run_delay <= 0xffffffff:
        raise ValueError('Computed UED trigger delay is out of range')
    prefix = 'TriggerEventManager.'
    settings = {
        'TriggerEventBuffer[0].TriggerSource': 0,
        'TriggerEventBuffer[0].Partition': cfg['readout_group'],
        'TriggerEventBuffer[0].PauseThreshold': 16,
        'TriggerEventBuffer[0].TriggerDelay': daq_delay,
        'TriggerEventBuffer[4].TriggerSource': 1,
        'TriggerEventBuffer[4].Partition': cfg['readout_group'],
        'TriggerEventBuffer[4].PauseThreshold': 16,
        'TriggerEventBuffer[4].TriggerDelay': cfg['run_buffer_delay'],
        'EvrV2CoreTriggers.EvrV2ChannelReg[4].EnableReg': 1,
        'EvrV2CoreTriggers.EvrV2ChannelReg[4].RateType': 2,
        'EvrV2CoreTriggers.EvrV2ChannelReg[4].RateSel': 6,
        'EvrV2CoreTriggers.EvrV2ChannelReg[4].DestType': 2,
        'EvrV2CoreTriggers.EvrV2ChannelReg[4].DestSel': 0,
        'EvrV2CoreTriggers.EvrV2TriggerReg[4].EnableTrig': 1,
        'EvrV2CoreTriggers.EvrV2TriggerReg[4].Source': 4,
        'EvrV2CoreTriggers.EvrV2TriggerReg[4].Polarity': 1,
        'EvrV2CoreTriggers.EvrV2TriggerReg[4].ComplEn': 0,
        'EvrV2CoreTriggers.EvrV2TriggerReg[4].ComplAnd': 0,
        'EvrV2CoreTriggers.EvrV2TriggerReg[4].Width': 1,
        'EvrV2CoreTriggers.EvrV2TriggerReg[4].Delay': run_delay,
    }
    for path, value in settings.items():
        put(rx, prefix + path, value, readback)
    return dict(registers=readback, rx_id=rx_id, tx_id=tx_id,
                partition_delay=partition_delay, start_ns=cfg['start_ns'],
                scope='UED external timing only; groupca/XPM PVs untouched')


def arm(pcie):
    """Arm only lane 0 and buffers 0/4; never call all-buffer DevRoot.StartRun."""
    pcie.CountReset()
    event = getattr(pcie.DevPcie.Application, 'AppLane[0]').EventBuilder
    event.Blowoff.set(False)
    event.SoftRst()
    manager = pcie.DevPcie.Hsio.TimingRx.TriggerEventManager
    result = {}
    for index in range(8):
        put(manager, 'TriggerEventBuffer[%d].MasterEnable' % index, index in (0, 4), result)
    pcie.RunState.set(True)
    return result


def snapshot(pcie):
    """Read real timing/trigger counters, including confirmation of a live link."""
    rx = pcie.DevPcie.Hsio.TimingRx
    paths = ['TimingFrameRx.RxLinkUp', 'TimingFrameRx.RxDown', 'TimingFrameRx.sofCount',
             'TimingFrameRx.CrcErrCount', 'TimingFrameRx.RxDecErrCount']
    paths += ['TriggerEventManager.TriggerEventBuffer[%d].%s' % (i, field)
              for i in (0, 4) for field in ('MasterEnable', 'TriggerCount')]
    result = {path: int(node(rx, path).get(read=True)) for path in paths}
    if result['TimingFrameRx.RxLinkUp'] != 1:
        raise RuntimeError('UED external timing link went down')
    return result
