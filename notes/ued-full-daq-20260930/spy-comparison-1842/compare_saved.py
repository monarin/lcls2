"""Compare saved DAQ YAML with standalone inputs plus documented overrides.

This is an offline reconstruction, not a complete standalone hardware readback.
"""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STAND = ROOT / 'notes/ued-commissioning-20260928'
BUNDLE = Path('/sdf/home/m/monarin/lcls2_submodules_09282026_ued_epixquad_test1')


def flat(obj, prefix=''):
    result = {}
    for key, value in obj.items():
        path = prefix + str(key)
        if isinstance(value, dict):
            result.update(flat(value, path + '.'))
        else:
            result[path] = value
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(label, expected, actual, evidence):
    rows = []
    for path in sorted(expected.keys() | actual.keys()):
        if path not in expected:
            status = 'no_standalone_reference'
        elif path not in actual:
            status = 'absent_from_spy_yaml'
        else:
            status = 'same' if expected[path] == actual[path] else 'different'
        rows.append({'path': path, 'standalone': expected.get(path),
                     'daq_spy': actual.get(path), 'status': status,
                     'standalone_evidence': evidence.get(path, 'not captured')})
    with (HERE / (label + '.csv')).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, 'standalone': json.dumps(row['standalone']),
                             'daq_spy': json.dumps(row['daq_spy'])})
    return {'status_counts': {status: sum(r['status'] == status for r in rows)
                             for status in sorted({r['status'] for r in rows})},
            'differences': [r for r in rows if r['status'] == 'different']}


def main():
    camera_path = HERE / 'config_epixquad_20260930_184227.yml'
    pcie_path = HERE / 'config_20260930_184051.yml'
    baseline_camera_path = STAND / 'attempt12-mapfhl/camera_yaml.yml'
    baseline_pcie_path = STAND / 'attempt12-mapfhl/pcie_yaml.yml'
    phase_path = STAND / 'clock3-rates-20260930/attempt12-phase02-test.json.gz'
    parent_path = STAND / 'clock3-rates-20260930/test-attempt12.json.gz'
    postflight_path = STAND / 'clock3-rates-20260930/postflight.json'
    software_path = STAND / 'clock3-rates-20260930/software-attempt12.json'
    phase = json.loads(gzip.decompress(phase_path.read_bytes()))
    parent = json.loads(gzip.decompress(parent_path.read_bytes()))
    postflight = json.loads(postflight_path.read_text())
    software = json.loads(software_path.read_text())
    for name in ('capture_standalone.py', 'capture_gain_writer.py', 'standalone_hardware.py',
                 'write_gain_mode_standalone.py'):
        # Pin the historical code: capture_standalone now has optional YAML saving.
        historical = subprocess.check_output(['git', '-C', str(ROOT), 'show',
            '7377c1524:psdaq/psdaq/debugtools/epixquad1kfps/' + name])
        assert hashlib.sha256(historical).hexdigest() == software['scripts_sha256'][name]
    for key, path in (('camera_yaml', baseline_camera_path), ('pcie_yaml', baseline_pcie_path)):
        assert digest(path) == parent['inputs'][key]['sha256']
    camera = flat(yaml.safe_load(camera_path.read_text())['Top'])
    baseline = flat(yaml.safe_load(baseline_camera_path.read_text())['Top'])
    assert set(baseline) <= set(camera)
    evidence = {p: 'archived YAML loaded by standalone; not full per-register readback'
                for p in baseline}
    raw = compare('camera-input-only', baseline, camera, evidence)

    def cam_set(path, value, basis):
        baseline[path], evidence[path] = value, basis

    for path, value in postflight['camera'].items():
        target = 'SystemRegs.AsicMaskReg' if path == 'SystemRegs.AsicMask' else path
        if target in camera and not isinstance(value, dict) and not any(
                name in path for name in ('TrigEn', 'TrigSrc', 'AutoTrig', 'RdoutEn', 'SeqCount', 'BuffersRdy')):
            cam_set(target, value, 'measured standalone postflight; stable settings retained from capture')
    clock = phase['readout_clock']
    for path, key in [('AcqCore.AsicRoClkT', 'full_period_ticks'),
                      ('AcqCore.AsicRoClkHalfT', 'half_period_ticks'),
                      ('RdoutCore.AdcPipelineDelay', 'adc_pipeline_delay')]:
        cam_set(path, clock[key], 'measured standalone readout_clock readback')
    for i in range(16):
        for name, value in [('trbit', 1), ('IsEn', False), ('enable', False)]:
            cam_set(f'Epix10kaSaci[{i}].{name}', value,
                    'capture_gain_writer.program MapFHL and cleanup writes')
    for path in ('SystemRegs.TrigEn', 'RdoutCore.RdoutEn', 'SystemRegs.AutoTrigEn'):
        cam_set(path, True, 'capture_standalone.acquire_phase enable writes during acquisition')
    for name, key in [('TrigSrcSel', 'source'), ('AutoTrigPer', 'auto_period_ticks'),
                      ('AutoTrigFreqHz', 'auto_frequency_hz')]:
        cam_set('SystemRegs.' + name, phase['trigger_readback'][key],
                'measured standalone trigger_readback')
    effective_camera = compare('camera-effective', baseline, camera, evidence)

    pcie = flat(yaml.safe_load(pcie_path.read_text())['DevRoot'])
    template = flat(yaml.safe_load(baseline_pcie_path.read_text())['DevRoot'])
    expected, basis = {}, {}
    # Expand PyRogue half-open index ranges against paths present in the saved YAML.
    for path, value in template.items():
        pieces = re.split(r'(\[[0-9]*:[0-9]*\])', path)
        pattern = ''
        for piece in pieces:
            match = re.fullmatch(r'\[([0-9]*):([0-9]*)\]', piece)
            if match:
                lo, hi = match.groups()
                indices = '|'.join(str(i) for i in range(int(lo or 0), int(hi))) if hi else r'[0-9]+'
                pattern += r'\[(?:' + indices + r')\]'
            else:
                pattern += re.escape(piece)
        for target in pcie:
            if re.fullmatch(pattern, target):
                expected[target], basis[target] = value, 'archived PCIe YAML input'
    # Canonical enum labels for equivalent numeric YAML values.
    enums = {'TriggerSource': {0: 'XPM', 1: 'EVR'},
             'RateType': {0: 'FixedRates', 2: 'ControlWord'}, 'DestType': {2: 'All'}}
    for path, value in expected.items():
        expected[path] = enums.get(path.rsplit('.', 1)[-1], {}).get(value, value)

    def pci_set(path, value, source):
        expected[path], basis[path] = value, source

    timing = 'DevPcie.Hsio.TimingRx.'
    pci_set(timing + 'TimingPhyMonitor.UseMiniTpg', True, 'DevRoot startup ConfigureXpmMini code')
    pci_set(timing + 'TimingFrameRx.ModeSelEn', 'UseClkSel', 'ConfigLclsTimingBase code')
    post_enums = {**enums, 'ClkSel': {0: 'LCLS-I Clock', 1: 'LCLS-II Clock'},
                  'ModeSel': {1: 'Lcls2Protocol'}, 'ModeSelEn': {0: 'UseClkSel', 1: 'UseModeSel'},
                  'Polarity': {1: 'Rising'}}
    for path, value in postflight['pcie'].items():
        if path in pcie and not isinstance(value, dict) and not any(
                name in path for name in ('MasterEnable', 'Blowoff', 'Count', 'RxDown')):
            value = post_enums.get(path.rsplit('.', 1)[-1], {}).get(value, value)
            pci_set(path, value, 'measured standalone postflight; stable settings retained from capture')
    for i in range(8):
        pci_set(timing + f'TriggerEventManager.TriggerEventBuffer[{i}].MasterEnable', True,
                'DevRoot.StartRun enables all timing buffers; not an in-run readback')
    for i in range(4):
        lane = f'DevPcie.Application.AppLane[{i}].'
        pci_set(lane + 'VcDataTap', 0, 'DevRoot.start applies dataVc=0')
        pci_set(lane + 'EventBuilder.Blowoff', False, 'DevRoot.StartRun clears all lane blowoffs')
    lane0 = 'DevPcie.Application.AppLane[0].'
    pci_set(lane0 + 'XpmPauseThresh', 32, 'standalone_hardware.configure_lane')
    for name in ('Timeout', 'Bypass', 'Blowoff'):
        pci_set(lane0 + 'EventBuilder.' + name, phase['diagnostics_before']['event_builder'][name],
                'measured standalone in-run diagnostics_before')
    effective_pcie = compare('kcu-effective', expected, pcie, basis)
    paths = [camera_path, pcie_path, baseline_camera_path, baseline_pcie_path, phase_path, parent_path,
             postflight_path, software_path,
             ROOT / 'psdaq/psdaq/debugtools/epixquad1kfps/capture_standalone.py',
             ROOT / 'psdaq/psdaq/debugtools/epixquad1kfps/capture_gain_writer.py',
             ROOT / 'psdaq/psdaq/debugtools/epixquad1kfps/standalone_hardware.py',
             BUNDLE / 'lcls2-pgp-pcie-apps/firmware/python/lcls2_pgp_pcie_apps/_DevRoot.py',
             BUNDLE / 'lcls2-pgp-pcie-apps/firmware/submodules/lcls2-pgp-fw-lib/python/lcls2_pgp_fw_lib/shared/_TimingRx.py',
             Path(__file__)]
    result = {'scope': 'Saved DAQ YAML versus reconstructed standalone 360 Hz phase; evidence level is recorded per field',
              'reference': 'attempt12-mapfhl phase02: DAQ map writer, 360 Hz internal, no extra FP',
              'historical_capture_source_revision': '7377c1524',
              'spy_source_host': 'drp-ued-cmp003',
              'spy_source_paths': ['/tmp/' + camera_path.name, '/tmp/' + pcie_path.name],
              'camera_input_only': raw, 'camera_effective': effective_camera,
              'kcu_effective': effective_pcie,
              'source_sha256': {str(p): digest(p) for p in paths},
              'limitations': ['Standalone did not save a complete post-configuration YAML readback',
                              'Spy YAML is a saved GUI view; freshness of every field is not independently verified',
                              'Scalar YAML does not verify the ASIC pixel matrix or unrecorded command history']}
    (HERE / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    for name in ('camera_effective', 'kcu_effective'):
        print(name, json.dumps(result[name], indent=2))


if __name__ == '__main__':
    main()
