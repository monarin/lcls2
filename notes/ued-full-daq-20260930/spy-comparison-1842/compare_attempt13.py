"""Archive and compare the new standalone capture with the saved DAQ GUI configs.

Offline only. YAML values from disabled devices can be cached, and pixel SRAM
contents are not included. Gain-bit assessment is independently recorded.
"""
import gzip
import json
import shutil
from pathlib import Path

import yaml

from compare_saved import HERE, STAND, compare, digest, flat


def main():
    source = STAND / 'attempt13-mapfhl'
    record = json.loads((source / 'test.json').read_text())
    assert record['status'] == 'completed'
    for snapshot in record['config_snapshots']:
        path = source / snapshot['path']
        assert digest(path) == snapshot['sha256']
        shutil.copyfile(path, HERE / ('attempt13-' + path.name))
    for name in ('software.json', 'console.log'):
        shutil.copyfile(source / name, HERE / ('attempt13-' + name))
    for name in ('review.json', 'after-first-mismatches.csv'):
        shutil.copyfile(source / 'review-after-first' / name, HERE / ('attempt13-' + name))
    for ext in ('json', 'log'):
        shutil.copyfile(STAND / ('restore-attempt13.' + ext), HERE / ('attempt13-restoration.' + ext))
    (HERE / 'attempt13-test.json.gz').write_bytes(gzip.compress(
        (source / 'test.json').read_bytes(), mtime=0))
    result = {'reference': 'attempt13 MapFHL, DAQ writer, internal 360 Hz request, full readout clock 3',
              'snapshot_scope': 'Prepared: read-first with camera quiescent. Armed: cached after PCIe StartRun, before camera enable.',
              'limitations': ['Disabled devices can contain cached values in both GUI and standalone YAML',
                              'Pixel SRAM and command history are not verified by SaveConfig',
                              'DAQ GUI snapshots were saved during a later unrecorded run, not simultaneously with XTC run 3'],
              'snapshot_metadata': record['config_snapshots'],
              'comparisons': {}, 'source_sha256': {}}
    pairs = [('camera', 'config_epixquad_20260930_184227.yml'),
             ('pcie', 'config_20260930_184051.yml')]
    for device, daq_name in pairs:
        standalone_path = HERE / ('attempt13-' + device + '-armed.yml')
        daq_path = HERE / daq_name
        a = flat(next(iter(yaml.safe_load(standalone_path.read_text()).values())))
        b = flat(next(iter(yaml.safe_load(daq_path.read_text()).values())))
        evidence = {key: 'attempt13 armed YAML; read-first prepared snapshot plus cached StartRun changes'
                    for key in a}
        result['comparisons'][device] = compare('attempt13-' + device, a, b, evidence)
        for path in (standalone_path, daq_path):
            result['source_sha256'][path.name] = digest(path)
    review = json.loads((HERE / 'attempt13-review.json').read_text())
    result.update(recorded_frames=review['recorded_frames'], assessed_frames=review['assessed_frames'],
                  gain_after_first=review['after_first'], stream_counts=record['stream_counts'],
                  capture_cleanup_errors=record['cleanup_errors'],
                  restoration=json.loads((HERE / 'attempt13-restoration.json').read_text()))
    result['source_sha256'][Path(__file__).name] = digest(Path(__file__))
    (HERE / 'attempt13-comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    for device, summary in result['comparisons'].items():
        print(device, summary['status_counts'])
    print('gain_after_first', result['gain_after_first'])


if __name__ == '__main__':
    main()
