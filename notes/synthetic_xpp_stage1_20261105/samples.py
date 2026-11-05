"""Representative source-only raw/calibrated distribution checks, HTTP GET only."""
import json
import os

import numpy as np

from common import ARTIFACTS, HERE, SOURCE, guard_http, wire


def fingerprint(value):
    if isinstance(value, str):
        return {'sha256': wire.sha(value.encode())}
    a = np.ascontiguousarray(value)
    return {'shape': list(a.shape), 'dtype': a.dtype.str,
            'sha256': wire.sha(memoryview(a).cast('B'))}


def stats(value):
    valid = value[np.isfinite(value)]
    return {'nonfinite': int(value.size - valid.size),
            'quantiles_0_1_50_99_100': np.quantile(valid, [0, .01, .5, .99, 1]).tolist() if valid.size else [],
            'zero_count': int(np.count_nonzero(value == 0)),
            'greater_than_5_count': int(np.count_nonzero(value > 5))}


def main():
    guard_http()
    wire.require(os.environ['PS_FAKESTEP_FLAG'] == '0', 'Fake steps enabled')
    from psana import DataSource
    import psana
    calibration = json.loads((HERE / 'calibration.json').read_text())
    source = np.load(ARTIFACTS / 'indexes/s003.npy', allow_pickle=False)
    times = source[source['service'] == 12]['timestamp']
    positions = sorted(set(map(int, np.linspace(0, len(times)-1, 16))) | {1, 2399, 2400, 97695, 97696, len(times)-2})
    requested = times[positions]
    ds = DataSource(exp='xpp101605526', run=311, dir=str(SOURCE), live=False,
                    detectors=['jungfrau1M'], timestamps=requested, batch_size=1)
    run = next(ds.runs())
    jf = run.Detector('jungfrau1M')
    checks = []
    for record in calibration['records']:
        name = record['document']['ctype']
        value, doc = jf.calibconst[name]
        wire.require(doc['_id'] == record['document']['_id'], 'Calibration selection changed')
        fp = fingerprint(value)
        wire.require(fp['sha256'] == record['sha256'], 'Live calibration payload differs from frozen selection')
        checks.append({'ctype': name, 'id': doc['_id'], **fp})
    masks = []
    for gain in range(3):
        mask = jf.raw._mask_from_status(gain_range_inds=(gain,))
        wire.require(mask.shape == (2, 512, 1024), 'Wrong mask shape')
        bad = int(np.count_nonzero(mask == 0))
        status_bad = int(np.count_nonzero(jf.calibconst['pixel_status'][0][gain]))
        wire.require(bad == status_bad, 'Mask disagrees with pixel status')
        masks.append({'gain': gain, 'masked_pixels': bad, **fingerprint(mask)})
    samples = []
    for i, evt in enumerate(run.events()):
        wire.require(i < len(requested) and evt.timestamp == requested[i], 'Unexpected filtered event order')
        raw = jf.raw.raw(evt)
        wire.require(raw is not None and raw.shape == (2, 512, 1024) and raw.dtype == np.uint16, 'Missing/wrong JF sample')
        calibrated = jf.raw.calib(evt, cversion=3, cmpars=None)
        wire.require(calibrated is not None and calibrated.shape == raw.shape, 'Missing calibrated sample')
        bits = raw >> 14
        sample = {'source_event_index': positions[i], 'timestamp': int(evt.timestamp),
                  'raw': fingerprint(raw), 'adc_stats': stats(raw & 0x3fff),
                  'gain_code_counts_0_1_2_3': np.bincount(bits.ravel(), minlength=4).tolist(),
                  'calibrated': fingerprint(calibrated), 'calibrated_stats': stats(calibrated)}
        samples.append(sample)
        print(f'Sampled source index {positions[i]}: above threshold {sample["calibrated_stats"]["greater_than_5_count"]}', flush=True)
    wire.require(len(samples) == len(requested), 'Missing requested source samples')
    report = {'status': 'pass', 'calibration_checks': checks, 'status_masks': masks,
              'sample_count': len(samples), 'samples': samples,
              'calibration_options': {'cversion': 3, 'cmpars': None},
              'threshold': 5, 'scope': 'Deterministic representative sample, not exhaustive scientific validation or common-mode equivalence.',
              'psana': psana.__file__, 'script_sha256': wire.file_sha(HERE / 'samples.py')}
    wire.save_json(HERE / 'samples.json', report)
    print('PASS representative raw/calibrated samples and frozen calibration identities', flush=True)


if __name__ == '__main__':
    main()
