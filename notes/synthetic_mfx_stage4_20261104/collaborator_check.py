"""Run under the collaborator's own login and activated psana environment."""
import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import platform


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True, help='Stage 4 pilot.json')
    parser.add_argument('--output', type=Path, default=Path('collaborator-report.json'))
    args = parser.parse_args()
    require(os.environ.get('PS_FAKESTEP_FLAG', '0') == '0', 'Disable fake steps first')
    os.environ['PS_FAKESTEP_FLAG'] = '0'
    os.environ['PS_PARALLEL'] = 'none'
    require(getpass.getuser() != 'monarin', 'Run this under the collaborator login, not Mona')
    require(not args.output.exists(), 'Output already exists')
    import numpy as np
    import psana
    from psana import DataSource
    reference = json.loads(args.reference.read_text())
    require(reference['status'] == 'pass' and reference['kind'] == 'pilot', 'Invalid reference')
    root = Path('/sdf/data/lcls/ds/prj/prjdatpsana/xtc')
    opened = []
    for folder, suffix in ((root, '.xtc2'), (root/'smalldata', '.smd.xtc2')):
        names = sorted(p.name for p in folder.glob('prjdatpsana-r0001-s*-c000'+suffix))
        expected = [f'prjdatpsana-r0001-s{i:03d}-c000'+suffix for i in range(10)]
        require(names == expected, 'Incomplete directory listing')
        for name in names:
            with (folder/name).open('rb') as f:
                require(len(f.read(24)) == 24, 'Cannot read XTC header')
            opened.append(str(folder/name))
    runs = 0
    for run in DataSource(exp='prjdatpsana', run=1, dir=str(root), live=False, max_events=1).runs():
        runs += 1
        require(run.expt == 'prjdatpsana' and run.runnum == 1, 'Wrong run identity')
        jf = run.Detector('jungfrau')
        ids = {k: d['_id'] for k, (_, d) in jf.calibconst.items()}
        require(ids == {r['ctype']: r['id_doc'] for r in reference['calibration'] if r['detector'] == 'jungfrau'}, 'Wrong calibration IDs')
        events = 0
        for evt in run.events():
            events += 1
            require(int(evt.timestamp) == reference['events'][0]['timestamp'], 'Wrong first event')
            a = jf.raw.calib(evt, cversion=3, cmpars=None)
            require(a is not None, 'Missing calibrated data')
            a = np.ascontiguousarray(a)
            if np.isnan(a).any():
                a = a.copy()
                a[np.isnan(a)] = np.nan
            digest = {'shape': list(a.shape), 'dtype': a.dtype.str,
                      'sha256': hashlib.sha256(memoryview(a).cast('B')).hexdigest()}
            require(digest == reference['events'][0]['calibrated']['calib'], 'Calibrated sample differs from reference')
        require(events == 1, 'Wrong event count')
    require(runs == 1, 'Wrong run count')
    record = {'status': 'pass', 'user': getpass.getuser(), 'uid': os.getuid(),
              'groups': os.getgroups(), 'host': platform.node(), 'psana': psana.__file__,
              'opened_files': opened, 'calibration_ids': ids, 'calibrated': digest,
              'PS_FAKESTEP_FLAG': os.environ['PS_FAKESTEP_FLAG'],
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'reference_sha256': hashlib.sha256(args.reference.read_bytes()).hexdigest()}
    with args.output.open('x') as f:
        json.dump(record, f, indent=2)
        f.write('\n')
    print('PASS; return', args.output, 'to Mona')


if __name__ == '__main__':
    main()
