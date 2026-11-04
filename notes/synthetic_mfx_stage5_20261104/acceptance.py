"""Normal calibrated DataSource: source boundary samples or complete run-2 steps."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import requests

from xtc_fixture import COUNTS, EVENTS, RUN, DEST_EXP, OLD_NOTES, file_sha, load_sources, require, save_json, schedule


def digest(value):
    if isinstance(value, str):
        return hashlib.sha256(value.encode()).hexdigest()
    array = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(array).cast('B')).hexdigest()


def fingerprint(value):
    require(value is not None, 'Missing sample data')
    a = np.ascontiguousarray(value)
    if a.dtype.kind in 'fc' and np.isnan(a).any():
        a = a.copy()
        a[np.isnan(a)] = np.nan
    return {'shape': list(a.shape), 'dtype': a.dtype.str, 'sha256': digest(a)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['source', 'destination'], required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get('PS_FAKESTEP_FLAG') == '0' and os.environ.get('PS_PARALLEL') == 'none', 'Wrong runtime flags')
    original_request = requests.sessions.Session.request

    def read_only(self, method, url, *a, **kw):
        require(method.upper() == 'GET', 'Acceptance forbids HTTP mutations')
        kw.setdefault('timeout', 180)
        return original_request(self, method, url, *a, **kw)

    requests.sessions.Session.request = read_only
    from psana import DataSource
    import psana
    sources = load_sources()
    plan = schedule(sources[0]['rows'])
    times = [r['timestamp'] for r in plan if r['service'] == 12]
    step_times = [r['timestamp'] for r in plan if r['service'] == 6]
    sample_indices = [j for step in range(len(COUNTS)) for j in (step*1000, (step+1)*1000-1)]
    sample_times = {times[i] for i in sample_indices}
    manifest = json.loads((OLD_NOTES/'manifest.json').read_text())
    deployed = {r['id_doc']:r for r in json.loads((OLD_NOTES/'stage2_deployment_journal.json').read_text())}
    frozen = {(r['detector'], r['source_document']['ctype']):r for r in manifest['calibration']}
    source_kind = args.kind == 'source'
    kwargs = {'exp':'mfx101555026' if source_kind else DEST_EXP,
              'run':10 if source_kind else RUN, 'dir':str(args.root), 'live':False,
              'batch_size':1 if source_kind else 129}
    if source_kind:
        kwargs['timestamps'] = np.array(sorted(sample_times), dtype=np.uint64)
    samples, calibration, counts = {}, [], []
    cursor = runs = 0
    for run in DataSource(**kwargs).runs():
        runs += 1
        require(run.expt == kwargs['exp'] and run.runnum == kwargs['run'], 'Wrong run identity')
        for name, info in manifest['detectors'].items():
            if not info['calibration_types']:
                continue
            constants = run.Detector(name).calibconst
            require(set(constants) == set(info['calibration_types']), 'Wrong calibration type set')
            for ctype, (data, doc) in constants.items():
                ref = frozen[name, ctype]
                expected_id = ref['source_document']['_id'] if source_kind else next(r['id_doc'] for r in deployed.values() if r['collection'] == ref['source_document']['shortname'] and r['ctype'] == ctype)
                require(doc['_id'] == expected_id and digest(data) == ref['sha256'], 'Calibration differs from frozen source')
                if not source_kind:
                    require(doc['experiment'] == DEST_EXP and doc['run'] == 1 and doc['run_end'] == 2, 'Wrong run-2 validity')
                calibration.append({'detector':name,'ctype':ctype,'id_doc':doc['_id'],'sha256':ref['sha256']})
        jf = run.Detector('jungfrau')

        def sample(evt):
            ts = int(evt.timestamp)
            if ts in sample_times:
                samples[str(ts)] = {'raw':fingerprint(jf.raw.raw(evt)),
                                    'calib':fingerprint(jf.raw.calib(evt, cversion=3, cmpars=None))}

        if source_kind:
            observed = []
            for evt in run.events():
                observed.append(int(evt.timestamp))
                sample(evt)
            require(observed == sorted(sample_times), 'Source sample timestamp mismatch')
        else:
            value, docstring = run.Detector('step_value'), run.Detector('step_docstring')
            for step_index, step in enumerate(run.steps()):
                require(step_index < len(COUNTS), 'Extra step')
                require(value(step) == step_index and int(step.evt.timestamp) == step_times[step_index], 'Wrong step identity/timestamp')
                expected_doc = {'detname':'scan','scantype':'synthetic_mfx','step':step_index,
                                'source_exp':'mfx101555026','source_run':10,
                                'source_event_range':[step_index*1000,(step_index+1)*1000]}
                require(json.loads(docstring(step)) == expected_doc, 'Wrong step provenance')
                count = 0
                for evt in step.events():
                    require(cursor < EVENTS and int(evt.timestamp) == times[cursor], 'Event omission/duplication/reordering')
                    require(value(evt) == step_index and json.loads(docstring(evt)) == expected_doc, 'Wrong event step state')
                    sample(evt)
                    cursor += 1
                    count += 1
                require(count == 1000, 'Wrong events per step')
                counts.append(count)
                print('Accepted step', step_index, 'events', count, flush=True)
            require(counts == COUNTS and cursor == EVENTS, 'Incomplete run')
        gc.collect()
    require(runs == 1 and len(calibration) == 23 and len(samples) == 60, 'Incomplete acceptance')
    reference_sha = None
    if not source_kind:
        require(args.reference is not None, 'Missing source reference')
        ref = json.loads(args.reference.read_text())
        require(ref['status'] == 'pass' and ref['kind'] == 'source' and samples == ref['samples'], 'Boundary raw/calibrated sample mismatch')
        reference_sha = file_sha(args.reference)
    save_json(args.report, {'status':'pass','kind':args.kind,'root':str(args.root),
                           'event_count':len(sample_times) if source_kind else cursor,
                           'step_counts':counts,'samples':samples,'calibration':calibration,
                           'reference_sha256':reference_sha,'script_sha256':file_sha(Path(__file__)),
                           'psana':psana.__file__,'PS_FAKESTEP_FLAG':os.environ['PS_FAKESTEP_FLAG'],
                           'PS_SMD_N_EVENTS':os.environ.get('PS_SMD_N_EVENTS')})
    print('PASS', args.kind, flush=True)


if __name__ == '__main__':
    main()
