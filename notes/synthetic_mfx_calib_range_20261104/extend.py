"""One-shot, explicitly scoped run_end update; verification mode is read-only."""
import argparse
import gc
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from psana.pscalib.calib import MDBWebUtils as wu
from psana.pscalib.calib import CalibConstants as cc

HERE = Path(__file__).resolve().parent
STAGE1 = HERE.parent / 'synthetic_mfx_stage1_20261103'
RO = 'https://pswww.slac.stanford.edu/ws/calib_ws/'
RW = 'https://pswww.slac.stanford.edu/ws-kerb/calib_ws/'
DB = 'cdb_prjdatpsana'
COLLECTIONS = ['jungfrau_000009', 'pv_000001']


def require(value, message):
    if not value:
        raise RuntimeError(message)


def save(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def snapshot():
    result = {}
    for collection in COLLECTIONS:
        response = requests.get(RO + DB + '/' + collection)
        response.raise_for_status()
        result[collection] = sorted(response.json(), key=lambda d: d['_id'])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    require(cc.URL == RO and cc.URL_KRB == RW, 'Unexpected service configuration')
    raw = (STAGE1 / 'stage2_deployment_journal.json').read_bytes()
    require(sha(raw) == 'd882f23d7b10818ac7c689afbe2196100eada365cc63e9c0b452ed3ae39ba629', 'Changed deployment journal')
    journal = json.loads(raw)
    raw = (STAGE1 / 'manifest.json').read_bytes()
    require(sha(raw) == '90a27eed3679111aa24574c09b5505be6b8775429e5b8de10dc0c9de117b983d', 'Changed manifest')
    manifest = json.loads(raw)
    expected = {r['id_doc']: r for r in journal}
    allowed = {RW + DB + '/' + r['collection'] + '/' + r['id_doc'] for r in journal}
    permitted_document = None
    original = requests.sessions.Session.request

    def guarded(self, method, url, *a, **kw):
        method = method.upper()
        require(method == 'GET' or (
            args.apply and method == 'PUT' and url in allowed
            and permitted_document is not None and kw.get('json') == permitted_document
            and url.endswith('/' + permitted_document['_id'])), 'Forbidden HTTP mutation')
        kw.setdefault('timeout', 180)
        kw['allow_redirects'] = False
        response = original(self, method, url, *a, **kw)
        require(not response.is_redirect, 'Unexpected redirect')
        response.raise_for_status()
        return response

    requests.sessions.Session.request = guarded
    before = snapshot()
    save(args.output / 'before.json', before)
    require(sum(map(len, before.values())) == 13, 'Unexpected document count')
    for collection, docs in before.items():
        require({d['_id'] for d in docs} == {r['id_doc'] for r in journal if r['collection'] == collection}, 'Unexpected document IDs')
        for doc in docs:
            ref = expected[doc['_id']]
            require(doc['experiment'] == 'prjdatpsana' and doc['detector'] == collection, 'Wrong scope')
            require(doc['id_data'] == ref['id_data'] and doc['ctype'] == ref['ctype'], 'Changed data identity')
            require(doc['run'] == 1 and doc['run_end'] == (1 if args.apply else 2), 'Unexpected starting validity; inspect before retrying')
    planned = {c: [dict(d, run_end=2) for d in ds] for c, ds in before.items()}
    save(args.output / 'planned.json', planned)
    if args.apply:
        for collection, docs in planned.items():
            for doc in docs:
                # Recheck immediately before each PUT. No automatic retry/rollback.
                live = snapshot()
                old = next(d for d in before[collection] if d['_id'] == doc['_id'])
                require(next(d for d in live[collection] if d['_id'] == doc['_id']) == old, 'Concurrent metadata change')
                permitted_document = doc
                save(args.output / (doc['_id'] + '.intent.json'), {'utc': datetime.now(timezone.utc).isoformat(), 'document': doc})
                response = wu.put(RW + DB + '/' + collection + '/' + doc['_id'], doc)
                permitted_document = None
                save(args.output / (doc['_id'] + '.response.json'), {'status': response.status_code, 'body': response.text})
                readback = snapshot()
                require(next(d for d in readback[collection] if d['_id'] == doc['_id']) == doc, 'Readback differs from planned document')
                print('Extended', collection, doc['ctype'], flush=True)
    after = snapshot()
    save(args.output / 'after.json', after)
    require(after == planned, 'Unexpected metadata changes')
    selected = []
    # Use unmodified normal psana selection and transport, not a custom selector.
    for run in (1, 2):
        for name, info in sorted(manifest['detectors'].items()):
            if not info['calibration_types']:
                continue
            constants = wu.calib_constants_all_types(info['uniqueid'], exp='prjdatpsana', run=run)
            require(set(constants) == set(info['calibration_types']), 'Wrong calibration types')
            for ctype, (data, doc) in constants.items():
                require(doc['_id'] in expected, 'Unexpected selection / shared fallback')
                ref = expected[doc['_id']]
                require(doc['ctype'] == ctype == ref['ctype'], 'Wrong calibration type')
                require(doc == next(d for d in after[ref['collection']] if d['_id'] == ref['id_doc']), 'Selected metadata differs')
                digest = sha(data.encode('utf-8') if isinstance(data, str) else memoryview(np.ascontiguousarray(data)).cast('B'))
                require(digest == ref['sha256'], 'Frozen calibration payload mismatch')
                selected.append({'run': run, 'detector': name, 'ctype': ctype, 'id_doc': doc['_id'], 'id_data': doc['id_data'], 'sha256': digest})
            del constants, data
            gc.collect()
            print('Verified run', run, name, flush=True)
    require(len(selected) == 46, 'Incomplete calibration verification')
    for docs in after.values():
        for ctype in {d['ctype'] for d in docs}:
            require(wu.select_doc_in_run_range([d for d in docs if d['ctype'] == ctype], 3) is None, 'Range extends into run 3')
    save(args.output / 'verification.json', {
        'status': 'PASS', 'utc': datetime.now(timezone.utc).isoformat(),
        'database': DB, 'run': 1, 'run_end': 2, 'document_count': 13,
        'only_metadata_change': 'run_end: 1 -> 2' if args.apply else 'none (read-only)',
        'selected': selected, 'run3_experiment_records_excluded': True,
        'script_sha256': sha(Path(__file__).read_bytes()), 'mdb_module': wu.__file__,
        'mdb_sha256': sha(Path(wu.__file__).read_bytes())})
    print('PASS: 13 records, 46 frozen-payload selections, range 1–2', flush=True)


if __name__ == '__main__':
    main()
