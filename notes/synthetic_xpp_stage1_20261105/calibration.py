"""Freeze actual run-311 psana selections and payloads using HTTP GET only."""
from datetime import datetime, timezone
import inspect
import json
import os
import platform

import numpy as np
import requests

from common import ARTIFACTS, HERE, SOURCE, guard_http, private_root, wire


def main():
    private_root()
    guard_http()
    from psana import DataSource
    from psana.pscalib.calib import CalibConstants as cc, MDBWebUtils as wu, MDBUtils as mu
    from psana.pscalib.calib import CalibDoc
    base = 'https://pswww.slac.stanford.edu/ws/calib_ws/'
    wire.require(cc.URL == base, 'Unexpected calibration endpoint')
    ds = DataSource(exp='xpp101605526', run=311, dir=str(SOURCE), live=False,
                    detectors=['jungfrau1M'], skip_calib_load='all', max_events=1)
    run = next(ds.runs())
    wire.require(run.expt == 'xpp101605526' and run.runnum == 311, 'Wrong source run')
    info = ds.dsparms.configinfo_dict['jungfrau1M']
    short = wu.pro_detector_name(info.uniqueid, add_shortname=False)
    wire.require(short is not None, 'Missing preexisting detector-name mapping')
    directory = ARTIFACTS / 'calibration'
    directory.mkdir(exist_ok=False)
    records, queries = [], []
    original_find = wu.find_docs

    def find(dbname, colname, query=None, **kw):
        result = original_find(dbname, colname, {} if query is None else query, **kw)
        queries.append({'database': dbname, 'collection': colname, 'query': query, 'documents': result})
        return result

    def snapshot(dbname, doc, **kw):
        wire.require(doc['data_type'] in ('ndarray', 'str'), 'Unsupported calibration encoding')
        response = requests.get(base + dbname + '/gridfs/' + doc['id_data'], timeout=120)
        response.raise_for_status()
        raw = response.content
        filename = f"{doc['ctype']}-{doc['_id']}.bin"
        with (directory / filename).open('xb') as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        value = mu.object_from_data_string(raw, doc)
        row = {'source_db': dbname, 'document': dict(doc), 'file': 'calibration/' + filename,
               'bytes': len(raw), 'sha256': wire.sha(raw)}
        if isinstance(value, np.ndarray):
            wire.require(value.nbytes == len(raw), 'Wrong calibration payload length')
            row.update(shape=list(value.shape), dtype=value.dtype.str,
                       nonfinite=int(np.count_nonzero(~np.isfinite(value))))
            if doc['ctype'] == 'pixel_status':
                row['nonzero_per_gain'] = [int(np.count_nonzero(x)) for x in value]
        records.append(row)
        print(f"Frozen {doc['ctype']}: {dbname}/{doc['_id']}, {len(raw)} bytes", flush=True)
        return value

    wu.find_docs = find
    wu.get_data_for_doc = snapshot
    selected = wu.calib_constants_all_types(info.uniqueid, exp='xpp101605526', run=311)
    wire.require({'pedestals', 'pixel_gain', 'pixel_status', 'pixel_offset', 'geometry'} <= set(selected),
                 'Missing expected Jungfrau calibration')
    wire.require(list(info.sorted_segment_ids) == [0, 1], 'Wrong detector segments')
    for name, (value, _) in selected.items():
        if isinstance(value, np.ndarray):
            wire.require(value.shape == (3, 2, 512, 1024), f'Unexpected {name} calibration shape')
    source_modules = {}
    for module in (wu, mu, CalibDoc):
        p = __import__('pathlib').Path(inspect.getfile(module))
        (directory / p.name).write_bytes(p.read_bytes())
        source_modules[p.name] = {'runtime_path': str(p), 'sha256': wire.file_sha(p)}
    wire.save_json(ARTIFACTS / 'calibration_queries.json', queries)
    report = {'status': 'pass', 'checked_utc': datetime.now(timezone.utc).isoformat(),
              'source_experiment': 'xpp101605526', 'source_run': 311,
              'detector': 'jungfrau1M', 'uniqueid': info.uniqueid, 'shortname': short,
              'segments': list(info.sorted_segment_ids), 'selection': 'Normal calib_constants_all_types; GET-only payload snapshot wrapper',
              'records': sorted(records, key=lambda x: x['document']['ctype']),
              'query_snapshot_sha256': wire.file_sha(ARTIFACTS / 'calibration_queries.json'),
              'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'modules': source_modules},
              'snapshot': str(ARTIFACTS), 'payload_bytes': sum(r['bytes'] for r in records),
              'planned_destination': {'database': 'cdb_prjdatpsana', 'collection': short, 'run': 3, 'run_end': 4},
              'deployed': False, 'script_sha256': wire.file_sha(HERE / 'calibration.py')}
    for row in records:
        wire.require(wire.file_sha(ARTIFACTS / row['file']) == row['sha256'], 'Frozen payload changed')
    wire.save_json(HERE / 'calibration.json', report)
    print('PASS frozen calibration selection; no DB writes', flush=True)


if __name__ == '__main__':
    main()
