"""Shared read-only acceptance helpers; no calibration selection overrides."""
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import requests

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STAGE1 = REPO / 'notes/synthetic_mfx_stage1_20261103'
MANIFEST = json.loads((STAGE1 / 'manifest.json').read_text())
DEST = Path('/sdf/data/lcls/ds/prj/prjdatpsana/xtc')
SOURCE = Path(MANIFEST['streams'][0]['xtc']['path']).parent
TIMES = MANIFEST['source']['pilot_timestamps']
COUNTS = [19, 20, 21]
SAMPLES = [0, 18, 19, 38, 39, 59]
CALIB_OPTIONS = {'cversion': 3, 'cmpars': None}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def setup():
    require(os.environ.get('PS_FAKESTEP_FLAG') == '0', 'Set PS_FAKESTEP_FLAG=0 before import')
    original = requests.sessions.Session.request

    def read_only(self, method, url, *args, **kwargs):
        require(method.upper() == 'GET', 'Acceptance forbids HTTP mutations')
        kwargs.setdefault('timeout', 180)
        return original(self, method, url, *args, **kwargs)

    requests.sessions.Session.request = read_only


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(value):
    """Exact dtype/shape/value digest, normalizing only floating NaN payloads."""
    if isinstance(value, np.ndarray):
        require(value.dtype.kind != 'O', 'Object arrays are not supported')
        a = np.ascontiguousarray(value)
        if a.dtype.kind in 'fc' and np.isnan(a).any():
            a = a.copy()
            a[np.isnan(a)] = np.nan
        return {'shape': list(a.shape), 'dtype': a.dtype.str,
                'sha256': sha(memoryview(a).cast('B'))}
    if isinstance(value, np.generic):
        return fingerprint(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return {'nonfinite': str(value)}
    if isinstance(value, dict):
        return {str(k): fingerprint(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple)):
        return [fingerprint(v) for v in value]
    require(value is None or isinstance(value, (str, int, float, bool)), 'Unsupported value type: '+str(type(value)))
    return value


def payload_sha(value):
    return sha(value.encode('utf-8') if isinstance(value, str) else memoryview(np.ascontiguousarray(value)).cast('B'))


def runtime():
    import psana
    modules = ['psana', 'psana.dgram', 'psana.smdreader', 'psana.eventbuilder',
               'psana.parallelreader', 'psana.psexp.run', 'psana.psexp.event_manager',
               'psana.detector.jungfrau', 'psana.detector.UtilsJungfrau',
               'psana.pscalib.calib.MDBWebUtils']
    files = {}
    for name in modules:
        mod = sys.modules.get(name)
        if mod is not None and getattr(mod, '__file__', None):
            path = Path(mod.__file__)
            files[name] = {'path': str(path), 'sha256': sha(path.read_bytes())}
    return {'python': sys.version, 'host': platform.node(), 'psana': psana.__file__,
            'environment': {k: os.environ.get(k) for k in
                            ['PS_FAKESTEP_FLAG', 'PS_PARALLEL', 'PS_SMD_N_EVENTS',
                             'PS_SMD_CHUNKSIZE', 'PS_EB_NODES', 'PS_SRV_NODES', 'OMP_NUM_THREADS']},
            'modules': files, 'code_sha256': {p.name: sha(p.read_bytes()) for p in sorted(HERE.glob('*.py'))}}


def save(path, report):
    with Path(path).open('x') as f:
        json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def expected_step(timestamp):
    index = TIMES.index(timestamp)
    return 0 if index < 19 else 1 if index < 39 else 2


def validate_events(events):
    """Works for distributed, unordered records; rejects duplicates and wrong steps."""
    require(len(events) == len(TIMES), 'Wrong total event count')
    require(sorted(r['timestamp'] for r in events) == TIMES, 'Duplicate, omitted or foreign timestamps')
    counts = [0, 0, 0]
    for r in events:
        require(r['step'] == expected_step(r['timestamp']), 'Event assigned to wrong step')
        counts[r['step']] += 1
    require(counts == COUNTS, 'Wrong aggregate step counts')
    return counts
