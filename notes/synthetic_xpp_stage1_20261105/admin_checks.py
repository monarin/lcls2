"""GET-only catalog/calibration checks and read-only destination capacity probes."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

import requests

from common import ARTIFACTS, DEST, HERE, SOURCE, guard_http, private_root, record_stat, wire


def main():
    private_root()
    guard_http()
    from krtc import KerberosTicket
    headers = KerberosTicket('HTTP@pswww.slac.stanford.edu').getAuthHeaders()
    base = 'https://pswww.slac.stanford.edu/ws-kerb/lgbk/lgbk/'

    def get(url, **kw):
        r = requests.get(url, **kw)
        r.raise_for_status()
        return r.json()

    source = get(base + 'xpp101605526/ws/311/files', headers=headers)
    wire.require(source.get('success') is True, 'Catalog request failed')
    catalog = []
    for row in source['value']:
        name = Path(row['path']).name
        if not name.endswith('.xtc2'):
            continue
        path = SOURCE / ('smalldata' if '.smd.' in name else '') / name
        wire.require(path.is_file() and path.stat().st_size == row['size'], 'Missing/wrong source catalog file')
        catalog.append({'file': str(path), 'catalog_bytes': row['size']})
    wire.require(len(catalog) == 14, 'Unexpected source file set')
    catalog_path = ARTIFACTS / 'source_catalog.json'
    if catalog_path.exists():
        wire.require(json.loads(catalog_path.read_text()) == source, 'Source catalog changed since earlier probe')
    else:
        wire.save_json(catalog_path, source)
    runs = get(base + 'prjdatpsana/ws/runs', headers=headers)
    wire.require(runs.get('success') is True, 'Destination run lookup failed')
    wire.require(not any(r['num'] in (3, 4) for r in runs['value']), 'Destination run already registered')
    disk_collisions = [str(p) for folder in (DEST, DEST / 'smalldata')
                       for n in (3, 4) for p in folder.glob(f'*r{n:04d}*.xtc2*')]
    wire.require(not disk_collisions, 'Destination files already exist')
    run_files = {}
    for n in (3, 4):
        result = get(base + f'prjdatpsana/ws/{n}/current_files_for_live_mode', headers=headers)
        wire.require(result.get('success') is True and result['value'] == [], 'Destination file registration exists')
        run_files[n] = result
    cbase = 'https://pswww.slac.stanford.edu/ws/calib_ws/cdb_prjdatpsana'
    collections = get(cbase)
    baseline = {name: sorted(get(cbase + '/' + name), key=lambda d: d['_id'])
                for name in collections if not name.startswith(('fs.', 'system.'))}
    old = json.loads((HERE.parent / 'synthetic_mfx_calib_range_20261104/execution/after.json').read_text())
    wire.require(baseline == old, 'Existing destination calibration metadata changed; inspect before planning deployment')
    wire.save_json(ARTIFACTS / 'destination_calibration_before.json', baseline)
    preserved = {str(p): record_stat(p) for folder in (DEST, DEST / 'smalldata')
                 for n in (1, 2) for p in folder.glob(f'*r{n:04d}*.xtc2')}
    wire.require(len(preserved) == 40, 'Missing run-1/2 baseline files')
    fs = os.statvfs(DEST)
    try:
        quota = subprocess.run(['weka', 'fs', 'quota', 'list', str(DEST)],
                               capture_output=True, text=True, timeout=30)
        quota_record = {'argv': quota.args, 'returncode': quota.returncode,
                        'stdout': quota.stdout, 'stderr': quota.stderr}
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        quota_record = {'error': type(error).__name__}
    wire.save_json(HERE / 'quota_probe.json', quota_record)
    acl_paths = [SOURCE, SOURCE / 'xpp101605526-r0311-s003-c000.xtc2', DEST, DEST / 'smalldata', ARTIFACTS]
    acl = subprocess.check_output(['getfacl', '-p', *map(str, acl_paths)], text=True)
    with (HERE / 'access.txt').open('x') as output:
        output.write(acl.rstrip() + '\n')
    report = {'status': 'pass_with_quota_and_sharing_prerequisites',
              'checked_utc': datetime.now(timezone.utc).isoformat(),
              'catalog_files': catalog, 'source_catalog_sha256': wire.file_sha(ARTIFACTS / 'source_catalog.json'),
              'disk_collisions': disk_collisions, 'registered_destination_files': run_files,
              'registered_runs': runs, 'filesystem_available_bytes': fs.f_bavail * fs.f_frsize,
              'directory_write_access_observed': {str(p): os.access(p, os.W_OK | os.X_OK) for p in (DEST, DEST/'smalldata')},
              'write_probe_performed': False,
              'quota': 'Read-only probe recorded separately; filesystem availability is not quota confirmation.',
              'existing_calibration_documents': sum(map(len, baseline.values())),
              'existing_calibration_matches_run1_2_snapshot': True,
              'destination_calibration_sha256': wire.file_sha(ARTIFACTS / 'destination_calibration_before.json'),
              'run1_2_files_before': preserved,
              'sharing': 'Private source; authorization to redistribute pixels/constants not established by this read-only stage.',
              'script_sha256': wire.file_sha(HERE / 'admin_checks.py')}
    wire.save_json(HERE / 'admin_checks.json', report)
    print('PASS catalog, collisions, existing constants and capacity observations; no mutations', flush=True)


if __name__ == '__main__':
    main()
