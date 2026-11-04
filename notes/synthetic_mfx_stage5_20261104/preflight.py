"""Read-only source/schedule/calibration/storage checks for run 2."""
import json
import os
from pathlib import Path

import requests
from make_pilot import scan_blocks
from pilot_admin import preflight
from xtc_fixture import COUNTS, DEST, file_sha, load_sources, save_json, schedule, schemas, to_ns, require


def main():
    here = Path(__file__).resolve().parent
    report = preflight()
    expected = json.loads((here.parent / 'synthetic_mfx_calib_range_20261104/execution/after.json').read_text())
    for collection, docs in expected.items():
        r = requests.get('https://pswww.slac.stanford.edu/ws/calib_ws/cdb_prjdatpsana/' + collection, timeout=60)
        r.raise_for_status()
        require(sorted(r.json(), key=lambda d: d['_id']) == docs, 'Calibration metadata changed')
    report['calibration'] = 'All 13 live documents exactly match verified validity-1-2 snapshot'
    sources = load_sources()
    plan = schedule(sources[0]['rows'])
    used = set(schemas(sources[0]['bd_config'])) | set(sources[0]['schemas'])
    names_id = next(n for n in reversed(range(256)) if n not in used)
    scan, steps = scan_blocks(names_id)
    total = sum(source['rows'][r['source_index']]['bd_size'] + source['rows'][r['source_index']]['smd_size'] for source in sources for r in plan)
    total += 2 * (len(scan) + sum(map(len, steps)))
    require(report['available_bytes'] > total + 100_000_000_000, 'Insufficient filesystem headroom')
    counts = [0] * len(COUNTS)
    boundaries = []
    for r in plan:
        if r['service'] == 12:
            counts[r['step']] += 1
        if r['inserted']:
            boundaries.append(r)
    require(counts == COUNTS, 'Full schedule counts wrong')
    report.update(total_bytes=total, records_per_stream=len(plan), step_counts=counts,
                  inserted_transitions=len(boundaries), scan_names_id=names_id,
                  min_adjacent_gap_ns=min(to_ns(b['timestamp'])-to_ns(a['timestamp']) for a,b in zip(plan,plan[1:])),
                  quota='Weka quota query requires admin authentication; filesystem free bytes are not project quota proof',
                  run1_files={str(p): {'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'inode':p.stat().st_ino}
                              for folder in (DEST, DEST/'smalldata') for p in folder.glob('*r0001*.xtc2')},
                  script_sha256=file_sha(Path(__file__)))
    save_json(here/'preflight.json', report)
    save_json(here/'schedule_boundaries.json', boundaries)
    print(json.dumps({k:report[k] for k in ('total_bytes','records_per_stream','step_counts','inserted_transitions','min_adjacent_gap_ns')},indent=2))


if __name__ == '__main__':
    main()
