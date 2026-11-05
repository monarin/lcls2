"""Interpret completed audit results and verify frozen artifacts; no live HTTP."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from common import ARTIFACTS, HERE, SOURCE_EVENTS, check_stat, wire

AUDITED_COMMIT = '343f26bc3d7f8b0760744a427115731668e50453'


def report(name):
    return json.loads((HERE / (name + '.json')).read_text())


def main():
    inventory, audit = report('inventory'), report('jungfrau_audit')
    calibration, admin, layout = report('calibration'), report('admin_checks'), report('layout')
    wire.require(audit['events'] == SOURCE_EVENTS and audit['issue_events'] == 0
                 and audit['damage_histogram'] == {'0': SOURCE_EVENTS}
                 and audit['segment_presence'] == {'(0, 1)': SOURCE_EVENTS},
                 'Source has damaged/missing frames: explicit disposition required')
    code = subprocess.check_output(['git', 'show', AUDITED_COMMIT + ':' +
                                   'notes/synthetic_xpp_stage1_20261105/audit_jungfrau.py'], cwd=HERE)
    wire.require(wire.sha(code) == audit['script_sha256'] == wire.file_sha(HERE / 'audit_jungfrau.py'),
                 'Audit was not produced by the reviewed script')
    hashes = {}

    def verify(relative, digest):
        path = ARTIFACTS / relative
        actual = wire.file_sha(path)
        wire.require(actual == digest, 'Frozen artifact changed: ' + str(path))
        hashes[str(relative)] = actual

    for entry in inventory['streams']:
        check_stat(entry['bigdata'])
        check_stat({k: v for k, v in entry['smd'].items() if k != 'sha256'})
        verify('smd/' + Path(entry['smd']['path']).name, entry['smd']['sha256'])
        verify(f"indexes/s{entry['stream']:03d}.npy", entry['index_sha256'])
        verify(f"config/s{entry['stream']:03d}.bd.bin", entry['bd_config_sha256'])
        for transition in entry['transitions']:
            verify(transition['file'], transition['sha256'])
    for row in calibration['records']:
        verify(row['file'], row['sha256'])
    for name, module in calibration['runtime']['modules'].items():
        verify('calibration/' + name, module['sha256'])
    for plan in layout['plans']:
        verify(Path(plan['mapping_path']).relative_to(ARTIFACTS), plan['mapping_sha256'])
    for relative, digest in (
        ('calibration_queries.json', calibration['query_snapshot_sha256']),
        ('source_catalog.json', admin['source_catalog_sha256']),
        ('destination_calibration_before.json', admin['destination_calibration_sha256']),
        ('jungfrau_issues.json', audit['issues_sha256']),
        ('jungfrau_block_hashes.json', audit['source_block_hashes_sha256']),
    ):
        verify(relative, digest)
    issues = json.loads((ARTIFACTS / 'jungfrau_issues.json').read_text())
    blocks = json.loads((ARTIFACTS / 'jungfrau_block_hashes.json').read_text())
    wire.require(issues == [], 'Nonempty issue artifact')
    source_size = inventory['streams'][3]['bigdata']['size']
    wire.require(blocks['file_size'] == audit['source_bytes_read_and_hashed'] == source_size,
                 'Wrong audited source size')
    wire.require(sorted(map(int, blocks['hashes'])) == list(range(0, source_size, blocks['block_bytes'])),
                 'Source block manifest has holes')
    for record in admin['run1_2_files_before'].values():
        check_stat(record)
    wire.require(report('transition_checks')['status'] == 'pass', 'Transition comparison incomplete')
    for name in ('inventory', 'calibration', 'samples', 'layout', 'transition_checks'):
        entry = report(name)
        wire.require(entry['status'] == 'pass', 'Required check did not pass: ' + name)
        for filename, digest in entry.get('code_sha256', {}).items():
            wire.require(wire.file_sha(HERE / filename) == digest, 'Script changed: ' + filename)
        if 'script_sha256' in entry:
            filename = 'plan_layout.py' if name == 'layout' else name + '.py'
            wire.require(wire.file_sha(HERE / filename) == entry['script_sha256'], 'Script changed: ' + filename)
    original_admin = subprocess.check_output(['git', 'show', AUDITED_COMMIT + ':' +
                                             'notes/synthetic_xpp_stage1_20261105/admin_checks.py'], cwd=HERE)
    wire.require(wire.sha(original_admin) == admin['script_sha256'], 'Admin evidence provenance mismatch')
    result = {
        'status': 'pass', 'checked_utc': datetime.now(timezone.utc).isoformat(),
        'full_jungfrau_audit': 'pass: 151152 complete two-segment frames; zero XTC damage/missing-frame issues',
        'audit_code_commit': AUDITED_COMMIT, 'audit_script_sha256': audit['script_sha256'],
        'issue_policy': 'Any damage/missing-segment event requires explicit disposition before generation; no silent dropping or replacement.',
        'source_files_unchanged': 14, 'run1_2_file_stats_unchanged': 40,
        'source_bytes_block_hashed': source_size, 'source_hash_blocks': len(blocks['hashes']),
        'private_artifacts_verified': hashes,
        'administrative_evidence': 'Original read-only probe preserved; only access.txt creation mode changed afterward. No administrative operation replayed.',
        'script_sha256': wire.file_sha(HERE / 'completion_checks.py'),
    }
    wire.save_json(HERE / 'completion_checks.json', result)
    print(f'PASS full audit interpretation, {len(hashes)} frozen artifacts, source and run-1/2 stat preservation')


if __name__ == '__main__':
    main()
