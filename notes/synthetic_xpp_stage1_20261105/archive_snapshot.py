"""Preserve validated private snapshots in a new owner-only results directory."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from common import ARTIFACTS, HERE, wire

ARCHIVE = Path('/sdf/data/lcls/ds/prj/prjdatpsana/results/synthetic_xpp_stage1_private_20261105')


def main():
    complete = json.loads((HERE / 'completion_checks.json').read_text())
    wire.require(complete['status'] == 'pass', 'Completion checks required before archival')
    expected = complete['private_artifacts_verified']
    actual = {str(p.relative_to(ARTIFACTS)) for p in ARTIFACTS.rglob('*') if p.is_file()}
    wire.require(actual == set(expected), 'Unverified or missing private artifact')
    os.umask(0o077)
    ARCHIVE.mkdir(mode=0o700, exist_ok=False)
    wire.require(ARCHIVE.stat().st_uid == os.getuid() and ARCHIVE.stat().st_mode & 0o077 == 0,
                 'Archive root is not owner-only')
    rows = []
    for relative, digest in sorted(expected.items()):
        source, target = ARTIFACTS / relative, ARCHIVE / relative
        wire.require(not source.is_symlink(), 'Symlink in snapshot')
        if target.parent != ARCHIVE:
            target.parent.mkdir(mode=0o700, exist_ok=True)
        wire.require(target.parent.stat().st_mode & 0o077 == 0, 'Archive subdirectory is not private')
        with source.open('rb') as reader:
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'wb') as writer:
                while True:
                    block = reader.read(4 * 1024 * 1024)
                    if not block:
                        break
                    writer.write(block)
                writer.flush()
                os.fsync(writer.fileno())
        wire.require(target.stat().st_mode & 0o077 == 0, 'Archive file is not private')
        wire.require(wire.file_sha(target) == digest == wire.file_sha(source), 'Archive checksum mismatch')
        rows.append({'file': relative, 'bytes': target.stat().st_size, 'sha256': digest})
    with (ARCHIVE / 'SHA256SUMS').open('x') as output:
        output.writelines(row['sha256'] + '  ' + row['file'] + '\n' for row in rows)
    acl = subprocess.check_output(['getfacl', '-p', str(ARCHIVE)], text=True)
    wire.save_json(HERE / 'archive_snapshot.json', {
        'status': 'pass', 'checked_utc': datetime.now(timezone.utc).isoformat(),
        'source': str(ARTIFACTS), 'archive': str(ARCHIVE), 'owner_uid': os.getuid(),
        'mode': oct(ARCHIVE.stat().st_mode & 0o777), 'acl': acl,
        'files': rows, 'total_bytes': sum(row['bytes'] for row in rows),
        'sha256sums_sha256': wire.file_sha(ARCHIVE / 'SHA256SUMS'),
        'script_sha256': wire.file_sha(HERE / 'archive_snapshot.py'),
        'sharing': 'Owner-only archive; no existing permissions changed or collaborator access granted.',
    })
    print(f'PASS archived {len(rows)} verified private artifacts, {sum(r["bytes"] for r in rows)} bytes')


if __name__ == '__main__':
    main()
