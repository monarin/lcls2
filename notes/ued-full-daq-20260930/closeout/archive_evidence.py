"""Copy UED test evidence to scratch and independently hash every destination.

Defaults to inventory only. --execute copies; existing unequal files are never
overwritten. Data and context phases have separate manifests so bulk data can
be secured while the post-mortem is being prepared.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]
DEFAULT = Path('/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930')
FFB = Path('/sdf/data/lcls/drpsrcf/ffb/ued/uedc00107/xtc')


def digest(path, limit=None):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        while limit is None or limit > 0:
            block = handle.read(8*1024*1024 if limit is None else min(limit, 8*1024*1024))
            if not block:
                break
            h.update(block)
            if limit is not None:
                limit -= len(block)
    return h.hexdigest()


def entries(phase, run=None):
    if phase in ('data', 'pr'):
        if phase == 'pr' and run is None:
            raise ValueError('PR archive requires --run')
        pattern = 'uedc00107-r000[1-7]-*.xtc2' if phase == 'data' else f'uedc00107-r{run:04d}-*.xtc2'
        xtc = sorted(FFB.glob(pattern))
        if len(xtc) != (14 if phase == 'data' else 2):
            raise RuntimeError('Expected two streams for each selected run')
        for source in xtc:
            expected = None
            if '-s001-' in source.name:
                run = int(source.name.split('-r')[1].split('-')[0])
                report = json.loads((ROOT/f'notes/ued-full-daq-20260930/run{run}-gainbits/review.json').read_text())
                expected = dict(sha256=report['complete_prefix_sha256'], bytes=report['complete_datagram_bytes'])
                if source.stat().st_size < expected['bytes']:
                    raise RuntimeError('Finalized XTC is shorter than analyzed prefix: '+str(source))
            yield source, Path('xtc')/source.name, expected
        for folder in (('gainmap-diagnostic-20260930', 'gainmap-test-20260930') if phase == 'data' else ()):
            base = Path('/sdf/home/u/uedopr/daq')/folder
            for source in sorted(base.rglob('*')):
                if source.is_file():
                    yield source, Path('operator-evidence')/folder/source.relative_to(base), None
    else:
        for base in sorted((ROOT/'notes').glob('ued-*')):
            for source in sorted(base.rglob('*')):
                if source.is_file() and '__pycache__' not in source.parts:
                    # Receipt is produced after the immutable snapshot.
                    if source.name == 'archive-receipt.json':
                        continue
                    yield source, source.relative_to(ROOT), None
        logs = Path('/sdf/home/u/uedopr/daq/logs/2026/09')
        for source in sorted(logs.glob('30_*.log')):
            if any(x in source.name for x in (':epixquad1kfps_0.', ':control.', ':teb0.', ':timing_0.')):
                yield source, Path('daq-logs/2026/09')/source.name, None


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--phase', choices=('data', 'context', 'pr'), required=True)
    parser.add_argument('--run', type=int)
    parser.add_argument('--destination', type=Path, default=DEFAULT)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    selected = list(entries(args.phase, args.run))
    required = sum(s.stat().st_size for s, _, _ in selected)
    print(json.dumps(dict(phase=args.phase, files=len(selected), source_bytes=required,
                          destination=str(args.destination), execute=args.execute)), flush=True)
    if not args.execute:
        return
    os.umask(0o077)
    args.destination.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.destination).free < required:
        raise RuntimeError('Insufficient free space for this archive phase')
    manifest_path = args.destination/f'{args.phase}-manifest.json'
    if manifest_path.exists():
        raise FileExistsError('Phase already archived; retain immutable manifest: '+str(manifest_path))
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), phase=args.phase,
                    source_root=str(ROOT), source_commit=subprocess.check_output(
                        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), files=[])
    for source, relative, expected in selected:
        before = source.stat()
        target = args.destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            source_hash = digest(source)
        else:
            partial = target.with_name(target.name+'.copying')
            h = hashlib.sha256()
            with source.open('rb') as inp, partial.open('xb') as out:
                for block in iter(lambda: inp.read(8*1024*1024), b''):
                    h.update(block)
                    out.write(block)
                out.flush()
                os.fsync(out.fileno())
            source_hash = h.hexdigest()
            partial.rename(target)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError('Source changed during copy: '+str(source))
        target_hash = digest(target)
        if source_hash != target_hash or target.stat().st_size != before.st_size:
            raise RuntimeError('Archive verification failed: '+str(source))
        if expected and digest(target, expected['bytes']) != expected['sha256']:
            raise RuntimeError('Source differs from recorded analysis checksum: '+str(source))
        manifest['files'].append(dict(source=str(source), destination=str(relative),
                                      bytes=before.st_size, sha256=source_hash,
                                      analyzed_prefix=expected))
        print(f'verified {relative} ({before.st_size} bytes)', flush=True)
    manifest.update(completed_utc=datetime.now(timezone.utc).isoformat(),
                    file_count=len(manifest['files']), total_bytes=required, status='verified')
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    (args.destination/f'{args.phase}-SHA256SUMS').write_text(''.join(
        f"{f['sha256']}  {f['destination']}\n" for f in manifest['files']))
    print(json.dumps({k:manifest[k] for k in ('status','file_count','total_bytes','completed_utc')}), flush=True)


if __name__ == '__main__':
    main()
