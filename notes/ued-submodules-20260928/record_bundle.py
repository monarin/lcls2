"""Record exact recursive Git pins and source patches without touching hardware."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('bundle', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
bundle = args.bundle.resolve()

def git(repo, *argv):
    return subprocess.run(['git', '-C', str(repo), *argv], check=True,
                          stdout=subprocess.PIPE, text=True).stdout

repos = []
def record(repo, expected=None):
    assert (repo/'.git').exists(), 'Uninitialized submodule: '+str(repo)
    sha = git(repo, 'rev-parse', 'HEAD').strip()
    if expected:
        assert sha == expected, (repo, sha, expected)
    changes = git(repo, 'diff', '--binary', '--ignore-submodules=all', 'HEAD')
    item = {'path': str(repo.relative_to(bundle)) or '.', 'commit': sha,
            'source_diff': changes}
    repos.append(item)
    for line in git(repo, 'ls-tree', '-r', 'HEAD').splitlines():
        fields, name = line.split('\t', 1)
        mode, kind, child_sha = fields.split()
        if mode == '160000':
            record(repo/name, child_sha)

record(bundle)
manifest = {
    'bundle': str(bundle), 'repositories': repos,
    'patch_sh_sha256': hashlib.sha256((bundle/'patch.sh').read_bytes()).hexdigest(),
    'hardware_tested': False,
}
args.output.write_text(json.dumps(manifest, indent=2)+'\n')
changed = [r['path'] for r in repos if r['source_diff']]
print('%d repositories pinned; %d repositories with source patches' %
      (len(repos), len(changed)))
for path in changed:
    print('  '+path)
