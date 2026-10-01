"""Full DAQ launch for clean master + IsEn fix, without diagnostic hooks."""
import os
from pathlib import Path
import runpy
import shlex

_root = Path('/sdf/home/m/monarin/lcls2_worktree/ued-epix10ka-rdsrv421-daq')
_prefix = Path('/sdf/home/m/monarin/lcls2_worktree/ued-epixquad-isen-pr/install-ued-isen-pr')
if Path(os.environ['TESTRELDIR']) != _prefix:
    raise RuntimeError('Source closeout/setup-pr.sh before launching the PR build')
_base = runpy.run_path(str(_root/'notes/ued-full-daq-20260930/epix_gainmap_test.py'))
platform = _base['platform']
procmgr_config = []
for _original in _base['procmgr_config']:
    _row = _original.copy()
    if _row['id'] == 'epixquad1kfps_0':
        _row['env'] += ' EPIXQUAD_STANDALONE_GAINMAP='
    procmgr_config.append(_row)
