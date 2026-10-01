"""Opt-in diagnostic launch; full DAQ TEST routes and normal XTC2 recording."""
import os
from pathlib import Path
import runpy

_here = Path(__file__).resolve().parent
_root = _here.parents[2]
_prefix = _root/'install-ued-diag-20260930'
if Path(os.environ.get('TESTRELDIR', '/unset')) != _prefix:
    raise RuntimeError('Source diagnostic-hook/setup.sh before using this launch')
_base = runpy.run_path(str(_here.parent/'epix_gainmap_test.py'))
platform = _base['platform']
procmgr_config = []
_destination = os.environ['UED_GAINMAP_DIAG_DIR']
if not Path(_destination).is_absolute() or any(c.isspace() or c in ',;\"\'`$' for c in _destination):
    raise ValueError('Diagnostic output must be an absolute path without shell/kwarg separators')
for _original in _base['procmgr_config']:
    _row = _original.copy()
    if _row['id'] == 'epixquad1kfps_0':
        _row['cmd'] += f' -k epixDiagDir={_destination},epixDiagFrames=51'
    procmgr_config.append(_row)
