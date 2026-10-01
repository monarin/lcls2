"""Opt-in diagnostic launch; full DAQ TEST routes and normal XTC2 recording."""
import os
from pathlib import Path
import runpy
import shlex

# daqmgr executes configurations without __file__; setup.sh exports this root.
_root = Path(os.environ['UED_GAINMAP_SRC']).resolve()
_here = _root/'notes/ued-full-daq-20260930/diagnostic-hook'
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
    if _row['id'] == 'daqstat':
        # Slurm exports a selected environment; daqstat reloads this file.
        _row['env'] = (_row.get('env', '')
                       + ' UED_GAINMAP_SRC=' + shlex.quote(str(_root))
                       + ' UED_GAINMAP_DIAG_DIR=' + shlex.quote(_destination))
    procmgr_config.append(_row)
