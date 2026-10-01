"""DAQ Configure calls the standalone writer; no post-Configure pixel repair."""
import os
from pathlib import Path
import runpy
import shlex

_root = Path(os.environ['UED_GAINMAP_SRC'])
_base = runpy.run_path(str(_root/'notes/ued-full-daq-20260930/diagnostic-hook/epix_diagnostic.py'))
platform = _base['platform']
procmgr_config = []
_mask = _root/'notes/ued-commissioning-20260928/map-inputs/expand2.npy'
for _original in _base['procmgr_config']:
    _row = _original.copy()
    if _row['id'] == 'epixquad1kfps_0':
        _row['env'] += ' EPIXQUAD_STANDALONE_GAINMAP=' + shlex.quote(str(_mask))
    procmgr_config.append(_row)
