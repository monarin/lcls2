"""Select the original DAQ gain writer; standalone port remains available."""
import os
from pathlib import Path
import runpy

_root = Path(os.environ['UED_GAINMAP_SRC'])
_base = runpy.run_path(str(_root/'notes/ued-full-daq-20260930/diagnostic-hook/epix_diagnostic.py'))
platform = _base['platform']
procmgr_config = []
for _original in _base['procmgr_config']:
    _row = _original.copy()
    if _row['id'] == 'epixquad1kfps_0':
        _row['env'] += ' EPIXQUAD_STANDALONE_GAINMAP='
    procmgr_config.append(_row)
