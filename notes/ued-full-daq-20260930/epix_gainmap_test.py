"""Minimal UED full-DAQ gain-map test; inherits the reviewed hutch routes."""
import hashlib
from pathlib import Path
import runpy

_source = Path('/sdf/home/u/uedopr/daq/scripts/epix.py')
if hashlib.sha256(_source.read_bytes()).hexdigest() != '0c501903261a61ee298f4bfd781ad08a512d714ea310458be519c3e427b0c0c9':
    raise RuntimeError('UED epix.py changed; review the launch routes before using this test configuration')
_base = runpy.run_path(str(_source))
platform = _base['platform']
_wanted = {'xpmpva', 'groupca', 'daqstat', 'control', 'control_gui',
           'teb0', 'timing_0', 'epixquad1kfps_0'}
procmgr_config = []
_seen = set()
for _row in _base['procmgr_config']:
    if _row['id'] in _wanted:
        _row = _row.copy()
        if _row['id'] == 'epixquad1kfps_0':
            # Match the successful Sep 25 launch on the Rocky 9 node.
            _row['cmd'] = _row['cmd'].replace('ep_domain=ens2 ', 'ep_domain=ens2np0 ')
            _row['env'] += ' EPIXQUAD_DEBUG_CONTROL_FILE=/sdf/home/m/monarin/lcls2_worktree/ued-epix10ka-rdsrv421-daq/notes/ued-full-daq-20260930/debug-disabled.json'
        if _row['id'] == 'control':
            _row['cmd'] = _row['cmd'].replace(' -C BEAM ', ' -C GAINMAP_TEST_20260930 ')
        procmgr_config.append(_row)
        _seen.add(_row['id'])
if _seen != _wanted:
    raise RuntimeError('Expected UED test processes are missing')
