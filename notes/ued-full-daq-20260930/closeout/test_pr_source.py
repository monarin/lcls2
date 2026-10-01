"""Replay the clean PR branch's actual Configure functions without hardware."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import runpy
import subprocess

import numpy as np

HERE = Path(__file__).resolve().parent
PR = Path('/sdf/home/m/monarin/lcls2_worktree/ued-epixquad-isen-pr')
namespace = runpy.run_path(str(HERE.parent/'gain-writer-diagnosis/replay.py'))
source = PR/'psdaq/psdaq/configdb/epixquad1kfps_config.py'
namespace['functions'].__globals__['SOURCE'] = source
functions = namespace['functions']()
cfg = json.loads(namespace['CONFIG'].read_text())
expected = np.where(np.load(namespace['MASK'], allow_pickle=False), 8, 12)
results = {}
for write in (True, False):
    recorder = namespace['Recorder'](cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        functions['config_expert'](recorder.base, copy.deepcopy(cfg), writePixelMap=write)
    result = recorder.result(expected)
    assert result['isen_history'] == ([[1,0,1,0]]*16 if write else [[1,0,0]]*16)
    if write:
        assert result['writes'] == {'WriteMatrixData:IsEn=1':16, 'WritePixelData:IsEn=1':22169}
        assert result['ideal_map_mismatches'] == 0
        baseline = json.loads((HERE.parent/'gain-writer-diagnosis/replay.json').read_text())
        assert result['map_register_trace_sha256'] == baseline['results']['original_daq_configure']['map_register_trace_sha256']
    else:
        assert result['writes'] == {}
    results[str(write)] = result
print(json.dumps(dict(status='passed', source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                     pr_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PR,text=True).strip(),
                     scope='Offline full Configure with saved TEST config; original map command order, programming IsEn and cleanup',
                     results=results), indent=2))
