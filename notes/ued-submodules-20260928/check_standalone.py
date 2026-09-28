"""Offline integration checks; never instantiate a detector or PCIe root."""
import importlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np

checkout = Path(__file__).resolve().parents[2]
bundle = Path(os.environ['SUBMODULEDIR']).resolve()
output = Path(__file__).resolve().parent
sys.path.insert(0, str(checkout / 'psdaq'))
tools = checkout / 'psdaq/psdaq/debugtools/epixquad1kfps'

# Match the standalone writer's helper import order, without calling _open_roots.
from psdaq.utils import enable_lcls2_pgp_pcie_apps, enable_epix_quad1kfps
import rogue
modules = {}
for name in ('ePixQuad', 'lcls2_pgp_pcie_apps', 'surf', 'axipcie',
             'lcls2_pgp_fw_lib', 'l2si_core', 'LclsTimingCore'):
    module = importlib.import_module(name)
    actual = Path(module.__file__).resolve()
    assert bundle in actual.parents, (name, actual)
    modules[name] = str(actual)
result = {'python': sys.version, 'python_executable': sys.executable,
          'rogue': rogue.Version.current(), 'submoduledir': str(bundle),
          'imports': modules, 'hardware_access': False}
print(json.dumps(result, indent=2))
if '--imports-only' in sys.argv:
    raise SystemExit(0)

env = dict(os.environ, PYTHONPATH=str(checkout / 'psdaq'),
           QT_QPA_PLATFORM='offscreen', PYTHONDONTWRITEBYTECODE='1')
def run(name, argv):
    completed = subprocess.run([sys.executable] + [str(x) for x in argv],
                               env=env, text=True, capture_output=True, timeout=60)
    (output / (name + '.log')).write_text(completed.stdout + completed.stderr)
    assert completed.returncode == 0, (name, completed.stdout, completed.stderr)
    return completed.stdout

with tempfile.TemporaryDirectory(prefix='ued-bundle-offline-') as temp:
    temp = Path(temp)
    for mode in ('FH', 'FM', 'FL', 'MapFHL', 'MapFML'):
        expected_path = temp / (mode + '.npy')
        args = [tools/'write_gain_mode_standalone.py', '--mode', mode,
                '--dry-run', '--save-expected-map', expected_path]
        if mode.startswith('Map'):
            args += ['--no-default-pixels', '--raw-pixel', '0,76,115']
        run('dry-run-' + mode, args)
        expected = np.load(expected_path)
        assert expected.shape == (4, 352, 384)
        if mode in ('FH', 'FM'):
            assert not expected.any()
        elif mode == 'FL':
            assert expected.all()
        else:
            assert expected.sum() == 1 and expected[0, 76, 115]
    print('Five gain-mode dry runs and saved expected maps passed')

    # Actual Rogue FileReader + existing CLI; synthetic FL records with both
    # header lengths and a short record that must not count as an image.
    from psdaq.configdb.epixquad_layout import (
        DAQ_RAW_FRAME_BYTES, ROGUE_FULL_FRAME_BASE_BYTES, RAW_SHAPE)
    datafile = temp/'synthetic.dat'
    with datafile.open('wb') as stream:
        def record(payload):
            stream.write(struct.pack('<IHBB', len(payload) + 4, 0, 0, 1))
            stream.write(payload)
        record(bytes(48))
        for extra in (0, 40):
            payload = bytearray(ROGUE_FULL_FRAME_BASE_BYTES + extra)
            start = 40 + extra
            pixels = np.full(RAW_SHAPE, 0x4000 | 1234, dtype='<u2').tobytes()
            assert len(pixels) == DAQ_RAW_FRAME_BYTES
            payload[start:start+len(pixels)] = pixels
            record(payload)
    prefix = temp/'decoded'
    stdout = run('synthetic-decode', [tools/'read_xpmmini_rogue_file.py',
                 datafile, '--max-frames', '2', '--expected-gainbit', '1',
                 '--fpfn-max-pixels', '0', '--save-prefix', prefix])
    assert 'decoded_image_frames: 2' in stdout
    assert 'decode_errors: 0' in stdout
    assert 'short_selected_records_below_image_payload: 1' in stdout
    assert stdout.count('false_positive=0 false_negative=0') == 2
    raw = np.load(str(prefix)+'_first_raw_u16.npy')
    assert raw.shape == RAW_SHAPE and np.all(raw == (0x4000 | 1234))
    print('Synthetic Rogue records: both headers decoded, short record excluded, zero FP/FN')
result['checks'] = ['imports from candidate', 'five writer dry runs',
                    'two synthetic full frames plus short record']
(output/'software-environment.json').write_text(json.dumps(result, indent=2)+'\n')
