"""Read-only installed-software checks; no detector objects or DB connections."""
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys


root = Path(os.environ['UED_GAINMAP_SRC'])
prefix = Path(os.environ['TESTRELDIR'])
bundle = Path(os.environ['SUBMODULEDIR'])
site = prefix / 'lib/python3.9/site-packages'
modules = {}
import_warnings = {}
# DAQ services run in separate processes. Combining p4p client/server and Qt
# imports here creates teardown warnings that do not model a DAQ process.
probe = """
import importlib, sys
if sys.argv[1] in ('ePixQuad', 'lcls2_pgp_pcie_apps'):
    import psdaq.configdb.epixquad1kfps_config
module = importlib.import_module(sys.argv[1])
print('MODULE_FILE=' + module.__file__)
"""
for name in ('psdaq.configdb.epixquad1kfps_config', 'psdaq.configdb.epixquad_store_gainmap',
             'psdaq.slurm.main', 'psdaq.control.control', 'psdaq.control_gui.app.control_gui',
             'psdaq.cas.xpmpva', 'psdaq.cas.groupca', 'psdaq.slurm.daqstat',
             'ePixQuad', 'lcls2_pgp_pcie_apps'):
    result = subprocess.run([sys.executable, '-c', probe, name], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 0, (name, result.stderr)
    paths = [line.removeprefix('MODULE_FILE=') for line in result.stdout.splitlines()
             if line.startswith('MODULE_FILE=')]
    assert len(paths) == 1, (name, result.stdout)
    path = Path(paths[0]).resolve()
    if result.stderr.strip():
        import_warnings[name] = result.stderr.strip()
    expected = bundle if name in ('ePixQuad', 'lcls2_pgp_pcie_apps') else site
    assert expected in path.parents, (name, path, expected)
    modules[name] = str(path)

compared = 0
for source in (root / 'psdaq/psdaq').rglob('*.py'):
    target = site / 'psdaq' / source.relative_to(root / 'psdaq/psdaq')
    if target.is_file():
        assert source.read_bytes() == target.read_bytes(), target
        compared += 1
assert compared > 100, compared

artifacts = {}
for name in ('drp', 'teb', 'daqmgr', 'control', 'control_gui', 'xpmpva', 'groupca', 'daqstat'):
    path = prefix / 'bin' / name
    assert path.is_file() and os.access(path, os.X_OK), path
    artifacts[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if name in ('drp', 'teb'):
        linked = subprocess.check_output(['ldd', str(path)], text=True)
        assert 'not found' not in linked, linked
        assert '/install/' not in linked, linked

# Check ordinary Unix access for another account without granting permissions.
for path in [root, bundle, prefix, *map(Path, modules.values()),
             *[prefix / 'bin' / name for name in artifacts]]:
    for component in (path, *path.parents):
        bits = component.stat().st_mode
        assert bits & stat.S_IROTH, ('not world-readable', component)
        if component.is_dir() or component.parent == prefix / 'bin':
            assert bits & stat.S_IXOTH, ('not world-traversable/executable', component)

print(json.dumps({'host': socket.gethostname(), 'uid': os.getuid(),
                  'prefix': str(prefix), 'bundle': str(bundle), 'modules': modules,
                  'import_warnings': import_warnings,
                  'psdaq_python_files_equal_to_source': compared,
                  'artifact_sha256': artifacts,
                  'scope': 'DAQ imports, source comparisons, ldd and Unix modes only; no DAQ/hardware/DB access'},
                 indent=2))
