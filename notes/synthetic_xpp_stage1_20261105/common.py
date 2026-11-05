"""Read-only Stage 1 helpers, with the previously reviewed wire parser pinned."""
import importlib.util
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
WIRE_PATH = HERE.parent / 'synthetic_mfx_stage5_20261104/xtc_fixture.py'
spec = importlib.util.spec_from_file_location('frozen_mfx_wire', WIRE_PATH)
wire = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wire)
wire.require(wire.file_sha(WIRE_PATH) == 'eeaf8bbac0a713796cb84f6e0158877fb579d08792c1e1a15175548f9fc2ff0a',
             'Previously reviewed wire helper changed')
SOURCE = Path('/sdf/data/lcls/ds/xpp/xpp101605526/xtc')
DEST = Path('/sdf/data/lcls/ds/prj/prjdatpsana/xtc')
ARTIFACTS = Path('/sdf/scratch/users/m/monarin/gpu-validation/synthetic-xpp/stage1-20261105')
SELECTED = (2, 3, 5, 6)
SOURCE_EVENTS = 151152


def record_stat(path):
    s = path.stat()
    return {'path': str(path), 'size': s.st_size, 'mtime_ns': s.st_mtime_ns,
            'inode': s.st_ino, 'device': s.st_dev}


def check_stat(record):
    wire.require(record_stat(Path(record['path'])) == record, 'Source file changed: ' + record['path'])


def guard_http():
    import requests
    original = requests.sessions.Session.request

    def read_only(self, method, url, *args, **kwargs):
        wire.require(method.upper() == 'GET', 'Stage 1 forbids HTTP mutations')
        kwargs.setdefault('timeout', 120)
        return original(self, method, url, *args, **kwargs)

    requests.sessions.Session.request = read_only


def private_root():
    os.umask(0o077)
    ARTIFACTS.mkdir(parents=True, exist_ok=True, mode=0o700)
    wire.require(ARTIFACTS.stat().st_mode & 0o077 == 0, 'Snapshot root is not private')
    return ARTIFACTS
