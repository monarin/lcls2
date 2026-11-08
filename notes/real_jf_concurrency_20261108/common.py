"""Shared declarations for the bounded real-MFX comparison."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PREP = HERE.parent/'mfx_expand10_performance_20261106'
sys.path.insert(0,str(PREP))
import runner as sim_runner
from perf_cube import Cube as SimCube, dense_cube
from timing import Recorder, admission_counts
from fixture import file_sha, require

ROOT = Path('/sdf/data/lcls/ds/prj/public01/xtc')
COUNT = 1000
SAMPLES = (0,500,999)


def source_stat():
    paths = sorted(ROOT.glob('mfx101555026-r0010-*.xtc2'))
    paths += sorted((ROOT/'smalldata').glob('mfx101555026-r0010-*.smd.xtc2'))
    require(len(paths)==20,'Expected ten bigdata and ten SMD files')
    return {str(p):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in paths}
