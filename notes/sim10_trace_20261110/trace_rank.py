"""Launch Nsight for BD rank 2 only; retain Slurm masks until accepted runner."""
import os
import sys
from pathlib import Path

NSYS = '/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/profilers/13.2/Nsight_Systems/bin/nsys'
prefix = sys.argv[1]
program = sys.argv[2:]
if int(os.environ['SLURM_PROCID']) == 2:
    assert not Path(prefix + '.nsys-rep').exists(), 'Never overwrite a capture'
    os.environ['SIM10_NSYS_CAPTURE'] = '1'
    os.execv(NSYS, [NSYS, 'profile', '--trace=cuda,nvtx,osrt',
                   '--sample=none', '--cpuctxsw=none',
                   '--capture-range=cudaProfilerApi', '--capture-range-end=stop',
                   '--force-overwrite=false', '-o', prefix, sys.executable, *program])
os.execv(sys.executable, [sys.executable, *program])
