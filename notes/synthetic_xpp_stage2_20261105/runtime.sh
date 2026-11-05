#!/usr/bin/env bash
set -euo pipefail
if [[ ${PS_FAKESTEP_FLAG:-0} != 0 ]]; then
    echo 'Refusing inherited nonzero PS_FAKESTEP_FLAG' >&2
    exit 2
fi
export PS_FAKESTEP_FLAG=0
export PS_PARALLEL=${PS_PARALLEL:-none}
export PS_SMD_N_EVENTS=${PS_SMD_N_EVENTS:-20}
export PS_SMD_CHUNKSIZE=${PS_SMD_CHUNKSIZE:-268435456}
export PS_SRV_NODES=0
export PS_EB_NODES=${PS_EB_NODES:-1}
export PS_EB_NODE_LOCAL=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/sdf/scratch/users/m/monarin/gpu-validation/pr187-pr208-integration-20261026-r1/B/install/lib/python3.11/site-packages
export LD_LIBRARY_PATH=/sdf/scratch/users/m/monarin/gpu-validation/pr187-pr208-integration-20261026-r1/B/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
exec /sdf/group/lcls/ds/ana/sw/conda2/inst/envs/xpp_drp_gpu_311/bin/python -u "$@"
