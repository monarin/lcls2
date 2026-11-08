#!/usr/bin/env bash
set -euo pipefail
source_dir=$1
output=$2
if [[ ${OMPI_COMM_WORLD_RANK:?} == 2 ]]; then
    exec /sdf/group/lcls/ds/tools/nsight-2025.3.1/bin/nsys profile \
        --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
        --output="$output/jf-64-repeat12-b16-s4-pool2" \
        python -u "$source_dir/runner.py" --output "$output"
else
    export CUDA_VISIBLE_DEVICES=''
    exec python -u "$source_dir/runner.py" --output "$output"
fi
