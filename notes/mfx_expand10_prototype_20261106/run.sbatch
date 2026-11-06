#!/usr/bin/env bash
#SBATCH --account=lcls
#SBATCH --partition=ampere
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
set -euo pipefail
prototype_dir=${1:?absolute prototype directory}
output=${2:?new scratch output directory}
export CUPY_CACHE_DIR=$(dirname "$output")/cupy-cache
export TMPDIR=$(dirname "$output")/tmp
mkdir -p "$CUPY_CACHE_DIR" "$TMPDIR"
exec bash "$prototype_dir/../mfx_expand10_stage2_20261105/runtime.sh" "$prototype_dir/prototype.py" --output "$output"
