#!/usr/bin/env bash
#SBATCH --account=lcls:data@ampere
#SBATCH --qos=normal
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=24G
#SBATCH --time=00:10:00
set -euo pipefail
source_dir=${1:?absolute source directory}
output=${2:?new project scratch directory}
umask 0007
node_root=${SLURM_TMPDIR:-/lscratch/$USER}
local_root="$node_root/jf-compute-overlap-${SLURM_JOB_ID}"
mkdir -p "$node_root"
mkdir -m 700 "$local_root"
mkdir -m 770 "$output"
export TMPDIR="$local_root/tmp" CUPY_CACHE_DIR="$local_root/cupy-cache"
mkdir "$TMPDIR" "$CUPY_CACHE_DIR"
sha256sum "$source_dir"/*.py "$source_dir"/*.sh "$source_dir/run.sbatch" > "$output/launch-sha256.txt"
bash "$source_dir/runtime.sh" mpiexec -n 3 --bind-to none \
    bash "$source_dir/rank.sh" "$source_dir" "$output"
/sdf/group/lcls/ds/tools/nsight-2025.3.1/bin/nsys export --type=sqlite \
    --output="$output/jf-64-repeat12-b16-s4-pool2.sqlite" "$output/jf-64-repeat12-b16-s4-pool2.nsys-rep"
