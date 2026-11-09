#!/usr/bin/env bash
#SBATCH --account=lcls:data@ampere
#SBATCH --qos=normal
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --exclusive
#SBATCH --mem=96G
#SBATCH --time=00:35:00
set -euo pipefail
source_dir=${1:?absolute source directory}
output=${2:?new project scratch output directory}
umask 0007
node_root=${SLURM_TMPDIR:-/lscratch/$USER}
local_root="$node_root/mfx-pipeline-${SLURM_JOB_ID}"
mkdir -p "$node_root"
mkdir -m 700 "$local_root"
export TMPDIR="$local_root/tmp" CUPY_CACHE_DIR="$local_root/cupy-cache"
mkdir "$TMPDIR" "$CUPY_CACHE_DIR"
sha256sum "$source_dir"/*.py "$source_dir"/*.sh "$source_dir/run.sbatch"
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.used,utilization.gpu --format=csv
bash "$source_dir/runtime.sh" /sdf/scratch/users/m/monarin/gpu-validation/kvikio-2610-validation-20261015-r1/venv/bin/python "$source_dir/campaign.py" "$output"
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.used,utilization.gpu --format=csv
