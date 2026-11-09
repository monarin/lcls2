#!/usr/bin/env bash
#SBATCH --account=lcls:data@ampere
#SBATCH --qos=normal
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --exclusive
#SBATCH --mem=64G
#SBATCH --time=00:06:00
set -euo pipefail
source_dir=${1:?source}
output=${2:?new output}
umask 0007
mkdir -m 770 "$output"
local_root="${SLURM_TMPDIR:-/lscratch/$USER}/mfx-detailed-${SLURM_JOB_ID}"
mkdir -p "$local_root/tmp" "$local_root/cupy-cache"
export TMPDIR="$local_root/tmp" CUPY_CACHE_DIR="$local_root/cupy-cache"
mapfile -t inventory < <(nvidia-smi --query-gpu=uuid,pci.bus_id,name --format=csv,noheader)
export CUDA_VISIBLE_DEVICES="${inventory[0]%%,*}"
bash "$source_dir/runtime.sh" mpiexec -n 3 --bind-to none /sdf/scratch/users/m/monarin/gpu-validation/kvikio-2610-validation-20261015-r1/venv/bin/python "$source_dir/analysis/detailed.py" --output "$output/report.json" > "$output/run.log" 2>&1
