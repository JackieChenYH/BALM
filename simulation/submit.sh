#!/usr/bin/env bash
#SBATCH --job-name=balm-simulation
#SBATCH --cpus-per-task=1
#SBATCH --output=balm-%A_%a.out
set -euo pipefail

release_dir="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$release_dir"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true

python_bin="${PYTHON:-python}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    exec "$python_bin" -m simulation.run fit "$@" --task "$SLURM_ARRAY_TASK_ID"
else
    exec "$python_bin" -m simulation.run fit "$@"
fi
