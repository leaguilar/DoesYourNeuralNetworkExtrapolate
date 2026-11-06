#!/bin/bash
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=cam_scratch
#SBATCH --output=logs/cam_scratch_%A_%a.out
#SBATCH --error=logs/cam_scratch_%A_%a.err

# From-scratch Camelyon17 cross-hospital OOD training, one (arm, seed) per
# array task. Grid: ARMS x SEEDS, task_id -> (arm, seed).
#   arms  = raw, stainnorm (phi), channorm (stain-agnostic control)
#   seeds = 0..4
# Submit ranges/accounts from slurm_submit_camelyon_scratch.sh.
#
# --account, --array, --partition and --gpus are provided at submit time
# (sbatch CLI) rather than as #SBATCH directives here, so that nothing in this
# file is tied to one SLURM site and the grid can be split across billing
# accounts to halve queue time.
#
# Site-specific knobs, all overridable from the environment:
#   SCRATCH_ROOT     repo root on the compute node (default: two levels above
#                    this script, i.e. the checkout it was submitted from)
#   CONDA_ENV        conda env with torch + torchvision + numpy + scikit-learn
#                    (default: feateng)
#   CONDA_BIN        path to the conda executable
#                    (default: $HOME/miniforge3/bin/conda)
#   CLUSTER_MODULES  space-separated environment modules to load before running,
#                    e.g. "gcc/12.2.0 cuda/12.6" (default: none -- we assume the
#                    conda env provides the CUDA runtime). Ignored if the site
#                    has no `module` command. Our runs additionally loaded the
#                    local compiler/CUDA stack and an outbound-proxy module.
# One task needs one GPU with >=16 GB and ~20 min at the default 20 epochs.

set -euo pipefail

# Optional site software stack (see CLUSTER_MODULES above).
if [ -n "${CLUSTER_MODULES:-}" ] && command -v module >/dev/null 2>&1; then
    for _m in ${CLUSTER_MODULES}; do module load "$_m"; done
fi

SCRATCH="${SCRATCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
EXP_DIR="$SCRATCH/src/experiments"
CONDA="${CONDA_BIN:-$HOME/miniforge3/bin/conda}"
ENV="${CONDA_ENV:-feateng}"
ARCH="${ARCH:-resnet18}"
LR="${LR:-0.01}"
EPOCHS="${EPOCHS:-20}"

ARMS=(raw stainnorm channorm colorctrl)
SEEDS=(0 1 2 3 4)
N_SEEDS=${#SEEDS[@]}

arm=${ARMS[$((SLURM_ARRAY_TASK_ID / N_SEEDS))]}
seed=${SEEDS[$((SLURM_ARRAY_TASK_ID % N_SEEDS))]}

mkdir -p "$SCRATCH/logs"
cd "$SCRATCH"
echo "=== task $SLURM_ARRAY_TASK_ID -> arm=$arm seed=$seed arch=$ARCH lr=$LR ep=$EPOCHS on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

EXTRA=""
if [ "${COLOUR_AUG:-0}" = "1" ]; then EXTRA="--colour-aug"; fi

$CONDA run -n $ENV python "$EXP_DIR/train_camelyon17_scratch.py" \
    --arm "$arm" --seed "$seed" --arch "$ARCH" $EXTRA \
    --cache-dir "$SCRATCH/src/data/camelyon_cache" \
    --out "$SCRATCH/src/results/expfm_v_histopath_scratch" \
    --epochs "$EPOCHS" --batch 256 --lr "$LR" --workers 6

echo "=== task $SLURM_ARRAY_TASK_ID done ==="
