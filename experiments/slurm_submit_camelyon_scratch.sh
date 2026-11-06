#!/bin/bash
# Submit the from-scratch Camelyon17 cross-hospital OOD grid as a SLURM array.
# We split it across two billing accounts to halve queue time; set only
# ACCOUNT_A (or neither) if your site has a single / default account.
#
# Grid = 4 arms (raw, stainnorm, channorm, colorctrl) x 5 seeds = 20 tasks
# (array 0-19).
#   tasks 0-9   -> ACCOUNT_A
#   tasks 10-19 -> ACCOUNT_B (falls back to ACCOUNT_A)
#
# Run ON the cluster login node, from the repo root:
#   bash src/experiments/slurm_submit_camelyon_scratch.sh
# Optional: ARCH=densenet121 LR=0.01 EPOCHS=20 ACCOUNT_A=... ACCOUNT_B=... \
#           bash src/experiments/slurm_submit_camelyon_scratch.sh
#
# Site-specific resource requests, all optional and all passed through to
# sbatch (the job script itself carries no partition / GPU directives):
#   CLUSTER_PARTITION  partition name; omitted if unset, i.e. the site default.
#                      Ours was a 4-hour GPU partition.
#   CLUSTER_GPUS       --gpus spec (default: 1). Use "<gres_name>:1" if your
#                      site requires a typed GRES; ours was one 24 GB consumer
#                      GPU.
#   CLUSTER_MODULES    forwarded to the job script; see its header.
#
# The raw+caug (colour-augmentation) arm is the same grid with COLOUR_AUG=1
# exported to the job script.

set -euo pipefail

SCRATCH="${SCRATCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
JOB="$SCRATCH/src/experiments/slurm_job_camelyon_scratch.sh"
export ARCH="${ARCH:-resnet18}"
export LR="${LR:-0.01}"
export EPOCHS="${EPOCHS:-20}"
export CLUSTER_MODULES="${CLUSTER_MODULES:-}"

ACCOUNT_A="${ACCOUNT_A:-}"
ACCOUNT_B="${ACCOUNT_B:-$ACCOUNT_A}"
acct_a=(); acct_b=()
[ -n "$ACCOUNT_A" ] && acct_a=(--account="$ACCOUNT_A")
[ -n "$ACCOUNT_B" ] && acct_b=(--account="$ACCOUNT_B")

PARTITION="${CLUSTER_PARTITION:-}"
GPUS="${CLUSTER_GPUS:-1}"
res=()
[ -n "$PARTITION" ] && res+=(--partition="$PARTITION")
[ -n "$GPUS" ] && res+=(--gpus="$GPUS")

mkdir -p "$SCRATCH/logs"
cd "$SCRATCH"   # #SBATCH --output/--error are relative to the submit directory

echo "Submitting Camelyon17 from-scratch grid (arch=$ARCH lr=$LR ep=$EPOCHS) ..."
sbatch "${acct_a[@]}" "${res[@]}" --array=0-9 \
    --export=ALL,ARCH="$ARCH",LR="$LR",EPOCHS="$EPOCHS" "$JOB"
sbatch "${acct_b[@]}" "${res[@]}" --array=10-19 \
    --export=ALL,ARCH="$ARCH",LR="$LR",EPOCHS="$EPOCHS" "$JOB"

echo "Submitted. Watch with:  squeue -u \$USER"
echo "Results land in:        $SCRATCH/src/results/expfm_v_histopath_scratch/"
