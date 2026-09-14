#!/bin/bash -l
#SBATCH --job-name=mace-feat
#SBATCH --account=CHANGE_ME            # was p200717                # >>> EDIT <<< GPU project id
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --time=1-00:00:00
#SBATCH --nodes=2                        # benchmark: ~55 GPU-min total -> 4 A100 ~15 min. 1 node is plenty.
#SBATCH --ntasks-per-node=4              # one task per A100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32               # 128 cores / 4 tasks -> DataLoader workers
#SBATCH --output=mace-feat-%j.out
#SBATCH --error=mace-feat-%j.err
#SBATCH --requeue
#
# Stage 1 of the chemenv-feature study: extract per-atom MACE invariant features.
# Mirrors benchmarking/benchmark_run/submit_meluxina.sh: whole GPU nodes, one task
# per A100, each task strides the dataset by its global rank. With --nodes=2 this
# is 8 shards == 8 GPUs working in parallel; scale --nodes for more.
#
# Each task processes structures[SLURM_PROCID::SLURM_NTASKS] and writes its own
# features_shard{PROCID}.npy + meta_shard{PROCID}.parquet into OUT_DIR. Stage 2
# globs them back together -- no merge step needed.

set -euo pipefail

# ============================================================================
# >>> USER CONFIGURATION -- EDIT THIS BLOCK <<<
#
# MACE_VENV   REQUIRED. Path to the env B virtualenv DIRECTORY -- see
#             envs/envB-mace/. (torch + mace + e3nn)
#
# XYZ         INPUT FILE: the tokenised, error-annotated extxyz written by
#             stage 10.2. The default is stage 10.2's default output name in
#             its default location; if you set CHEMENV_OUT_XYZ there, set the
#             matching name here.
#
# FEATURE_MODEL  REQUIRED. The MACE .model FILE the per-atom invariant
#             descriptors are extracted from. The published analysis used
#             MACE-OMAT-0 medium (mace-omat-0-medium.model, a public foundation
#             model) -- THE SAME MODEL STAGE 10.1 EVALUATED WITH. The features
#             and the errors must come from one model to be comparable.
#             >>> USE THE PLAIN .model / *_run-1.model, NOT *_compiled.model. <<<
#             This script reaches into the module tree
#             (model.products[0].linear.irreps_out) to size the descriptor, and
#             a TorchScript-compiled model has no such attributes -- it fails
#             with a bare NotImplementedError.
#
# OUT_DIR     Where the per-rank features_shard*.npy + meta_shard*.parquet go.
#             Stage 10.4 globs them from here.
#
# DTYPE       float64 --enable-cueq is the published configuration and is what
#             the srun line below uses. WITHOUT cueq -- on CPU, or on a GPU
#             where cuequivariance is not installed -- float64 can fail inside
#             e3nn's compiled tensor products ("both inputs should have same
#             dtype"). Set DTYPE=float32 and drop --enable-cueq there; that is
#             the script's own default and it runs cleanly.
#
#     XYZ=... FEATURE_MODEL=... sbatch --account=<your allocation> submit_extract_features.sh
# ============================================================================
MACE_ENV="${MACE_VENV:?set MACE_VENV to the envB-mace virtualenv directory -- see envs/envB-mace}"
XYZ="${XYZ:-../2_analyze_chemenvs/all_structures_with_errors_and_v4chemenvs_mace_omat_medium.xyz}"
MODEL="${FEATURE_MODEL:?set FEATURE_MODEL to the MACE .model to extract features from -- the SAME model stage 10.1 used, and NOT a *_compiled.model}"
OUT_DIR="${OUT_DIR:-$SLURM_SUBMIT_DIR/mace_features}"
DTYPE="${DTYPE:-float64}"          # float32 if you are not using cueq -- see above
if [ -z "${CUEQ_FLAG+set}" ]; then CUEQ_FLAG="--enable-cueq"; fi   # CUEQ_FLAG="" disables cueq
# ============================================================================

if [ ! -f "$XYZ" ]; then
    echo "ERROR: $XYZ not found -- has stage 10.2 finished?" >&2
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "ERROR: FEATURE_MODEL=$MODEL not found." >&2
    exit 1
fi
mkdir -p "$OUT_DIR"

# --- modules / env (matches the working uv-venv launch pattern) -------------
# NOTE: this is a uv venv with its own self-contained interpreter -- do NOT
# load a Python/... module on top of it (it shadows the venv python). Just load
# CUDA, drop any active conda env, then source the venv.
module load env/release/2024.1
module load CUDA/12.6.0
conda deactivate 2>/dev/null || true     # guarded so set -e doesn't abort if no conda env
source "$MACE_ENV/bin/activate"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # avoid fragmentation OOM

# Each task pins ONE GPU via SLURM_LOCALID (its index on the node) and takes its
# shard from the global SLURM_PROCID / SLURM_NTASKS. -K0: a hard rank death (OOM,
# CUDA error) does not kill the surviving shards.
srun --kill-on-bad-exit=0 bash -c '
    export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
    python -u "'"$SLURM_SUBMIT_DIR"'/extract_mace_features.py" \
        --xyz "'"$XYZ"'" \
        --model "'"$MODEL"'" \
        --out-dir "'"$OUT_DIR"'" \
        --shard  $SLURM_PROCID \
        --nshards $SLURM_NTASKS \
        --batch-size 64 \
        --num-workers 8 \
        --dtype "'"$DTYPE"'" \
        '"$CUEQ_FLAG"'   # cueq self-check passed (max abs diff 1.8e-7); keep it on GPU
' || true

echo "All shards finished. Run Stage 2 with:"
echo "  sbatch submit_chemenv_metrics.slurm"
