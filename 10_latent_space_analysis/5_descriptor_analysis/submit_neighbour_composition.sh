#!/bin/bash -l
#SBATCH --job-name=neighbour-composition
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1                     # exact kNN on the GPU (faiss-gpu); seconds instead of ~25 min
#SBATCH --mem=0                          # ~9 GB peak (tokens + 50-d projection; features streamed)
#SBATCH --time=01:00:00
#SBATCH --account=CHANGE_ME
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --output=neighbour-composition-%j.out
#SBATCH --error=neighbour-composition-%j.err
#
# Stage 10.5b: composition of the latent-space neighbours that carry a different
# descriptor (SI S5.1: 99.9 % same central element, 69.7 % same ligand-element
# set, 56.9 % same coordination number). Same env B venv as stage 10.4.
# A 14-core workstation without a GPU also runs it (KNN_BACKEND=faiss, ~30 min).
# summary_stats.py needs no job: it reads stage 10.4's outputs in seconds.

set -euo pipefail

# ============================================================================
# >>> USER CONFIGURATION -- EDIT THIS BLOCK <<<
#
# MACE_VENV   REQUIRED. The env B virtualenv DIRECTORY used for stage 10.4
#             (it has faiss-gpu-cu12 and pyarrow).
#
# FEAT_DIR    INPUT DIRECTORY holding stage 10.3's features_shard*.npy and
#             meta_shard*.parquet. Default: stage 10.3's default output directory.
#
# REPORT_CARD OPTIONAL stage 10.4 report card; used only to check that the
#             characterised descriptor set is the same. Empty = skip the check.
#
# KNN_BACKEND faiss-gpu | faiss. Both are exact; only the speed differs.
#
#     FEAT_DIR=... sbatch --account=<your allocation> submit_neighbour_composition.sh
# ============================================================================
MACE_ENV="${MACE_VENV:?set MACE_VENV to the envB-mace virtualenv directory -- see envs/envB-mace}"
FEAT_DIR="${FEAT_DIR:-../3_extract_features/mace_features}"   # stage 10.3 output
REPORT_CARD="${REPORT_CARD:-../4_cluster_metrics/report_card.parquet}"
KNN_BACKEND="${KNN_BACKEND:-faiss-gpu}"
OUT_JSON="${OUT_JSON:-$SLURM_SUBMIT_DIR/neighbour_composition.json}"
# ============================================================================

module load env/release/2024.1
conda deactivate 2>/dev/null || true
source "$MACE_ENV/bin/activate"

if ! compgen -G "$FEAT_DIR/features_shard*.npy" > /dev/null; then
    echo "ERROR: no feature shards under $FEAT_DIR -- has stage 10.3 finished?" >&2
    exit 1
fi
RC_ARG=()
if [ -n "$REPORT_CARD" ] && [ -f "$REPORT_CARD" ]; then
    RC_ARG=(--report-card "$REPORT_CARD")
fi

# --var-keep / --max-dims must equal stage 10.4's; --seed 3 is the published draw.
python -u "$SLURM_SUBMIT_DIR/neighbour_composition.py" \
    --feature-glob "$FEAT_DIR/features_shard*.npy" \
    --meta-glob    "$FEAT_DIR/meta_shard*.parquet" \
    ${RC_ARG[@]+"${RC_ARG[@]}"} \
    --min-atoms 30 \
    --k 30 \
    --n-query 40000 \
    --seed 3 \
    --var-keep 0.95 \
    --max-dims 50 \
    --knn-backend "$KNN_BACKEND" \
    --out-json "$OUT_JSON" \
    > neighbour_composition.log 2>&1

echo "Done. See neighbour_composition.log."
