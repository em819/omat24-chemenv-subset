#!/bin/bash -l
#SBATCH --job-name=chemenv-metrics
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128              # Metric 3 (joblib) fans classes over all cores
#SBATCH --gres=gpu:4                     # Metric 1 kNN -> faiss spreads over all GPUs on the node
#SBATCH --mem=0                          # PCA loads the full feature set into RAM (~12.5 GB)
#SBATCH --time=02:00:00
#SBATCH --account=CHANGE_ME            # was p200717                # GPU project id (same as extraction)
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --output=chemenv-metrics-%j.out
#SBATCH --error=chemenv-metrics-%j.err
#
# Stage 2: full-coverage feature-space metrics. Runs in the SAME uv venv as the
# extraction (it already has numpy/scipy/pandas/scikit-learn/joblib); needs
# pyarrow (read parquet) + faiss-gpu-cu12 (exact GPU kNN) installed on top:
#     uv pip install pyarrow faiss-gpu-cu12
# Reads the shards from submit_extract_features.sbatch, writes one
# report_card.parquet row per chemenv_id (purity, compactness_z, components).

set -euo pipefail

MACE_ENV="${MACE_VENV:?set MACE_VENV -- see envs/envB-mace}"

module load env/release/2024.1
#module load CUDA/12.6.0
conda deactivate 2>/dev/null || true
source "$MACE_ENV/bin/activate"

# Metric 3 spawns one joblib process per class-batch; pin BLAS to 1 thread per
# process so the per-class pool doesn't oversubscribe the 128 cores. (Metric 1
# runs on the GPU and PCA is cheap, so single-threaded BLAS costs us nothing.)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

FEAT_DIR="${FEAT_DIR:-../3_extract_features/mace_features}"   # stage 10.3 output
if ! compgen -G "$FEAT_DIR/features_shard*.npy" > /dev/null; then
    echo "ERROR: no feature shards under $FEAT_DIR -- has stage 10.3 finished?" >&2
    exit 1
fi

# --knn-backend faiss-gpu -> exact IndexFlatL2 on the GPU(s); identical results
#   to CPU, just far faster, so we process ALL atoms (--max-atoms 0 = no subsample).
#   For a quick sanity pass first, set --max-atoms 1500000.
python -u "$SLURM_SUBMIT_DIR/chemenv_feature_metrics.py" \
    --feature-glob "$FEAT_DIR/features_shard*.npy" \
    --meta-glob    "$FEAT_DIR/meta_shard*.parquet" \
    --out          "$SLURM_SUBMIT_DIR/report_card.parquet" \
    --k 30 \
    --k-graph 15 \
    --var-keep 0.95 \
    --max-dims 50 \
    --max-atoms 0 \
    --min-size 20 \
    --knn-backend faiss-gpu \
    --n-jobs "$SLURM_CPUS_PER_TASK" \
    > chemenv_metrics.log 2>&1
    # add --cosine to repeat in cosine geometry and check robustness

echo "Done. Inspect report_card.parquet (sorted by n_atoms)."
