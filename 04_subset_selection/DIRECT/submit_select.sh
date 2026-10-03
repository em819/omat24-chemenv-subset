#!/bin/bash -l
#SBATCH --job-name=direct-select
#SBATCH --account=CHANGE_ME            # was p200715
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128             # BLAS threads for the PCA; BIRCH is single-threaded
#SBATCH --mem=0                         # the whole node (~500 GB)
#SBATCH --time=24:00:00
#SBATCH --output=logs/select_%j.out
#SBATCH --error=logs/select_%j.out

# DIRECT steps 3-5: stack the shard features, then StandardScaler -> PCA ->
# BIRCH(n) -> k per cluster. With --target-size, k is chosen so the subset size
# is closest to TARGET. Submit after submit_featurize.sh has finished:
#     sbatch --dependency=afterok:<featurize_jobid> submit_select.sh

set -euo pipefail

# ============================================================================
# >>> USER CONFIGURATION -- EDIT THIS BLOCK <<<
#
# DIRECT_VENV   REQUIRED. Path to the env E virtualenv DIRECTORY -- see
#               envs/envE-direct/.
#
# FEAT_DIR      INPUT: the shard_<i>.npy written by submit_featurize.sh.
#
# NSHARDS       Must equal SHARDS_PER_NODE * NUM_NODE_GROUPS of the featurize job.
#
# NCLUST, TARGET  BIRCH cluster count and desired subset size. The paper used
#               116500 and 460000 (-> k = 4, 458,754 structures).
#
# OUT           OUTPUT prefix: <OUT>.indices.npy/.json (rows of feats.npy) and
#               <OUT>.ids.json ("<chunk>:<frame>" ids, the input of
#               submit_generate_train_set_direct.sh). feats.npy / feats.ids.json
#               (65 M x 128 float32, ~33 GB) are written to the current directory.
#
#     DIRECT_VENV=... sbatch --account=<your allocation> submit_select.sh
# ============================================================================
DIRECT_VENV="${DIRECT_VENV:?set DIRECT_VENV to the envE-direct virtualenv directory -- see envs/envE-direct}"
FEAT_DIR="${FEAT_DIR:-feats}"
NSHARDS="${NSHARDS:-512}"
NCLUST="${NCLUST:-116500}"
TARGET="${TARGET:-460000}"
OUT="${OUT:-selected_460k}"
# ============================================================================

mkdir -p logs

export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64
export OPENBLAS_NUM_THREADS=64

module load env/release/2025.1
module load GCC
export LD_LIBRARY_PATH="${EBROOTGCCCORE}/lib64:${LD_LIBRARY_PATH:-}"
source "${DIRECT_VENV}/bin/activate"

echo "[$(date)] stacking ${NSHARDS} shards from ${FEAT_DIR}/ ..."
python stack_shards.py "$FEAT_DIR" "$NSHARDS"

echo "[$(date)] selecting subset: n=${NCLUST}, target=${TARGET} ..."
python direct_sampling.py \
    --features feats.npy \
    --ids feats.ids.json \
    --n-clusters "${NCLUST}" \
    --target-size "${TARGET}" \
    --out "${OUT}"

echo "[$(date)] done -> ${OUT}.indices.npy / .json  and  ${OUT}.ids.json"
