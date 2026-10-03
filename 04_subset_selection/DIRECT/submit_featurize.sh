#!/bin/bash -l
#SBATCH --job-name=direct-feat
#SBATCH --account=CHANGE_ME            # was p200715
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --nodes=1                       # one node per array task
#SBATCH --ntasks=128                    # one single-core shard process per core
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --array=0-3                     # 4 nodes x 128 shards = 512 shards (the paper's run)
#SBATCH --output=logs/feat_%A_%a.out
#SBATCH --error=logs/feat_%A_%a.out

# DIRECT step 2: M3GNet features of every chunk. Array task g launches
# SHARDS_PER_NODE single-core shards; global shard G = g * SHARDS_PER_NODE + local
# reads chunk files[G::NSHARDS] and writes shard_<G>.npy + shard_<G>.ids.json.

set -euo pipefail

# ============================================================================
# >>> USER CONFIGURATION -- EDIT THIS BLOCK <<<
#
# DIRECT_VENV   REQUIRED. Path to the env E virtualenv DIRECTORY -- see
#               envs/envE-direct/ (matgl, torch-cpu, dgl, pymatgen, ase).
#
# CHUNK_DIR     INPUT: the chunk directory written by submit_split.sh.
#
# FEAT_DIR      OUTPUT: shard_<G>.npy / shard_<G>.ids.json. Read by submit_select.sh.
#
# SHARDS_PER_NODE, NUM_NODE_GROUPS
#               NSHARDS = SHARDS_PER_NODE * NUM_NODE_GROUPS. NUM_NODE_GROUPS must
#               equal the number of array tasks (--array=0-3 -> 4). The paper
#               used 128 x 4 = 512. Shards with no chunk assigned exit cleanly.
#               NSHARDS sets which chunks each shard reads, hence the row order
#               of the stacked feature matrix, and BIRCH depends on that order:
#               keep NSHARDS=512 to reproduce the published subset.
#
# FEAT_LIMIT    Optional. Structures per shard, for a timing test (e.g. 2000 with
#               --array=0-0). Leave empty for the production run.
#
# The M3GNet model is loaded by name from the matgl cache (~/.cache/matgl) with
# HF_HUB_OFFLINE=1. Fetch it once on a machine with internet:
#     python -c "import matgl; matgl.load_model('M3GNet-Eform-MP-2018.6.1')"
#
#     DIRECT_VENV=... sbatch --account=<your allocation> submit_featurize.sh
# ============================================================================
DIRECT_VENV="${DIRECT_VENV:?set DIRECT_VENV to the envE-direct virtualenv directory -- see envs/envE-direct}"
CHUNK_DIR="${CHUNK_DIR:-chunks}"
FEAT_DIR="${FEAT_DIR:-feats}"
SHARDS_PER_NODE="${SHARDS_PER_NODE:-128}"
NUM_NODE_GROUPS="${NUM_NODE_GROUPS:-4}"
FEAT_LIMIT="${FEAT_LIMIT:-}"
MODEL=M3GNet-Eform-MP-2018.6.1
# ============================================================================

NSHARDS=$(( SHARDS_PER_NODE * NUM_NODE_GROUPS ))
LIMIT_ARGS=""
[ -n "$FEAT_LIMIT" ] && LIMIT_ARGS="--limit ${FEAT_LIMIT}"
mkdir -p "$FEAT_DIR" logs

if [[ ! -f "${DIRECT_VENV}/bin/activate" ]]; then
    echo "ERROR: venv not found at DIRECT_VENV=${DIRECT_VENV}/bin/activate" >&2
    exit 3
fi
# torch/dgl need a newer libstdc++ than the base OS provides: load a recent GCC
# and put its lib64 first (MeluXina module names; use your site's equivalents).
module load env/release/2025.1
module load GCC
export LD_LIBRARY_PATH="${EBROOTGCCCORE}/lib64:${LD_LIBRARY_PATH:-}"
source "${DIRECT_VENV}/bin/activate"

export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export DGL_NUM_THREADS=1

echo "[$(date)] node-group ${SLURM_ARRAY_TASK_ID}/${NUM_NODE_GROUPS}: ${SHARDS_PER_NODE} shards (NSHARDS=${NSHARDS}) -> ${FEAT_DIR}/"

for LOCAL in $(seq 0 $((SHARDS_PER_NODE - 1))); do
    GLOBAL=$(( SLURM_ARRAY_TASK_ID * SHARDS_PER_NODE + LOCAL ))
    srun --exclusive --ntasks=1 --cpus-per-task=1 \
        python featurize_m3gnet.py \
            --input "$CHUNK_DIR" \
            --model "$MODEL" \
            --device cpu \
            --shard "${GLOBAL}/${NSHARDS}" \
            --shard-by file \
            --out "${FEAT_DIR}/shard_${GLOBAL}" \
            --checkpoint-every 5000 \
            $LIMIT_ARGS \
        > "logs/feat_shard_${GLOBAL}.out" 2>&1 &
done
wait

echo "[$(date)] node-group ${SLURM_ARRAY_TASK_ID} done."
