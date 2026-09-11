#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-task=0
#SBATCH --time=12:00:00          # parallel read is minutes; stats is the long pole (~2h @128)
#SBATCH --mem=0                  # whole node
#SBATCH --account=CHANGE_ME            # was p200715
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH -J preprocess-parallel
#SBATCH --output=preprocess-parallel-%j.out
#SBATCH --error=preprocess-parallel-%j.err

# Parallel, fork-safe MACE preprocessing (preprocess_parallel.py).
#
# DELIBERATELY no CUDA module here (unlike the old eval/train scripts):
#   - preprocessing never touches a GPU;
#   - loading CUDA maps its libs into every worker, inflating per-process virtual
#     memory -- that VSZ x num_process is exactly what blew the fork() commit limit.
# Dropping it gives MORE fork headroom. The python script also sets
# CUDA_VISIBLE_DEVICES="" itself, so torch stays CPU-only either way.
# We keep env/release/2024.1 because that is the runtime the mace-work venv was
# built against (matches your working eval/train jobs).

module load env/release/2024.1
conda deactivate 2>/dev/null || true
source ${MACE_VENV:?set MACE_VENV -- see envs/envB-mace}/bin/activate

# One thread per process: each of the 128 workers stays single-threaded, so 128
# processes x 1 thread = full node with no oversubscription.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ---- parallelism (a 128-core MeluXina cpu node) -----------------------------
READ_PROCS=128     # concurrent xyz->HDF5 parse workers (cheap; safe with CUDA off)
STATS_PROCS=128    # concurrent statistics workers (the long pole; 1 thread each)
TRAIN_SHARDS=128   # # train HDF5 files == parse/stats tasks; keep == STATS_PROCS
# Temp xyz shards (~one extra copy of each input set). MUST be writable by your
# user under THIS job's account. Do NOT rely on $SCRATCH: under --account=p200715
# it points into p200715 space you can't write. Default below = a subdir of the
# current working dir (same place processed_data/ goes -> guaranteed writable).
# Point it at your own scratch if you prefer, e.g.
#   SHARD_DIR="/project/scratch/p200243/evgeny/preprocess_shards_${SLURM_JOB_ID}"
SHARD_DIR="./preprocess_shards_${SLURM_JOB_ID}"
# -----------------------------------------------------------------------------

time srun python -u ./preprocess_parallel.py \
    --config       ./preprocess.yaml \
    --read-procs   "$READ_PROCS" \
    --stats-procs  "$STATS_PROCS" \
    --train-shards "$TRAIN_SHARDS" \
    --shard-dir    "$SHARD_DIR"

echo "Done. Outputs under the h5_prefix from preprocess.yaml (train/ val/ test/ + statistics.json)."
echo "If it ENOMEMs at fork (strict overcommit), halve READ_PROCS/STATS_PROCS to 64."

# ---- statistics.json -> statistics_fixed.json --------------------------------
H5_PREFIX=$(python - <<'EOF'
import yaml
print(yaml.safe_load(open("preprocess.yaml")).get("h5_prefix", "") or ".")
EOF
)
echo "Converting statistics.json in ${H5_PREFIX}"
( cd "$H5_PREFIX" && python "$SLURM_SUBMIT_DIR/convert_stats.py" )
