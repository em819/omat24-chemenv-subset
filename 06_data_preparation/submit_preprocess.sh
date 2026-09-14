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

# ============================================================================
# >>> USER CONFIGURATION -- EDIT THIS BLOCK <<<
#
# MACE_VENV     REQUIRED. Path to the env B virtualenv DIRECTORY (the one whose
#               bin/activate you source) -- see envs/envB-mace/.
#               e.g. export MACE_VENV=/opt/envs/envB-mace
#
# CONFIG        The preprocessing config. ITS THREE DATASET PATHS ARE
#               PLACEHOLDERS: edit train_file / valid_file / test_file at the top
#               of preprocess.yaml before submitting, or the job stops
#               immediately with a message naming the key. See DATA.md.
#
# SHARD_DIR     Scratch directory for the temporary xyz shards -- about one
#               extra copy of each input set. Must be writable by you under THIS
#               job's account. The default is a subdirectory of the working
#               directory, which is where processed_data/ goes too, so it is
#               guaranteed writable. Point it at fast local scratch if you have
#               some.
#
# READ_PROCS / STATS_PROCS / TRAIN_SHARDS
#               Parallelism, sized for a 128-core node. Keep
#               TRAIN_SHARDS == STATS_PROCS. If preprocessing ENOMEMs at fork
#               under strict overcommit, halve READ_PROCS and STATS_PROCS to 64.
#
#     sbatch --account=<your allocation> submit_preprocess.sh
#
# OUTPUT: <h5_prefix>/{train,val,test}/ HDF5 shards + statistics_fixed.json,
#         which stage 07 reads as ../processed_data/.
# ============================================================================
MACE_VENV="${MACE_VENV:?set MACE_VENV to the envB-mace virtualenv directory -- see envs/envB-mace}"
CONFIG="${CONFIG:-./preprocess.yaml}"
SHARD_DIR="${SHARD_DIR:-./preprocess_shards_${SLURM_JOB_ID}}"
READ_PROCS="${READ_PROCS:-128}"     # concurrent xyz->HDF5 parse workers
STATS_PROCS="${STATS_PROCS:-128}"   # concurrent statistics workers (the long pole)
TRAIN_SHARDS="${TRAIN_SHARDS:-128}" # train HDF5 files; keep == STATS_PROCS
# ============================================================================

module load env/release/2024.1
conda deactivate 2>/dev/null || true
source "$MACE_VENV/bin/activate"

# One thread per process: each of the 128 workers stays single-threaded, so 128
# processes x 1 thread = full node with no oversubscription.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

time srun python -u ./preprocess_parallel.py \
    --config       "$CONFIG" \
    --read-procs   "$READ_PROCS" \
    --stats-procs  "$STATS_PROCS" \
    --train-shards "$TRAIN_SHARDS" \
    --shard-dir    "$SHARD_DIR"

echo "Done. Outputs under the h5_prefix from preprocess.yaml (train/ val/ test/ + statistics.json)."
echo "If it ENOMEMs at fork (strict overcommit), halve READ_PROCS/STATS_PROCS to 64."

# ---- statistics.json -> statistics_fixed.json --------------------------------
H5_PREFIX=$(CONFIG="$CONFIG" python - <<'EOF'
import os, yaml
print(yaml.safe_load(open(os.environ["CONFIG"])).get("h5_prefix", "") or ".")
EOF
)
echo "Converting statistics.json in ${H5_PREFIX}"
( cd "$H5_PREFIX" && python "$SLURM_SUBMIT_DIR/convert_stats.py" )
