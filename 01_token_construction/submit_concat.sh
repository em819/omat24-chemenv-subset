#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks=11
#SBATCH --ntasks-per-node=11
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --account=CHANGE_ME            # was p200715
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH -J omat24_concat

# ============================================================================
# Stage 01b -- merge the per-rank HDF5 shards of stage 01a into one parquet
# file per sub-dataset.  Environment: envA-chemenv.
#
# Run this from the SAME directory you submitted stage 01a from: it expects one
# subdirectory per sub-dataset, named exactly as in subsets.txt, each holding
# that sub-dataset's chemenv_<subset>_rank*.hdf5 shards. The sub-dataset label
# in the parquet output is taken from the DIRECTORY NAME.
#
# EXPECTED_RANKS is only a completeness check: the script warns if it finds
# fewer shards than ranks that should have written one. It is the total task
# count of the stage-01a job (20 nodes x 128 = 2560).
#
# Output: parquet_file_collection/total_chemenv_<subset>.parquet
#         -- which is exactly where stage 02 looks for it.
# ==========================================================================

# -u catches an unset required variable; -e is deliberately NOT set because
# `conda activate` returns non-zero in some shells even when it succeeds.
# The failure modes that matter are checked explicitly below.
set -uo pipefail

# ============================================================================
# >>> USER CONFIGURATION -- EDIT THIS BLOCK <<<
#
# EXPECTED_RANKS  Completeness check only: the TOTAL number of MPI ranks the
#                 stage-01a job used (20 nodes x 128 = 2560). The script warns
#                 if it finds fewer chemenv_*_rank*.hdf5 shards than that. The
#                 two 3000-atom AIMD sets were re-run at 10240 ranks in the
#                 published run -- pass EXPECTED_RANKS=10240 for those.
# WORK_DIR        Directory holding one subdirectory per sub-dataset, each with
#                 that sub-dataset's stage-01a shards. Defaults to the directory
#                 you submitted from, which is where submit_tokenize.sh put them.
# CHEMENV_CONDA_ENV   Name of the conda environment holding env A.
#
# Account / partition are SLURM directives above and cannot read variables:
#     sbatch --account=<your allocation> submit_concat.sh
#
# OUTPUT: $WORK_DIR/parquet_file_collection/total_chemenv_<subset>.parquet
#         -- exactly where stage 02 looks for it.
# ============================================================================
EXPECTED_RANKS="${EXPECTED_RANKS:-2560}"
WORK_DIR="${WORK_DIR:-$SLURM_SUBMIT_DIR}"
CHEMENV_CONDA_ENV="${CHEMENV_CONDA_ENV:-active_learning_omat24_python310}"
# ============================================================================

conda activate "$CHEMENV_CONDA_ENV"

cd "$WORK_DIR"

while read -r SUBSET; do
    [ -z "$SUBSET" ] && continue
    case "$SUBSET" in \#*) continue ;; esac
    if [ ! -d "$SUBSET" ]; then
        echo "WARNING: no directory '$SUBSET' -- did stage 01a finish for it?" >&2
        continue
    fi
    srun -n 1 --exact python "$SLURM_SUBMIT_DIR/concat_to_parquet.py" "$SUBSET" "$EXPECTED_RANKS" \
        > "log_${SUBSET}.txt" 2>&1 &
done < "$SLURM_SUBMIT_DIR/subsets.txt"
wait

if [ ! -d parquet_file_collection ]; then
    echo "ERROR: no parquet_file_collection/ was produced. Nothing was merged --" >&2
    echo "       WORK_DIR=$WORK_DIR holds no <subset>/ directory with stage-01a" >&2
    echo "       shards in it. Has stage 01a finished?" >&2
    exit 1
fi

echo "Done. Parquet files:"
ls -la parquet_file_collection/
