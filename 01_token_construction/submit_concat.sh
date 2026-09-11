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

EXPECTED_RANKS="${EXPECTED_RANKS:-2560}"

conda activate active_learning_omat24_python310

cd "$SLURM_SUBMIT_DIR"

while read -r SUBSET; do
    [ -z "$SUBSET" ] && continue
    case "$SUBSET" in \#*) continue ;; esac
    if [ ! -d "$SUBSET" ]; then
        echo "WARNING: no directory '$SUBSET' -- did stage 01a finish for it?" >&2
        continue
    fi
    srun -n 1 --exact python ./concat_to_parquet.py "$SUBSET" "$EXPECTED_RANKS" \
        > "log_${SUBSET}.txt" 2>&1 &
done < subsets.txt
wait

echo "Done. Parquet files:"
ls -la parquet_file_collection/
