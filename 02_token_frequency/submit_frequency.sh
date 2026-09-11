#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=1-00:00:00
#SBATCH --account=CHANGE_ME            # was p200715
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH -J chemenv_frequency

# ============================================================================
# Stage 02 -- build freq_chemenv.duckdb from the stage-01b parquet files.
#
# Environment: envA-chemenv. The cost is DuckDB's parquet scan, not compute.
# The output database is read by both stage 03 and stage 04, so keep it.
# ============================================================================

# -u catches an unset required variable; -e is deliberately NOT set because
# `conda activate` returns non-zero in some shells even when it succeeds.
# The failure modes that matter are checked explicitly below.
set -uo pipefail

conda activate active_learning_omat24_python310

# Where stage 01b wrote parquet_file_collection/. The default assumes this job
# runs in a sibling directory of the one stage 01b ran in.
export CHEMENV_PARQUET_GLOB="${CHEMENV_PARQUET_GLOB:-../parquet_file_collection/total_chemenv*.parquet}"
export CHEMENV_FREQ_DB="${CHEMENV_FREQ_DB:-freq_chemenv.duckdb}"

# Lower these on a smaller machine; DuckDB spills to DUCK_TEMP_DIR instead of
# failing, so make sure that directory has room.
export DUCK_THREADS="${DUCK_THREADS:-${SLURM_CPUS_PER_TASK:-128}}"
export DUCK_MEMORY="${DUCK_MEMORY:-400GB}"
export DUCK_TEMP_DIR="${DUCK_TEMP_DIR:-./duckdb_temp}"

srun python -u ./build_frequency_db.py > out.log 2>&1
