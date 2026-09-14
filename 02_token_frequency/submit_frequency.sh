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

# ============================================================================
# >>> USER CONFIGURATION -- EDIT THIS BLOCK <<<
#
# CHEMENV_PARQUET_GLOB  Glob matching the total_chemenv_<subset>.parquet files
#                  stage 01b wrote. The default assumes this job runs in a
#                  SIBLING directory of the one stage 01b ran in (work/selection/
#                  next to work/parquet_file_collection/). Give an absolute
#                  pattern if your layout differs.
# CHEMENV_FREQ_DB  OUTPUT DuckDB file, created here. Read by BOTH stage 03 and
#                  stage 04 -- keep it after stage 03 finishes.
# DUCK_THREADS / DUCK_MEMORY / DUCK_TEMP_DIR
#                  DuckDB resources. Lower DUCK_MEMORY on a smaller machine;
#                  DuckDB then spills to DUCK_TEMP_DIR instead of failing, so
#                  give that directory several hundred GB on the full corpus.
# CHEMENV_CONDA_ENV  Name of the conda environment holding env A.
#
#     sbatch --account=<your allocation> submit_frequency.sh
# ============================================================================
CHEMENV_CONDA_ENV="${CHEMENV_CONDA_ENV:-active_learning_omat24_python310}"
export CHEMENV_PARQUET_GLOB="${CHEMENV_PARQUET_GLOB:-../parquet_file_collection/total_chemenv*.parquet}"
export CHEMENV_FREQ_DB="${CHEMENV_FREQ_DB:-freq_chemenv.duckdb}"
export DUCK_THREADS="${DUCK_THREADS:-${SLURM_CPUS_PER_TASK:-128}}"
export DUCK_MEMORY="${DUCK_MEMORY:-400GB}"
export DUCK_TEMP_DIR="${DUCK_TEMP_DIR:-./duckdb_temp}"
# ============================================================================

conda activate "$CHEMENV_CONDA_ENV"

srun python -u ./build_frequency_db.py > out.log 2>&1
