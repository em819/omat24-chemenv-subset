#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=2-00:00:00
#SBATCH --account=CHANGE_ME            # was p200715
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH -J chemenv_integer_tokens

# ============================================================================
# Stage 03 -- integer token ids + per-material token vectors.
#
# Environment: envA-chemenv.  Resources: 128 cores, large memory, 2 days.
# Not launched under srun: it is a single DuckDB process that wants the whole
# node to itself.
# ============================================================================

# -u catches an unset required variable; -e is deliberately NOT set because
# `conda activate` returns non-zero in some shells even when it succeeds.
# The failure modes that matter are checked explicitly below.
set -uo pipefail

# ============================================================================
# >>> USER CONFIGURATION -- EDIT THIS BLOCK <<<
#
# CHEMENV_FREQ_DB  INPUT: the freq_chemenv.duckdb written by stage 02. The
#                  default assumes this job runs in the SAME directory stage 02
#                  ran in; give an absolute path otherwise.
# TOKEN_VEC_DB     OUTPUT: the material-vector database, read by stage 04 as
#                  $V10_TOKEN_VEC_DB. 1000 material_vectors_batch*.parquet files
#                  are also written next to it, in the CURRENT directory.
# DUCK_THREADS / DUCK_MEMORY / DUCK_TEMP_DIR
#                  DuckDB resources. DUCK_MEMORY is the one to lower on a
#                  smaller machine; DUCK_TEMP_DIR can reach several hundred GB.
# CHEMENV_CONDA_ENV  Name of the conda environment holding env A.
#
#     sbatch --account=<your allocation> submit_integer_tokens.sh
# ============================================================================
CHEMENV_CONDA_ENV="${CHEMENV_CONDA_ENV:-active_learning_omat24_python310}"
export CHEMENV_FREQ_DB="${CHEMENV_FREQ_DB:-freq_chemenv.duckdb}"   # stage 02 output
export TOKEN_VEC_DB="${TOKEN_VEC_DB:-material_vectors.duckdb}"     # stage 04 input
export DUCK_THREADS="${DUCK_THREADS:-${SLURM_CPUS_PER_TASK:-128}}"
export DUCK_MEMORY="${DUCK_MEMORY:-400GB}"
export DUCK_TEMP_DIR="${DUCK_TEMP_DIR:-./duckdb_temp}"
# ============================================================================

conda activate "$CHEMENV_CONDA_ENV"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python -u ./build_integer_tokens.py > out.log 2>&1
