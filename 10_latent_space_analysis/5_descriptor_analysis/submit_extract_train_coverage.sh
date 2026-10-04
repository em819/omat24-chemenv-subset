#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
#SBATCH --account=CHANGE_ME            # was p200715
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH -J chemenv_train_coverage

# ============================================================================
# Stage 10.5 -- training-set coverage of the analysed descriptors, read from
# stage 03's material_vectors.duckdb (opened read-only). The query is a full
# scan of env_counts, so run it on the machine that holds the database.
# Everything else in this directory runs on a workstation.
#
# Environment: envA-chemenv (duckdb, pandas, pyarrow).
# ============================================================================
# -u catches an unset required variable; -e is deliberately NOT set because
# `conda activate` returns non-zero in some shells even when it succeeds, so
# the activation is checked explicitly below.
set -uo pipefail

# ============================================================================
# >>> USER CONFIGURATION -- EDIT THIS BLOCK <<<
#
# TOKEN_VEC_DB   INPUT: stage 03's material_vectors.duckdb.
# REPORT_CARD    INPUT: stage 10.4's report_card.parquet.
# COVERAGE_OUT   OUTPUT parquet, read by split_descriptors.py --train-coverage.
#                A <name>_meta.json with table sizes is written next to it.
# MIN_ATOMS      analysis floor; must match split_descriptors.py (manuscript: 30).
# CHEMENV_CONDA_ENV  name of the conda environment holding env A.
#
#     TOKEN_VEC_DB=... REPORT_CARD=... sbatch --account=<your allocation> submit_extract_train_coverage.sh
# ============================================================================
CHEMENV_CONDA_ENV="${CHEMENV_CONDA_ENV:-active_learning_omat24_python310}"
TOKEN_VEC_DB="${TOKEN_VEC_DB:?set TOKEN_VEC_DB to the stage-03 material_vectors.duckdb}"
REPORT_CARD="${REPORT_CARD:-$SLURM_SUBMIT_DIR/../4_cluster_metrics/report_card.parquet}"
COVERAGE_OUT="${COVERAGE_OUT:-train_coverage.parquet}"
MIN_ATOMS="${MIN_ATOMS:-30}"
# ============================================================================

conda activate "$CHEMENV_CONDA_ENV"
if [ "${CONDA_DEFAULT_ENV:-}" != "$CHEMENV_CONDA_ENV" ] || ! python -c "import duckdb, pandas, pyarrow" 2>/dev/null; then
    echo "ERROR: conda env '$CHEMENV_CONDA_ENV' not active or lacks duckdb/pandas/pyarrow" >&2
    exit 1
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
python -u "$SLURM_SUBMIT_DIR/extract_train_coverage.py" \
    --db          "$TOKEN_VEC_DB" \
    --report-card "$REPORT_CARD" \
    --out         "$COVERAGE_OUT" \
    --min-atoms   "$MIN_ATOMS" \
    --threads     "${SLURM_CPUS_PER_TASK:-128}" \
    --memory      200GB \
    > extract_train_coverage.log 2>&1
