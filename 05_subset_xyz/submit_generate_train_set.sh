#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4          # single-threaded + I/O-bound; cores don't speed it up
#SBATCH --time=12:00:00            # one streaming pass over the source xyz; bump to 1-00:00:00 if huge
#SBATCH --account=CHANGE_ME            # was p200715
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH -J gen-train-set
#SBATCH --output=gen-train-set-%j.out
#SBATCH --error=gen-train-set-%j.err

# Materialise ONE training set: read a file of OMat24 mat_ids and copy the
# matching structures out of the source extxyz into a single train.xyz.
# (Validation = the official OMat24 valid split, so no split is done here.)
#
# Unlike the k-fold partition job this is light: no DuckDB, no 128-core greedy,
# no big RAM matrix -- just numpy (to read the id list) + sequential file I/O.
# The cost is one full scan of the source xyz regardless of subset size, so the
# wall time is set by the TOTAL source data, not the subset.

# ============================================================================
# >>> USER CONFIGURATION -- EDIT THIS BLOCK <<<
#
# SUBSET_IDS    REQUIRED. The FILE of selected OMat24 material ids, one per
#               line, written by stage 04 as best_subset_*.txt. You can also use
#               the published id list from the data archive to skip stages
#               01-04 entirely. Example:
#                 /data/work/selection/best_subset_coarsened_v14andAdeep_T100_...topup.txt
#
# OMAT_XYZ_DIR  REQUIRED. Root DIRECTORY holding the OMat24 training split as
#               extxyz. Searched recursively by GLOB_PATTERN, so both layouts
#               work: <dir>/<subset>.xyz (flat) and <dir>/<subset>/*.xyz
#               (nested). This must be the extxyz form -- this stage streams raw
#               text and never parses, so ASE databases cannot be used here and
#               the files must NOT be gzipped. Typically the same directory you
#               gave stage 01 as $OMAT_DATA_DIR.
#
# TRAIN_XYZ     OUTPUT file: the training set, all selected frames concatenated.
#               This is what you point stage 06's preprocess.yaml train_file at.
#
# GLOB_PATTERN  Optional. Narrow it if OMAT_XYZ_DIR also holds xyz files that
#               are not OMat24 source data; the log prints how many files were
#               matched and how many frames scanned, so a wrong glob is visible
#               immediately in the counts.
#
# CHEMENV_CONDA_ENV  Name of the conda environment holding env A.
#
#     sbatch --account=<your allocation> submit_generate_train_set.sh
# ============================================================================
SUBSET_IDS="${SUBSET_IDS:?set SUBSET_IDS to the best_subset_*.txt written by stage 04 (one material id per line)}"
OMAT_XYZ_DIR="${OMAT_XYZ_DIR:?set OMAT_XYZ_DIR to the root directory holding the OMat24 training split as extxyz}"
TRAIN_XYZ="${TRAIN_XYZ:-subset_train.xyz}"
GLOB_PATTERN="${GLOB_PATTERN:-**/*xyz}"
CHEMENV_CONDA_ENV="${CHEMENV_CONDA_ENV:-active_learning_omat24_python310}"
# ============================================================================

deactivate 2>/dev/null || true
conda activate "$CHEMENV_CONDA_ENV"

# Keep BLAS single-threaded (genfromtxt is the only numpy use; harmless anyway).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python -u ./generate_train_set.py \
    --subset-mat-ids "$SUBSET_IDS" \
    --xyz-dir        "$OMAT_XYZ_DIR" \
    --output         "$TRAIN_XYZ" \
    --glob-pattern   "$GLOB_PATTERN" \
    > "out_$(basename "${TRAIN_XYZ%.xyz}").log" 2>&1

echo "Done. Check the tail of out_$(basename "${TRAIN_XYZ%.xyz}").log: 'mat_ids matched X / Y' should be ~100%."
