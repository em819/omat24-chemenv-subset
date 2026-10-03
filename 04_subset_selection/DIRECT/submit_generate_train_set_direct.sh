#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4          # single-threaded and I/O-bound
#SBATCH --time=12:00:00            # one pass over the selected chunks
#SBATCH --account=CHANGE_ME            # was p200715
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH -J gen-train-set-direct
#SBATCH --output=gen-train-set-direct-%j.out
#SBATCH --error=gen-train-set-direct-%j.err

# DIRECT step 6: copy the selected frames out of the chunk files into train.xyz.

# ============================================================================
# >>> USER CONFIGURATION -- EDIT THIS BLOCK <<<
#
# DIRECT_VENV   REQUIRED. Path to the env E virtualenv DIRECTORY (only the
#               Python standard library is used; any Python 3 works).
#
# SELECTED_IDS  INPUT: <OUT>.ids.json written by submit_select.sh.
#
# CHUNK_DIR     INPUT: the chunk directory written by submit_split.sh -- the
#               same files the featurisation read.
#
# TRAIN_XYZ     OUTPUT: the DIRECT training set (-> stage 06's train_file).
#
#     DIRECT_VENV=... sbatch --account=<your allocation> submit_generate_train_set_direct.sh
# ============================================================================
DIRECT_VENV="${DIRECT_VENV:?set DIRECT_VENV to the envE-direct virtualenv directory -- see envs/envE-direct}"
SELECTED_IDS="${SELECTED_IDS:-selected_460k.ids.json}"
CHUNK_DIR="${CHUNK_DIR:-chunks}"
TRAIN_XYZ="${TRAIN_XYZ:-subset_DIRECT_train.xyz}"
# ============================================================================

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

source "${DIRECT_VENV}/bin/activate"

python -u ./generate_train_set_direct.py \
    --selected-ids "$SELECTED_IDS" \
    --chunks-dir   "$CHUNK_DIR" \
    --output       "$TRAIN_XYZ" \
    > "out_$(basename "${TRAIN_XYZ%.xyz}").log" 2>&1

echo "Done. Check the tail of out_$(basename "${TRAIN_XYZ%.xyz}").log: 'frames written' should equal 'requested' (100%)."
