#!/bin/bash -l
#SBATCH --job-name=direct-split
#SBATCH --account=CHANGE_ME            # was p200715
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --array=0-10                    # one task per sub-dataset in subsets.txt
#SBATCH --output=logs/split_%A_%a.out
#SBATCH --error=logs/split_%A_%a.out

# DIRECT step 1: split each OMat24 sub-dataset into chunks of 50,000 frames
# (1,298 chunks in total for the 64,648,643 frames of the training split).
# Frames with rare earths or actinides are dropped here, as in stage 01.

set -euo pipefail

# ============================================================================
# >>> USER CONFIGURATION -- EDIT THIS BLOCK <<<
#
# DIRECT_VENV   REQUIRED. Path to the env E virtualenv DIRECTORY (only the
#               Python standard library is used; any Python 3 works).
#
# OMAT_XYZ_DIR  REQUIRED. Root DIRECTORY of the OMat24 training split as
#               uncompressed extxyz -- the same files stage 05 reads. For each
#               sub-dataset in ../../01_token_construction/subsets.txt, exactly
#               one file named *<subset>.xyz must exist below it, e.g.
#               <dir>/<subset>/all_structures_<subset>.xyz.
#
# CHUNK_DIR     OUTPUT directory of the chunk files, all_structures_<subset>_NNNNN.xyz.
#               It is the input of submit_featurize.sh and
#               submit_generate_train_set_direct.sh.
#
# FRAMES_PER_CHUNK  50000 reproduces the paper.
#
#     DIRECT_VENV=... OMAT_XYZ_DIR=... sbatch --account=<your allocation> submit_split.sh
#
# The chunk names are fixed to all_structures_<subset>_NNNNN.xyz, whatever your
# source files are called: featurisation rows follow the sorted chunk names, and
# BIRCH depends on the row order, so the names are part of the result.
# ============================================================================
DIRECT_VENV="${DIRECT_VENV:?set DIRECT_VENV to the envE-direct virtualenv directory -- see envs/envE-direct}"
OMAT_XYZ_DIR="${OMAT_XYZ_DIR:?set OMAT_XYZ_DIR to the root directory holding the OMat24 training split as extxyz}"
CHUNK_DIR="${CHUNK_DIR:-chunks}"
FRAMES_PER_CHUNK="${FRAMES_PER_CHUNK:-50000}"
# ============================================================================

mkdir -p "$CHUNK_DIR" logs
source "${DIRECT_VENV}/bin/activate"

SUBSETS_TXT="$SLURM_SUBMIT_DIR/../../01_token_construction/subsets.txt"
SUBSET=$(grep -v '^\s*#' "$SUBSETS_TXT" | grep -v '^\s*$' | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
if [ -z "$SUBSET" ]; then
    echo "ERROR: no sub-dataset at index $SLURM_ARRAY_TASK_ID in $SUBSETS_TXT" >&2
    exit 1
fi
mapfile -t MATCHES < <(find -L "$OMAT_XYZ_DIR" -type f -name "*${SUBSET}.xyz")
if [ "${#MATCHES[@]}" -ne 1 ]; then
    echo "ERROR: expected exactly one *${SUBSET}.xyz under $OMAT_XYZ_DIR, found ${#MATCHES[@]}: ${MATCHES[*]}" >&2
    exit 1
fi
F="${MATCHES[0]}"

echo "[$(date)] splitting ${F} ..."
python split_xyz_shards.py \
    --input "$F" \
    --outdir "$CHUNK_DIR" \
    --frames-per-chunk "$FRAMES_PER_CHUNK" \
    --prefix "all_structures_${SUBSET}"
echo "[$(date)] done: ${CHUNK_DIR}/all_structures_${SUBSET}_*.xyz"
