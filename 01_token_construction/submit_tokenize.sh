#!/bin/bash -l
#SBATCH -N 20
#SBATCH --ntasks-per-node=128          # 20 x 128 = 2560 MPI ranks
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:00
#SBATCH --account=CHANGE_ME            # was p200715
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH -J omat24_tokenize
#SBATCH --array=0-10                   # one task per OMat24 sub-dataset

# ============================================================================
# Stage 01a -- Chemenv tokenisation of the full OMat24 training split.
#
# Environment: envA-chemenv (conda, Python 3.10)
# Resources:   20 CPU nodes x 2 days PER sub-dataset. With --array=0-10 that is
#              ~440 node-days in total. This is by far the most expensive step
#              in the pipeline; read README.md first.
#
# Input is AUTO-DETECTED per sub-dataset:
#   $OMAT_DATA_DIR/<subset>/      a directory -> fairchem AseDBDataset
#   $OMAT_DATA_DIR/<subset>.xyz   a file      -> extxyz  (.extxyz/.xyz.gz too)
# A mix is fine -- each sub-dataset is resolved on its own. If a sub-dataset is
# present in BOTH forms the job refuses to start, because the two can hold
# different data; set OMAT_INPUT_FORMAT=ase-db or =extxyz to decide.
# Set OMAT_FILE_TEMPLATE if your extxyz files are named differently, e.g.
#   export OMAT_FILE_TEMPLATE='all_structures_{subset}.xyz'
#
# The frames must carry sid= in the comment line and per-atom initial_magmoms.
# The script checks frame 0 for both and refuses to start otherwise -- a
# magmom-less input would silently give every atom spin 0 and produce a
# complete, plausible, wrong token set.
# ==========================================================================

# -u catches an unset required variable; -e is deliberately NOT set because
# `conda activate` returns non-zero in some shells even when it succeeds.
# The failure modes that matter are checked explicitly below.
set -uo pipefail

# >>> EDIT <<<
export OMAT_DATA_DIR="${OMAT_DATA_DIR:?set OMAT_DATA_DIR to the directory holding the 11 sub-datasets}"
export OMAT_INPUT_FORMAT="${OMAT_INPUT_FORMAT:-auto}"       # auto | ase-db | extxyz

conda activate active_learning_omat24_python310    # see envs/envA-chemenv/

module --force purge
module load env/release/2024.1
module load OpenMPI/5.0.3-GCC-13.3.0
module load HDF5/1.14.5-gompi-2024a

export MPICC=$(which mpicc)
export MPICXX=$(which mpicxx)

export OMAT_DB_INDEX=${SLURM_ARRAY_TASK_ID:-0}

# ---- one working directory per sub-dataset ---------------------------------
# REQUIRED, not cosmetic. Each rank writes chemenv_<subset>_rank<N>.hdf5 into
# the CURRENT directory, and stage 01b takes a directory and derives the
# sub-dataset label from that directory's NAME. Running several array elements
# in one directory would mix their shards and mislabel the parquet output.
SUBSET=$(grep -v '^\s*#' "$SLURM_SUBMIT_DIR/subsets.txt" | grep -v '^\s*$' \
         | sed -n "$((OMAT_DB_INDEX + 1))p")
if [ -z "$SUBSET" ]; then
    echo "ERROR: no sub-dataset at index $OMAT_DB_INDEX in subsets.txt" >&2
    exit 1
fi
mkdir -p "$SLURM_SUBMIT_DIR/$SUBSET"
cd "$SLURM_SUBMIT_DIR/$SUBSET"

echo "Tokenising sub-dataset '$SUBSET' (index $OMAT_DB_INDEX) in $PWD"
echo "  format=$OMAT_INPUT_FORMAT  data_dir=$OMAT_DATA_DIR"

srun python3 "$SLURM_SUBMIT_DIR/tokenize_omat24.py" >> "out_${SUBSET}.log" 2>&1
