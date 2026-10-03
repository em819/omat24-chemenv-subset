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
# The frames must carry sid= in the comment line; the script checks frame 0
# and refuses to start otherwise. Per-atom initial_magmoms are used where
# present; atoms without one get spin 0.
# ==========================================================================

# -u catches an unset required variable; -e is deliberately NOT set because
# `conda activate` returns non-zero in some shells even when it succeeds.
# The failure modes that matter are checked explicitly below.
set -uo pipefail

# ============================================================================
# >>> USER CONFIGURATION -- EDIT THIS BLOCK <<<
#
# OMAT_DATA_DIR   REQUIRED. Directory that HOLDS the 11 sub-datasets named in
#                 subsets.txt -- NOT one sub-dataset and NOT a single file.
#                 Inside it, each sub-dataset is either a directory
#                 <subset>/ (ASE database) or a file <subset>.xyz (extxyz).
#                 Example: /data/omat24/train_extracted
#                 Public source: https://huggingface.co/datasets/fairchem/OMAT24
#                 Every frame needs a periodic cell and info['sid'] --
#                 verify with check_input.py first.
# OMAT_INPUT_FORMAT   auto (default) | ase-db | extxyz. 'auto' resolves each
#                 sub-dataset on its own and refuses if both forms exist.
# OMAT_FILE_TEMPLATE  Optional. extxyz filename pattern inside OMAT_DATA_DIR
#                 when the files are not named <subset>.xyz, e.g.
#                 'all_structures_{subset}.xyz'.
# CHEMENV_CONDA_ENV   Name of the conda environment holding env A
#                 (see envs/envA-chemenv/).
#
# Also site-specific, but SLURM directives cannot read shell variables, so edit
# them at the top of this file or override on the command line:
#     sbatch --account=<your allocation> --partition=<cpu partition> submit_tokenize.sh
#
# OUTPUT: one directory per sub-dataset under $SLURM_SUBMIT_DIR, each holding
#         chemenv_<subset>_rank<N>.hdf5. Created automatically below.
# ============================================================================
export OMAT_DATA_DIR="${OMAT_DATA_DIR:?set OMAT_DATA_DIR to the directory HOLDING the 11 sub-datasets named in subsets.txt, e.g. /data/omat24/train_extracted}"
export OMAT_INPUT_FORMAT="${OMAT_INPUT_FORMAT:-auto}"       # auto | ase-db | extxyz
CHEMENV_CONDA_ENV="${CHEMENV_CONDA_ENV:-active_learning_omat24_python310}"
# ============================================================================

conda activate "$CHEMENV_CONDA_ENV"                # see envs/envA-chemenv/

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
