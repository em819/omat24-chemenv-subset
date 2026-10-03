#!/bin/bash -l
#SBATCH -N 5
#SBATCH --ntasks-per-node=128          # 5 x 128 = 640 MPI ranks
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:00
#SBATCH --account=CHANGE_ME
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH -J omat24_convert
#SBATCH --array=0-10                   # one task per OMat24 sub-dataset

# Stage 00: convert one OMat24 sub-dataset (array index -> line of
# ../01_token_construction/subsets.txt) from its ASE databases to
# $OMAT_XYZ_DIR/<subset>/all_structures_<subset>.xyz, without rare earths.

set -uo pipefail

# ============================================================================
# >>> USER CONFIGURATION -- EDIT THIS BLOCK <<<
#
# OMAT_DB_DIR   REQUIRED. Directory HOLDING the unpacked sub-datasets of one
#               OMat24 split, one directory of .aselmdb files per sub-dataset:
#               <dir>/aimd-from-PBE-1000-npt/, <dir>/rattled-relax/, ...
#               Public source: https://huggingface.co/datasets/fairchem/OMAT24
#
# OMAT_XYZ_DIR  REQUIRED. OUTPUT root. Gets <subset>/all_structures_<subset>.xyz, the
#               layout stage 05, 04_subset_selection/{DIRECT,random} read.
#
# CHEMENV_CONDA_ENV  Name of the conda environment holding env A (fairchem-core,
#               ase, mpi4py; see envs/envA-chemenv/).
#
#     OMAT_DB_DIR=... OMAT_XYZ_DIR=... sbatch --account=<your allocation> submit_convert.sh
# ============================================================================
OMAT_DB_DIR="${OMAT_DB_DIR:?set OMAT_DB_DIR to the directory holding the unpacked OMat24 sub-datasets}"
OMAT_XYZ_DIR="${OMAT_XYZ_DIR:?set OMAT_XYZ_DIR to the output root for the extxyz files}"
CHEMENV_CONDA_ENV="${CHEMENV_CONDA_ENV:-active_learning_omat24_python310}"
# ============================================================================

conda activate "$CHEMENV_CONDA_ENV"

module --force purge
module load env/release/2024.1
module load OpenMPI/5.0.3-GCC-13.3.0

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

SUBSETS_TXT="$SLURM_SUBMIT_DIR/../01_token_construction/subsets.txt"
SUBSET=$(grep -v '^\s*#' "$SUBSETS_TXT" | grep -v '^\s*$' | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
if [ -z "$SUBSET" ]; then
    echo "ERROR: no sub-dataset at index $SLURM_ARRAY_TASK_ID in $SUBSETS_TXT" >&2
    exit 1
fi
if [ ! -d "$OMAT_DB_DIR/$SUBSET" ]; then
    echo "ERROR: $OMAT_DB_DIR/$SUBSET is not a directory" >&2
    exit 1
fi

srun python3 ./convert_omat24.py \
    --db-dir "$OMAT_DB_DIR/$SUBSET" \
    --output "$OMAT_XYZ_DIR/$SUBSET/all_structures_${SUBSET}.xyz" \
    > "out_convert_${SUBSET}.log" 2>&1
