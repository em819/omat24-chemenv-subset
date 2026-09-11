#!/bin/bash -l
#SBATCH -N 20
#SBATCH --ntasks-per-node=128          # 20 x 128 = 2560 MPI ranks
#SBATCH --cpus-per-task=1
#SBATCH --time=0-48:00:00
#SBATCH --mem=0
#SBATCH --account=CHANGE_ME            # was p200715
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH -J chemenv-analyze-predictions

# ============================================================================
# Stage 10.2 -- tokenise the structures of stage 10.1 and attach per-site
# force/energy errors to each Chemenv token.
#
# Environment: envA-chemenv (NOT the MACE venv -- this step is pure
#              pymatgen/MPI and uses the same tokeniser as stage 01).
# Resources:   20 CPU nodes x 48 h.
#
# The tokenisation functions here are AST-identical to stage 01's
# tokenize_omat24.py (all 10 shared constants, all 7 shared functions), which
# is what makes the database-side and prediction-side tokens the same join key.
# If you edit one, edit the other.
#
# Input:  ../all_structures_ref_and_predicted.xyz   (from stage 10.1)
# Output: all_structures_with_errors_and_v4chemenvs_*.xyz
# ============================================================================

conda deactivate 2>/dev/null || true
conda activate active_learning_omat24_python310

module --force purge
module load env/release/2024.1
module load OpenMPI/5.0.3-GCC-13.3.0
module load HDF5/1.14.5-gompi-2024a

export MPICC=$(which mpicc)
export MPICXX=$(which mpicxx)

export UCX_LOG_LEVEL=error
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

srun python3 ./analyze_predictions.py >> out.log 2>&1
