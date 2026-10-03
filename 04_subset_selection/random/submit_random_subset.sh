#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=01-00:00:00
#SBATCH --account=CHANGE_ME            # was p200715
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH -J random-subset

# One random baseline subset: select the material ids, then copy their frames
# out of the OMat24 extxyz into train.xyz. Submit once per seed.

# ============================================================================
# >>> USER CONFIGURATION -- EDIT THIS BLOCK <<<
#
# RANDOM_SEED   REQUIRED. Seed of the DuckDB reservoir sample. One job per seed;
#               the paper used four seeds (rdm1..rdm4).
#
# CHEMENV_FREQ_DB  INPUT: freq_chemenv.duckdb written by stage 02 (only its
#               material_elements table is read). Default: the current directory.
#
# OMAT_XYZ_DIR  REQUIRED. Root DIRECTORY of the OMat24 training split as
#               uncompressed extxyz, laid out as <dir>/<subset>/*.xyz -- the
#               same files stage 05 reads.
#
# OUT_DIR       OUTPUT directory: random_subset_mat_ids.txt and train.xyz
#               (-> stage 06's train_file).
#
# N_TARGET      Number of materials. 356249 (~6.5 M atoms) reproduces the paper.
#
# CHEMENV_CONDA_ENV  Name of the conda environment holding env A.
#
#     RANDOM_SEED=1 OMAT_XYZ_DIR=... sbatch --account=<your allocation> submit_random_subset.sh
# ============================================================================
RANDOM_SEED="${RANDOM_SEED:?set RANDOM_SEED (one job per seed)}"
CHEMENV_FREQ_DB="${CHEMENV_FREQ_DB:-freq_chemenv.duckdb}"
OMAT_XYZ_DIR="${OMAT_XYZ_DIR:?set OMAT_XYZ_DIR to the root directory holding the OMat24 training split as extxyz}"
OUT_DIR="${OUT_DIR:-random_seed${RANDOM_SEED}}"
N_TARGET="${N_TARGET:-356249}"
CHEMENV_CONDA_ENV="${CHEMENV_CONDA_ENV:-active_learning_omat24_python310}"
# ============================================================================

deactivate 2>/dev/null || true
conda activate "$CHEMENV_CONDA_ENV"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python -u ./select_random_subset.py \
    --freq-db    "$CHEMENV_FREQ_DB" \
    --xyz-dir    "$OMAT_XYZ_DIR" \
    --output-dir "$OUT_DIR" \
    --n-target   "$N_TARGET" \
    --seed       "$RANDOM_SEED" \
    --threads    "${SLURM_CPUS_PER_TASK:-128}" \
    > "out_random_seed${RANDOM_SEED}.log" 2>&1
