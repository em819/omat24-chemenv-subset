#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
##SBATCH --ntasks=1
#SBATCH --time=02-00:00:00
#SBATCH --account=CHANGE_ME            # was p200715
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH -J T160_1

#Check limits
ulimit -a  # Print all limits
echo "Max processes: $(ulimit -u)"
echo "Max open files: $(ulimit -n)"

# Monitor memory usage in background
(while true; do 
    date
    free -h
    ps aux | grep python | head -5
    sleep 300  # Every 5 minutes
done) > memory_monitor.log 2>&1 &

# ============================================================================
# >>> USER CONFIGURATION -- EDIT THIS BLOCK <<<
#
# V10_TOKEN_VEC_DB  INPUT: material_vectors.duckdb written by stage 03.
# V10_ENV_FREQ_DB   INPUT: freq_chemenv.duckdb written by stage 02.
#                   Both default to a bare filename, i.e. the CURRENT working
#                   directory -- which is the intended layout (stages 02, 03 and
#                   04 all run in work/selection/). Give absolute paths if not.
# V10_THREADS       DuckDB threads.
# V10_MEMORY        What DuckDB is ALLOWED to take. This is NOT the real memory
#                   requirement: the Python side alone peaks near 165 GiB and
#                   has no knob. Budget a ~400 GB node.
# CHEMENV_CONDA_ENV Name of the conda environment holding env A.
#
# Selection knobs: leave unset to reproduce the published subset. The paper used
#   V10_T=100  V10_DESIGN=a  V10_SUPPORT=materials  V10_NCOV=1  V11_COST_ALPHA=1
# which are also the in-file defaults.
#
#     sbatch --account=<your allocation> submit_select_subset.sh
#
# OUTPUT, into the CURRENT working directory:
#   best_subset_*.txt               -> stage 05's --subset-mat-ids
#   env_ids_found_distribution.txt
#   restart.npz                     -> DELETE before starting a fresh run,
#                                      otherwise this run silently RESUMES.
# ============================================================================
CHEMENV_CONDA_ENV="${CHEMENV_CONDA_ENV:-active_learning_omat24_python310}"
export V10_TOKEN_VEC_DB="${V10_TOKEN_VEC_DB:-material_vectors.duckdb}"  # stage 03
export V10_ENV_FREQ_DB="${V10_ENV_FREQ_DB:-freq_chemenv.duckdb}"        # stage 02
export V10_THREADS="${V10_THREADS:-${SLURM_CPUS_PER_TASK:-128}}"
export V10_MEMORY="${V10_MEMORY:-400GB}"
# ============================================================================

conda activate "$CHEMENV_CONDA_ENV"

export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-128}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
#module purge --force
#module load OpenMPI/5.0.3-GCC-13.3.0
#module load HDF5/1.14.5-gompi-2024a

#export CC=mpicc
#export HDF5_MPI="ON"
#python3 ./concat_to_parquet.py
python -u ./select_subset.py > out_select_subset.log 2>&1
