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

conda activate active_learning_omat24_python310

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
# ---- inputs and resources (see select_subset.py main() for the full list) ---
export V10_TOKEN_VEC_DB="${V10_TOKEN_VEC_DB:-material_vectors.duckdb}"  # stage 03
export V10_ENV_FREQ_DB="${V10_ENV_FREQ_DB:-freq_chemenv.duckdb}"        # stage 02
export V10_THREADS="${V10_THREADS:-${SLURM_CPUS_PER_TASK:-128}}"
export V10_MEMORY="${V10_MEMORY:-400GB}"

# Selection knobs. The published subset used T=100, design 'a',
# support='materials', ncov=1, alpha=1. Every one of these has an environment
# override; unset, the in-file production defaults apply.
#   export V10_T=100  V10_DESIGN=a  V10_SUPPORT=materials  V10_NCOV=1
#   export V11_COST_ALPHA=1

python -u ./select_subset.py > out_select_subset.log 2>&1
