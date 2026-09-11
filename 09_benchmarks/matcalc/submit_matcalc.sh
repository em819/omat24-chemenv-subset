#!/bin/bash -l
#SBATCH --job-name=matcalc-bench
#SBATCH --account=CHANGE_ME            # was p200716                 # <-- your project id
#SBATCH --partition=gpu
#SBATCH --qos=default                    # <-- whichever QOS gives you 48 h
#SBATCH --time=0-04:00:00                # 48 h wall limit
#SBATCH --nodes=2                        # <-- scale this up; 4 GPUs per node
#SBATCH --ntasks-per-node=4              # one task per A100 on the node
#SBATCH --gres=gpu:4                     # whole node: 4x A100-40GB
#SBATCH --cpus-per-task=32               # 128 cores / 4 tasks
#SBATCH --output=matcalc-bench-%j.out
#SBATCH --error=matcalc-bench-%j.err
#SBATCH --requeue                        # checkpoints let it resume if requeued

set -euo pipefail

VENV="${MATCALC_VENV:?set MATCALC_VENV -- see envs/envC-matcalc}"
MODEL="/project/scratch/p200243/evgeny/Projects/OMat24_for_So3lr/active_learning_of_OMat24/4th_generation_tokens_train-mlff/mace-models/v14_T100_alpha1_support_material/be1024_universal_loss_dft_e0s_use_stress_delta_0.01_lr_0.001_num_radial_basis10/compile_model/converged/mace-subset-2nd_gen_chemenv_v14_T100_a1_suppm-omat-medium-be1024_universal_loss_dft_e0s_use_stress_delta_0.01_lr_0.001_rb10_compiled.model"   # <-- your checkpoint
MODEL_NAME="mace-omat-medium-subset-v14_T100_a1_suppm_0.01"
OUTDIR="$SLURM_SUBMIT_DIR/results"

module load env/release/2024.1
module load Python/3.12.3-GCCcore-13.3.0  # same Python the venv was built with
module load CUDA/12.6.0
source "$VENV/bin/activate"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# MatCalc-Bench reference data lives in ~/.cache/matcalc (matcalc hardcodes this
# directory and ignores HF_HOME). matcalc downloads it on first use, but MeluXina
# compute nodes have no internet, so it must be staged from a login node:
#
#   python -c "
#   from matcalc.benchmark import get_benchmark_data
#   for n in ('wbm-random-pbe52-equilibrium-2025.1.json.gz',
#             'mp-binary-pbe-elasticity-2025.1.json.gz',
#             'alexandria-binary-pbe-phonon-2025.1.json.gz',
#             'wbm-high-energy-states.json.gz'):
#       get_benchmark_data(n); print('cached', n)"
#
# (Those are the four defaults of EquilibriumBenchmark / ElasticityBenchmark /
# PhononBenchmark / SofteningBenchmark in matcalc 0.5.1; each is pulled from the
# HF dataset materialyze/matcalc-bench into ~/.cache/matcalc when the benchmark
# object is constructed.)
#
if [ ! -d "$HOME/.cache/matcalc" ]; then
    echo "WARNING: ~/.cache/matcalc not found. matcalc will try to download the" >&2
    echo "         benchmark data now; on an offline compute node that will fail." >&2
    echo "         Stage it from a login node first (see the comment above)." >&2
    if [ "${BENCH_OFFLINE:-0}" = "1" ]; then exit 1; fi
else
    export HF_HUB_OFFLINE=1      # cache present -> never reach for the network
fi

# Each task pins one GPU via SLURM_LOCALID inside the driver. srun exports
# SLURM_PROCID / SLURM_NTASKS, which the driver uses to pick its structure stride.
#
# --kill-on-bad-exit=0 (-K0): if one rank dies hard (OOM, segfault, CUDA error),
# do NOT terminate the whole step -- let the surviving ranks finish their shards.
# The driver already swallows per-structure failures and exits 0, so this only
# matters for catastrophic rank deaths. `|| true` keeps `set -e` from aborting
# before the merge hint below if srun still reports a non-zero overall step exit.
srun --kill-on-bad-exit=0 python "$SLURM_SUBMIT_DIR/run_matcalc_bench.py" \
    --model "$MODEL" \
    --model-name "$MODEL_NAME" \
    --outdir "$OUTDIR" \
    --all-tabulated \
    --dtype float64 || true

echo "All tasks finished. Merge with:"
echo "  python merge_results.py --outdir $OUTDIR --model-name $MODEL_NAME"
