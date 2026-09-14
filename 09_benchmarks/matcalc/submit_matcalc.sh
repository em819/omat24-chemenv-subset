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

# ============================================================================
# >>> USER CONFIGURATION -- EDIT THIS BLOCK <<<
#
# MATCALC_VENV  REQUIRED. Path to the env C virtualenv DIRECTORY -- see
#               envs/envC-matcalc/. e.g. export MATCALC_VENV=/opt/envs/envC-matcalc
#
# MODEL         REQUIRED. Path to the MACE model FILE to benchmark. Use the
#               *_compiled.model produced by stage 08 -- MatCalc takes the
#               TorchScript export, unlike the stability/NEB benchmarks which
#               need *_run-1.model.
#
# MODEL_NAME    A short label. It becomes part of every output CSV filename, so
#               use something you will recognise and keep it stable between the
#               run and merge_results.py.
#
# OUTDIR        Where the per-rank CSVs are written. merge_results.py reads the
#               same directory.
#
# MATCALC_CACHE The reference-data cache. matcalc HARDCODES ~/.cache/matcalc and
#               ignores HF_HOME, so this is only the directory the presence check
#               below looks at; change it only if you have redirected $HOME.
#               The four benchmark files must be staged there from a machine with
#               internet -- see the comment further down.
#
#     sbatch --account=<your allocation> --partition=<gpu partition> submit_matcalc.sh
# ============================================================================
VENV="${MATCALC_VENV:?set MATCALC_VENV to the envC-matcalc virtualenv directory -- see envs/envC-matcalc}"
MODEL="${MODEL:?set MODEL to the *_compiled.model exported by stage 08}"
MODEL_NAME="${MODEL_NAME:-mace-omat-medium-subset-v14_T100_a1_suppm_0.01}"
OUTDIR="${OUTDIR:-$SLURM_SUBMIT_DIR/results}"
MATCALC_CACHE="${MATCALC_CACHE:-$HOME/.cache/matcalc}"
# ============================================================================

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
if [ ! -d "$MATCALC_CACHE" ]; then
    echo "WARNING: $MATCALC_CACHE not found. matcalc will try to download the" >&2
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
