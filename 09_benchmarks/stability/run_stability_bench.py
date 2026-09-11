#!/usr/bin/env python3
"""
Run MLIP Arena's MD *stability* benchmark for a custom MACE model, sharded across
all SLURM tasks (one task per GPU) -- WITHOUT Prefect or dask.

What this benchmark is
----------------------
Two MD "torture tests" over the RM24 structures (random mixtures of MP-stable
materials, downloaded from the HF dataset `atomind/mlip-arena`, file
`stability/random-mixture.db`):

  * heating  (NVT): Nose-Hoover, temperature ramped 300 -> 3000 K, 10 ps
  * compress (NPT): Nose-Hoover, temperature 300 -> 3000 K AND pressure 0 -> 500 GPa

The score is `normalized_final_step = final_step / target_steps`: the fraction of
the 10 ps trajectory the model survives before the dynamics blow up (NaN forces /
integrator crash). 1.0 = survived the whole run; <1.0 = diverged early. Energy /
temperature / centre-of-mass-drift statistics and throughput are also recorded.

Why not Prefect/dask
--------------------
As shipped, the benchmark is Jupyter notebooks driving a Prefect flow that spins
up a dask_jobqueue.SLURMCluster. That is a heavier, different execution model than
your matcalc runner, and Prefect's SQLite backend on a shared filesystem is a known
source of HPC hangs. The MD step itself is a plain ASE loop wrapped as a Prefect
`@task` whose body uses no Prefect context, so we call the undecorated function via
`MD.fn(...)` -- no Prefect engine, no dask.

The analysis helpers (get_atoms_from_db / get_runtime_stats / gather_results) live in
`mlip_arena.flows.stability`, which the PyPI wheel does NOT ship (only the GitHub repo
has it). They are small and self-contained, so they are VENDORED verbatim below from
the arena source -- so the metric matches the published `*-heating`/`*-compression`
parquets exactly, and the script depends only on `mlip_arena.tasks.MD` plus
ase/mace/huggingface_hub.

Parallelism model (identical to run_matcalc_bench.py)
-----------------------------------------------------
Each MD is independent, so we build one work-list of (ensemble, structure) items
and give each SLURM task a disjoint stride of it (SLURM_PROCID/SLURM_NTASKS), one
pinned GPU per task (SLURM_LOCALID). Each item writes its own .traj file; a `.done`
marker makes reruns/requeues skip finished items. After the parallel step, run this
script once with --aggregate to glob every rank's trajectories into the parquet.

Environment (set by srun)
-------------------------
SLURM_PROCID  -> global rank   (which stride this task owns)
SLURM_NTASKS  -> world size    (stride length)
SLURM_LOCALID -> local rank    (which of the node's GPUs to pin)
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from ase import Atoms, units

# ----------------------------------------------------------------------------
# Elements the model is allowed to see. Atomic numbers 1-56 (H..Ba) and 72-83
# from Po (84) up -- i.e. the rare earths and actinides the OMat24 model was not
# trained on. RM24 is random mixtures across the WHOLE periodic table, so many of
# its structures contain elements outside this set: the model will (correctly)
# fail on them because the chemistry is out of support, NOT because it is
# dynamically unstable. `--element-policy` decides how to handle that -- see main().
# This set is IDENTICAL to ALLOWED_Z in run_matcalc_bench.py so the two benchmarks
# judge the model on the same chemistry.
# ----------------------------------------------------------------------------
ALLOWED_Z = frozenset(range(1, 57)) | frozenset(range(72, 84))


# ============================================================================
# VENDORED from mlip_arena/flows/stability.py (GitHub main) -- not shipped in the
# PyPI wheel. get_atoms_from_db and gather_results/get_runtime_stats are copied so
# our output parquet matches the published arena format exactly. loguru's `logger`
# and `tqdm` are replaced with plain stderr prints to avoid extra dependencies.
# ============================================================================
def get_atoms_from_db(
    db_path,
    repo_id: str = "atomind/mlip-arena",
    repo_type: str = "dataset",
    subfolder: str = "stability",
    force_download: bool = False,
):
    """Yield ASE Atoms from the RM24 db, downloading it from the HF hub on first use."""
    from ase.db import connect
    from huggingface_hub import hf_hub_download

    db_path = Path(db_path)
    if not db_path.exists():
        db_path = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            subfolder=subfolder,
            filename=db_path.name,
            token=os.getenv("HF_TOKEN", None),
            force_download=force_download,
        )
    with connect(db_path) as db:
        for row in db.select():
            yield row.toatoms()


def get_runtime_stats(traj: list, atoms0: Atoms):
    """Compute runtime statistics for an ASE trajectory. (Verbatim from the arena.)"""
    restarts = []
    steps, times = [], []
    Ts, Ps, Es, KEs = [], [], [], []
    com_drifts = []

    for atoms in traj:
        try:
            energy = atoms.get_potential_energy()
            assert np.isfinite(energy), f"invalid energy: {energy}"
        except Exception:
            continue

        # Frame 0 is written before the MD callback stamps the timing info keys, so
        # its info dict is empty even though it carries an energy (the MACE calc
        # attaches a SinglePointCalculator to the initial frame). The arena's own
        # runs happen to skip it via the energy guard above; ours don't, so guard
        # the info access explicitly. This is a no-op for the metric: frame 0 has no
        # restart/step/datetime, so it contributes nothing to timing or final_step.
        if not {"restart", "datetime", "step"} <= atoms.info.keys():
            continue

        restarts.append(atoms.info["restart"])
        times.append(atoms.info["datetime"])
        steps.append(atoms.info["step"])
        Es.append(energy)
        KEs.append(atoms.get_kinetic_energy())
        Ts.append(atoms.get_temperature())
        try:
            Ps.append(atoms.get_stress()[:3].mean())
        except Exception:
            Ps.append(np.nan)
        com_drifts.append((atoms.get_center_of_mass() - atoms0.get_center_of_mass()).tolist())

    restarts = np.array(restarts)
    times = np.array(times)
    steps = np.array(steps)

    unique_restarts = np.unique(restarts)

    total_time_seconds = 0
    total_steps = 0
    for block in unique_restarts:
        indices = restarts == block
        block_time = times[indices][-1] - times[indices][0]
        total_time_seconds += block_time.total_seconds()
        total_steps += steps[indices][-1] - steps[indices][0]

    # Pull target_steps from the first frame that carries it (frame 0 has no info).
    target_steps = next(
        (a.info["target_steps"] for a in traj if "target_steps" in a.info), 0
    )
    natoms = len(atoms0)

    return {
        "natoms": natoms,
        "total_time_seconds": total_time_seconds,
        "total_steps": total_steps,
        "steps_per_second": total_steps / total_time_seconds if total_time_seconds != 0 else 0,
        "seconds_per_step": total_time_seconds / total_steps if total_steps != 0 else float("inf"),
        "seconds_per_step_per_atom": total_time_seconds / total_steps / natoms
        if total_steps != 0
        else float("inf"),
        "energies": Es,
        "kinetic_energies": KEs,
        "temperatures": Ts,
        "pressures": Ps,
        "target_steps": target_steps,
        "final_step": steps[-1] if len(steps) != 0 else 0,
        "timestep": steps,
        "com_drifts": com_drifts,
    }


def gather_results(run_dir: Path, prefix: str, run_type: str) -> pd.DataFrame:
    """Gather stability results from all matching trajectories into one DataFrame.

    Verbatim from the arena, except: loguru->stderr prints, and an extra `source`
    column (the trajectory stem) so we can compute a correct PER-TRAJECTORY summary
    (the arena df is one row per FRAME, with the per-traj scalars broadcast)."""
    from ase.io import read

    df = pd.DataFrame()
    run_dir = Path(run_dir)
    files = list(run_dir.glob(f"{prefix}_*{run_type}.traj"))

    for fpath in files:
        try:
            traj = read(fpath, index=":")
        except Exception as e:
            print(f"[gather] error reading {fpath}: {e}", file=sys.stderr)
            continue
        try:
            stats = get_runtime_stats(traj, atoms0=traj[0])
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "source": fpath.stem,
                            "formula": traj[0].get_chemical_formula(),
                            "normalized_timestep": stats["timestep"] / stats["target_steps"],
                            "normalized_final_step": stats["final_step"] / stats["target_steps"],
                            "pressure": np.array(stats["pressures"]) / units.GPa,
                        }
                        | stats
                    ),
                ],
                ignore_index=True,
            )
        except Exception as e:
            print(f"[gather] error processing {fpath}: {e}", file=sys.stderr)
            continue
    return df


# ============================================================================
# Driver
# ============================================================================
def get_calculator(model_path: str, device: str, dtype: str, cueq: bool = False, rank: int = 0):
    """Wrap the user's MACE checkpoint in its native ASE calculator.

    float32 is the default here (unlike matcalc's float64): MD stability is not a
    finite-difference property, float32 is what the arena used for MACE, and it is
    ~2x faster -- which matters for 200 x 10 ps trajectories.

    cueq=True routes the equivariant tensor products through cuEquivariance's fused
    CUDA kernels (weights are identical -- purely a load-time kernel choice). If the
    cuequivariance packages fail to load (e.g. wheels absent, or run on a node with
    no CUDA driver), MACE would silently fall back to the e3nn path; we make that
    fallback LOUD so a run that quietly lost its speedup is visible in the logs.
    """
    from mace.calculators import MACECalculator

    kwargs = dict(model_paths=model_path, device=device, default_dtype=dtype)
    if cueq:
        # Probe the import explicitly so we can report active-vs-fallback ourselves,
        # rather than trusting MACE's internal silent fallback.
        try:
            import cuequivariance_ops_torch  # noqa: F401
            kwargs["enable_cueq"] = True
            if rank == 0:
                print("[rank 0] cuEquivariance ENABLED (enable_cueq=True)", flush=True)
        except Exception as e:
            if rank == 0:
                print(f"[rank 0] WARNING: --cueq requested but cuequivariance_ops_torch "
                      f"failed to import ({type(e).__name__}: {e}); falling back to the "
                      f"e3nn path (no speedup).", flush=True)
    return MACECalculator(**kwargs)


def offending_elements(atoms) -> list[str]:
    """Element symbols in `atoms` whose atomic number is outside ALLOWED_Z."""
    from ase.data import chemical_symbols

    return sorted(
        {chemical_symbols[z] for z in set(atoms.get_atomic_numbers()) if z not in ALLOWED_Z}
    )


def supported_structures(structures: list) -> list:
    """RM24 structures whose elements are ALL within ALLOWED_Z, preserving db order.
    Used by --element-policy supported to draw a full-size sample from supported
    chemistry only (filter FIRST, then take the leading n) instead of drop's
    take-then-remove (which shrinks the sample below n)."""
    return [a for a in structures if not offending_elements(a)]


def load_structures(db_name: str, force_download: bool) -> list:
    """All RM24 structures, in db order (pulls the db from the HF hub on first use;
    pre-stage it on a login node -- see the --prefetch step)."""
    return list(get_atoms_from_db(db_name, force_download=force_download))


def traj_path(run_dir: Path, model_name: str, kind: str, idx: int, atoms) -> Path:
    """Trajectory filename. Must match gather_results' glob `{model_name}_*{kind}.traj`.
    `idx` guarantees uniqueness even when material_id is absent and formulas repeat."""
    mid = atoms.info.get("material_id", "random")
    return run_dir / f"{model_name}_{mid}_{atoms.get_chemical_formula()}_{idx}_{kind}.traj"


class _RunTimeout(BaseException):
    """Per-run wall-clock timeout. Subclasses BaseException (not Exception) so it
    propagates cleanly out of MD.fn / the ASE stack even if something in there
    catches `Exception`, exactly like Prefect's TimeoutError-based cancellation."""


@contextmanager
def wall_timeout(seconds: float):
    """SIGALRM wall-clock guard around a synchronous call. The alarm fires at the
    next Python bytecode boundary after `seconds` -- i.e. between MD steps -- so it
    can only overshoot by at most one (possibly slow) integrator step, the same
    granularity Prefect's task timeout gets. `seconds<=0` disables the guard.
    Main-thread only, which is where each srun rank runs its MD loop."""
    if not seconds or seconds <= 0:
        yield
        return

    def _handler(signum, frame):
        raise _RunTimeout(f"per-run timeout after {seconds:g}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)  # cancel before any cleanup runs
        signal.signal(signal.SIGALRM, old)


def run_one(kind: str, atoms, idx: int, calc, run_dir: Path, model_name: str, args) -> str:
    """Run one MD (nvt heating or npt compression) for one structure.

    Returns a one-word status: done / timeout / crash / oom / empty / skip.

    Arena-faithful orchestration WITHOUT Prefect (mirrors mlip_arena.flows.stability,
    which submits every run with `.with_options(timeout_seconds=600, retries=2)` and
    the MD default `restart=True`): each run gets up to `--retries`+1 attempts, each
    capped at `--per-run-timeout` wall-seconds. Because restart=True, every attempt
    RESUMES from the trajectory (MD reads the last frame, subtracts finished steps,
    appends), so the attempts ACCUMULATE ~(retries+1) x timeout of MD progress -- the
    arena default of 3 x 600 s = 30 min. get_runtime_stats already sums across these
    restart blocks, so a resumed trajectory scores correctly.

    A `.done` marker, written once the attempts are spent, makes SLURM requeues skip
    the item (a partial/blow-up .traj IS the stability signal we keep); a partial
    trajectory from a walltime-killed requeue is RESUMED, not discarded. MD.fn is the
    raw ASE loop (no Prefect)."""
    from mlip_arena.tasks import MD

    tpath = traj_path(run_dir, model_name, kind, idx, atoms)
    marker = tpath.with_suffix(".done")
    if marker.exists():
        return "skip"

    # Arena settings, copied verbatim from mlip_arena.flows.stability. restart=True
    # (the MD default the arena relies on) lets each retry pick up where the previous
    # attempt's timeout/crash left off instead of starting the trajectory over.
    common = dict(
        atoms=atoms,
        calculator=calc,            # a built ASE calculator instance -> used as-is
        dynamics="nose-hoover",
        time_step=None,             # -> 0.5 fs if H present else 2.0 fs
        total_time=args.total_time,
        temperature=[300, 3000],
        traj_file=str(tpath),
        traj_interval=args.traj_interval,
        restart=True,
    )
    if kind == "nvt":
        kw = dict(ensemble="nvt", pressure=None,
                  dynamics_kwargs=dict(ttime=args.ttime * units.fs))
    else:  # npt compression
        kw = dict(
            ensemble="npt",
            pressure=[0, 5e2 * units.GPa],
            dynamics_kwargs=dict(
                ttime=args.ttime * units.fs,
                pfactor=((75 * units.fs) ** 2) * 1e2 * units.GPa,
            ),
        )

    attempts = max(1, args.retries + 1)
    last = None  # "timeout" | "oom" | "crash" -- reason the final attempt ended
    completed = False
    for _attempt in range(attempts):
        try:
            with wall_timeout(args.per_run_timeout):
                MD.fn(**common, **kw)
            completed = True
            break
        except _RunTimeout:
            # Ran out of this attempt's wall budget. The next attempt resumes from the
            # trajectory -- this is the mechanism that truncates slow runs at 30 min.
            last = "timeout"
        except Exception:
            # A genuine blow-up (NaN forces / integrator error) or a CUDA OOM. Record
            # the traceback and, like Prefect retries=2, try again (resume). A
            # deterministic failure simply re-fails on the next attempt. OOM is a
            # memory artifact (or, in NPT, a downstream symptom of the cell collapsing
            # to unphysical density), NOT the "PES went unstable" signal -- a separate
            # .oom marker lets aggregation flag it. TorchScript wraps the real error,
            # so match on the text.
            tb = traceback.format_exc()
            (run_dir / f".failed_{tpath.stem}.txt").write_text(tb)
            low = tb.lower()
            last = "oom" if ("out of memory" in low or "outofmemoryerror" in low) else "crash"
        finally:
            # Free the fragmented cache so the next attempt / structure is not starved
            # by leftover allocations. (The itimer is already cancelled by now.)
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    # An attempt was made: mark done so a requeue skips it (and never loops forever on
    # a pathological structure).
    marker.write_text("done\n")

    if completed:
        return "done"
    if last == "oom":
        (run_dir / f".oom_{tpath.stem}.txt").write_text("cuda-oom\n")
        return "oom"
    if not tpath.exists():
        return "empty"
    return last or "crash"


def build_worklist(structures: list, args):
    """(kind, atoms, idx) items for both ensembles, before sharding. NVT uses the
    first --n-nvt structures, NPT the first --n-npt (paper: 120 / 80); -1 = all.
    Element policy is applied HERE (identically on every rank) so the strides stay
    disjoint and reproducible.

    element-policy=supported restricts the candidate pool to in-support structures
    BEFORE take(), so `take(n)` yields the first n SUPPORTED structures -- a full-size
    n-structure sample of supported chemistry -- rather than drop's "first n, minus the
    ones we then remove" (which leaves fewer than n)."""
    pool = supported_structures(structures) if args.element_policy == "supported" else structures

    def take(n):
        return pool if n < 0 else pool[:n]

    kinds = {"nvt": take(args.n_nvt), "npt": take(args.n_npt)}
    if args.only:
        kinds = {args.only: kinds[args.only]}

    work, dropped = [], []
    for kind, subset in kinds.items():
        for idx, atoms in enumerate(subset):
            bad = offending_elements(atoms)
            if bad and args.element_policy == "drop":
                dropped.append((kind, idx, atoms.get_chemical_formula(), ",".join(bad)))
                continue
            work.append((kind, atoms, idx, bad))
    return work, dropped


def write_element_flags(structures: list, args, run_dir: Path) -> None:
    """--element-policy flag: record which structures contain out-of-support
    elements so the analysis can slice 'diverged because out-of-support' from 'truly
    dynamically unstable'. Keyed by formula (all gather_results exposes); written
    once by rank 0."""
    rows = []
    def take(n):
        return structures if n < 0 else structures[:n]
    for kind, subset in {"nvt": take(args.n_nvt), "npt": take(args.n_npt)}.items():
        if args.only and kind != args.only:
            continue
        for idx, atoms in enumerate(subset):
            bad = offending_elements(atoms)
            rows.append({
                "kind": kind, "idx": idx,
                "material_id": atoms.info.get("material_id", "random"),
                "formula": atoms.get_chemical_formula(),
                "has_unsupported": bool(bad),
                "offending": ",".join(bad),
            })
    out = run_dir / "element_flags.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    n_bad = sum(r["has_unsupported"] for r in rows)
    print(f"[rank 0] wrote {out.name}: {n_bad}/{len(rows)} items contain out-of-support "
          f"elements (outside Z in 1-56, 72-83)", flush=True)


def do_prefetch(args) -> None:
    """Login-node step: force the RM24 db download into the HF cache + parse-check it."""
    n = sum(1 for _ in get_atoms_from_db(args.db, force_download=args.force_download))
    print(f"cached RM24 db OK: {n} structures", flush=True)


def do_run(args) -> None:
    import torch

    rank = int(os.environ.get("SLURM_PROCID", 0))
    world = int(os.environ.get("SLURM_NTASKS", 1))
    local = int(os.environ.get("SLURM_LOCALID", 0))

    # Under SLURM cgroup GPU isolation each task may see only its own GPU as cuda:0.
    if torch.cuda.is_available():
        device = f"cuda:{local}" if torch.cuda.device_count() > local else "cuda:0"
    else:
        device = "cpu"
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

    run_dir = Path(args.outdir)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[rank {rank}/{world}] host={socket.gethostname()} device={device} "
          f"dtype={args.dtype} cueq={args.cueq} element_policy={args.element_policy} "
          f"only={args.only or 'nvt+npt'}", flush=True)

    structures = load_structures(args.db, args.force_download)
    if args.limit:  # calibration: cheap timing run
        structures = structures[: args.limit]
    print(f"[rank {rank}] loaded {len(structures)} RM24 structures from {args.db}", flush=True)

    if rank == 0 and args.element_policy == "flag":
        write_element_flags(structures, args, run_dir)

    if rank == 0 and args.element_policy == "supported":
        pool = supported_structures(structures)
        print(f"[rank 0] element-policy=supported: {len(pool)}/{len(structures)} structures are "
              f"fully in-support; taking first {args.n_nvt} (NVT) + first {args.n_npt} (NPT).",
              flush=True)
        for kind, n in (("nvt", args.n_nvt), ("npt", args.n_npt)):
            if n >= 0 and len(pool) < n:
                print(f"[rank 0] WARNING: only {len(pool)} in-support structures but --n-{kind}={n} "
                      f"requested -- {kind} sample will be short.", flush=True)

    work, dropped = build_worklist(structures, args)
    if dropped and rank == 0:
        # Persist the exclusion list so "dropped from the report" is auditable, not silent.
        pd.DataFrame(dropped, columns=["kind", "idx", "formula", "offending"]).to_csv(
            run_dir / "dropped_structures.csv", index=False)
        ex = "; ".join(f"{k}#{i} {f} [{b}]" for k, i, f, b in dropped[:8])
        more = "" if len(dropped) <= 8 else f" (+{len(dropped) - 8} more)"
        print(f"[rank 0] element-policy=drop excluded {len(dropped)} item(s) from the "
              f"simulation -> dropped_structures.csv: {ex}{more}", flush=True)

    shard = work[rank::world]
    print(f"[rank {rank}] {len(shard)}/{len(work)} MD run(s) in this shard", flush=True)

    calc = get_calculator(args.model, device, args.dtype, cueq=args.cueq, rank=rank)

    tally = {"done": 0, "skip": 0, "timeout": 0, "crash": 0, "empty": 0, "oom": 0}
    for kind, atoms, idx, _bad in shard:
        t0 = time.time()
        status = run_one(kind, atoms, idx, calc, run_dir, args.model_name, args)
        tally[status] += 1
        if status != "skip":
            print(f"[rank {rank}] {kind} #{idx} {atoms.get_chemical_formula()}: "
                  f"{status} in {(time.time() - t0) / 60:.1f} min", flush=True)

    # Exit 0 even on per-structure failures: a non-zero exit would tear down sibling
    # ranks under the default SLURM step behaviour.
    print(f"[rank {rank}] DONE: " + ", ".join(f"{k}={v}" for k, v in tally.items()), flush=True)


def do_aggregate(args) -> None:
    """Glob every rank's trajectories into the arena's parquet files and print a
    headline stability number. Run once (no srun) after the parallel step."""
    run_dir = Path(args.outdir)
    kinds = [args.only] if args.only else ["nvt", "npt"]
    label = {"nvt": "heating", "npt": "compression"}
    for kind in kinds:
        df = gather_results(run_dir, prefix=args.model_name, run_type=kind)
        if df.empty:
            print(f"[aggregate] {kind}: no usable trajectories found in {run_dir}", flush=True)
            continue
        # Flag trajectories terminated by CUDA OOM: run_one drops a `.oom_<stem>.txt`
        # marker for each. An OOM here is USUALLY a genuine instability (the model
        # collapses the cell -> density/neighbour-list explosion -> out of memory),
        # so its normalized_final_step is a real failure signal -- but it CAN instead
        # be a pure resource artifact (a physically healthy cell too large for the
        # GPU). We can't tell which without inspecting the trajectory, so we report
        # BOTH statistics -- OOM runs counted as failures, and OOM runs excluded --
        # and leave the `oom_terminated` flag in the parquet so you can reclassify.
        oom_sources = {m.name[len(".oom_"):-len(".txt")]
                       for m in run_dir.glob(".oom_*.txt")}
        df["oom_terminated"] = df["source"].isin(oom_sources)
        out = run_dir / f"{args.model_name}-{label[kind]}.parquet"
        df.to_parquet(out, index=False)
        # df is one row per FRAME; normalized_final_step is constant within a
        # trajectory, so reduce to one value per trajectory via the `source` column.
        per_traj = df.groupby("source")["normalized_final_step"].first()
        nfs = pd.to_numeric(per_traj, errors="coerce")
        is_oom = per_traj.index.isin(oom_sources)
        oom_here = list(per_traj.index[is_oom])

        #     matches the arena metric and does not flatter the model.
        n_all = int(nfs.notna().sum())
        surv_all = int((nfs >= 0.999).sum())
        #     instability, failures).
        clean = nfs[~is_oom]
        n_clean = int(clean.notna().sum())
        surv_clean = int((clean >= 0.999).sum())

        print(f"[aggregate] {kind} ({label[kind]}): {per_traj.size} trajectories "
              f"-> {out.name}  ({len(oom_here)} OOM-terminated)", flush=True)
        print(f"             OOM included  (arena-comparable): "
              f"mean survival {np.nanmean(nfs):.3f}, fully survived {surv_all}/{n_all}", flush=True)
        print(f"             OOM excluded  (resource-artifact read): "
              f"mean survival {np.nanmean(clean) if n_clean else float('nan'):.3f}, "
              f"fully survived {surv_clean}/{n_clean}", flush=True)
        if oom_here:
            print("             OOM-terminated runs (inspect traces -- collapse=real "
                  "failure, healthy=resource artifact):", flush=True)
            for s in oom_here:
                print(f"               {s.replace(args.model_name + '_random_', '')} "
                      f"= {nfs[s]:.3f}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="Path to MACE .model checkpoint (required for the MD run)")
    p.add_argument("--model-name", default="mace-omat-custom",
                   help="Used for traj/parquet filenames; keep stable across the run + aggregate.")
    p.add_argument("--outdir", default="results", help="Trajectory + parquet output dir.")
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    p.add_argument("--cueq", action="store_true",
                   help="Route MACE tensor products through cuEquivariance fused CUDA "
                        "kernels (enable_cueq=True). Requires the cuequivariance* wheels "
                        "installed and a GPU node. ~1.5-3x faster; identical weights. Prints "
                        "a loud warning + falls back to e3nn if the import fails.")
    p.add_argument("--db", default="random-mixture.db",
                   help="RM24 ASE db name in the HF dataset atomind/mlip-arena (subfolder stability).")
    p.add_argument("--force-download", action="store_true",
                   help="Re-download the RM24 db even if cached (default: use HF cache).")
    p.add_argument("--n-nvt", type=int, default=120,
                   help="Number of leading RM24 structures for NVT heating (paper: 120; -1 = all).")
    p.add_argument("--n-npt", type=int, default=80,
                   help="Number of leading RM24 structures for NPT compression (paper: 80; -1 = all).")
    p.add_argument("--only", choices=["nvt", "npt"], default=None,
                   help="Run/aggregate a single ensemble instead of both.")
    p.add_argument("--element-policy", choices=["flag", "drop", "keep", "supported"], default="flag",
                   help="RM24 structures with elements outside your OMat24 support "
                        "(Z in 1-56,72-83). flag (default): run all, write element_flags.csv so "
                        "you can separate out-of-support failures from true instability "
                        "(leaderboard-comparable). drop: take the leading n, THEN skip out-of-support "
                        "ones (fair read of supported chemistry but a SMALLER sample; NOT comparable "
                        "to published parquets). supported: filter to in-support structures FIRST, then "
                        "take the leading n -- a full-size n-structure sample of supported chemistry "
                        "(NOT comparable to published parquets). keep: run all, no flag file.")
    p.add_argument("--total-time", type=float, default=1e4, help="MD length in fs (default 1e4 = 10 ps).")
    p.add_argument("--ttime", type=float, default=25.0, help="Nose-Hoover ttime in fs (arena: 25).")
    p.add_argument("--traj-interval", type=int, default=10, help="Frames written every N steps (arena: 10).")
    p.add_argument("--per-run-timeout", type=float, default=600.0,
                   help="Wall-clock seconds PER ATTEMPT before a run is abandoned "
                        "(arena: 600). Combined with --retries and restart=True, runs "
                        "resume across attempts for a total budget of "
                        "(retries+1)*timeout. <=0 disables (old unbounded behaviour).")
    p.add_argument("--retries", type=int, default=2,
                   help="Extra attempts after the first if a run times out or fails; "
                        "each RESUMES via restart=True (arena: 2 -> 3 attempts, ~30 min "
                        "total at the 600 s default).")
    p.add_argument("--limit", type=int, default=None,
                   help="Calibration: cap the RM24 set to N structures (before splitting) for a quick run.")
    p.add_argument("--prefetch", action="store_true",
                   help="Login-node step: only download + parse-check the RM24 db, then exit.")
    p.add_argument("--aggregate", action="store_true",
                   help="Skip MD; glob existing trajectories into parquet + print stability summary.")
    args = p.parse_args()

    if args.prefetch:
        do_prefetch(args)
        return
    if args.aggregate:
        do_aggregate(args)
        return
    if not args.model:
        p.error("--model is required unless --prefetch or --aggregate is given")
    do_run(args)


if __name__ == "__main__":
    main()
