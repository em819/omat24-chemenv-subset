#!/usr/bin/env python3
"""
Run the LiTraj *nebDFT2k* Li-ion migration-barrier benchmark for a custom MACE model,
sharded across all SLURM tasks (one task per GPU) -- no Prefect, no dask. Structured to
mirror mlip_arena/run_neb_bench.py exactly.

What this benchmark is
----------------------
nebDFT2k (Dembitskiy et al., npj Comput. Mater. 11, 131 (2025), doi:10.1038/s41524-025-01571-z)
is 1,681 Li-vacancy hops in Materials-Project-derived crystals. For every hop the dataset
ships TWO bands of 5 or 7 images each:

  * `<edge_id>_init.xyz`     -- the BVSE-NEB preconditioned band (the *starting guess*)
  * `<edge_id>_relaxed.xyz`  -- the DFT climbing-image-NEB relaxed band (the *reference*),
                                with DFT energies attached to every image

The uMLIP task, exactly as published, is: take the BVSE-NEB band as the initial guess,
relax the two end-points and then the band with the MLIP, and read off

  em_pred   = max(E) - min(E) over the MLIP-relaxed band   (litraj.utils.get_barrier)
  pa_mae    = path-averaged per-atom displacement between the MLIP-relaxed band and the
              DFT-relaxed band                             (litraj.metrics.PA_MAE)

against the tabulated `em_dft`. Headline metrics are MAE / RMSE / Rp / slope / R2, computed
with litraj.metrics.get_metrics -- the dataset's own scoring functions, imported rather than
reimplemented, so the numbers are directly comparable to Table 4 of the paper.

Why this benchmark, and why it is NOT a drop-in comparison to MLIP Arena
-----------------------------------------------------------------------
MLIP Arena's NEB task is single-element fcc/hcp self-diffusion. This one is multi-element
ionic crystals with a real migrating species, which is the regime OMat24 models are actually
used in. But note the reference level of theory (Methods of the paper):

    VASP, GGA-PBE, VASP-recommended PAWs except Li (1-valence-electron potential),
    ENCUT 520 eV, a single Gamma k-point, Gaussian smearing 0.1 eV,
    *** non-spin-polarised, and NO Hubbard U ***

OMat24 was generated with MPRelaxSet: spin-polarised PBE, PBE_54 PAWs, KSPACING 0.25, and
MP's Hubbard U on oxides/fluorides of Co, Cr, Fe, Mn, Mo, Ni, V, W. So nebDFT2k is a *pure
PBE* reference throughout -- it satisfies the "PBE only" requirement by construction, there
is nothing to filter -- but for the U-carrying chemistries the MODEL is +U while the
REFERENCE is not, and the paper measures that gap directly (MACE-MP-0: MAE 0.07-0.08 eV
without those elements, 0.20-0.21 eV with them; they attribute the difference to the missing
U and spin polarisation, not to the model).

The dataset's `has_specific_TM` index column encodes exactly this. VERIFIED against the
shipped nebDFT2k_index.csv: has_specific_TM is True for a row IFF its chemsys contains one
of {Co, Cr, Fe, Mn, Mo, Ni, V, W} -- i.e. precisely MP's (and OMat24's) Hubbard-U element
set. 646 rows are False, 1035 True. `--tm-policy` therefore controls a genuine
level-of-theory switch, not a chemistry preference:

    flag (default) -- run everything, report the two splits separately (published layout)
    drop           -- score only the 646 U-free rows: the one subset where the model's and
                      the reference's level of theory actually agree
    keep           -- run everything, single pooled metric (not recommended: it averages
                      over a functional mismatch)

Faithfulness to the published protocol
--------------------------------------
The recipe below is transcribed from notebooks/nebDFT2k_benchmark_MACE_MP_small.ipynb in
the LiTraj repository. Every numeric knob is theirs:

    FIRE(images[0]).run()                       <- NO arguments: ASE defaults, fmax=0.05
                                                   and an effectively unbounded step budget
    FIRE(images[-1]).run(fmax=0.1, steps=100)
    NEB(images, parallel=False, k=5.0, climb=True, method='improvedtangent')
    FIRE(neb).run(steps=100)                    <- NO fmax: ASE default, fmax=0.05
    converged = max(|F|_inf over band, source, target) < 0.1
    em = max(E) - min(E) over the relaxed band

WHY FIRE, and not BFGS as in the Sai Gautam scripts
---------------------------------------------------
Because the DFT reference was produced with the same driver. LiTraj ships it as
scripts/ase_run_vasp.py: ASE drives VASP as a calculator (nsw=0, ibrion=-1 -- VASP's internal
relaxer is switched off entirely) and the ionic motion is ASE's, with

    qn_source = FIRE(images[0]);  qn_source.run(steps=100, fmax=0.1)
    qn_target = FIRE(images[-1]); qn_target.run(steps=100, fmax=0.1)
    neb = NEB(images, parallel=False, k=5.0, climb=True, method='improvedtangent')
    FIRE(neb).irun(fmax=0.1, steps=100)
    converged  <=>  abs(neb.get_forces()).max() < 0.1

The uMLIP benchmark notebook is that file with the calculator swapped. So k=5.0, climb=True,
improvedtangent, the 100-step budget and the 0.1 eV/A gate are not arbitrary uMLIP settings --
they are the settings the REFERENCE was computed with. Both sides of the comparison relax
under identical dynamics, which is what makes a difference in E_m attributable to the
potential energy surface rather than to the optimiser. Swapping in BFGS would break that.
(The Sai Gautam benchmark has no such constraint: its DFT references come from many different
studies, so its notebook is free to use QuasiNewton + BFGS, and does.)

Two places where the NOTEBOOK departs from that DFT driver -- neither is a considered protocol
choice, and both are reproduced verbatim anyway because they are what produced the published
Table 4 numbers:

  * The DFT driver relaxes BOTH end-points at (fmax=0.1, steps=100). The notebook calls
    `qn_source.run()` with NO arguments, i.e. ASE's defaults: fmax=0.05 and an effectively
    unbounded step budget. The target keeps (0.1, 100). That asymmetry exists only in the
    notebook. `--symmetric-endpoints --endpoint-fmax 0.1 --endpoint-steps 100` restores the
    DFT driver's own setting and is arguably the better configuration; use it for a
    sensitivity check, not for numbers you compare to the paper.
  * The DFT band is relaxed at fmax=0.1; the notebook's `qn.run(steps=100)` leaves fmax at
    ASE's 0.05 default, so the MLIP band is driven TIGHTER than the reference band was, and
    is correspondingly more likely to exhaust the 100-step budget. `--band-fmax 0.1` matches
    the DFT driver.
  * Consequence of the uncapped source relaxation: a pathological system can spin forever.
    --per-run-timeout is the backstop (as in run_neb_bench.py); a timed-out item is
    reported, never silently dropped.

  * The notebook writes only CONVERGED bands to disk, so non-converged hops vanish from its
    metrics. We keep every result and record `converged`, then have --aggregate report BOTH
    the published gate (converged only, the headline) and the all-items number, plus an
    explicit count of what the gate removed. Strictly more information; the headline is
    unchanged.

  * The notebook instantiates a SEPARATE mace_mp calculator for every image. This is not
    optional: ase.mep.NEB.get_forces() raises ValueError("One or more NEB images share the
    same calculator") if two images hold the same Calculator object. We do the same, but
    build the pool ONCE per rank and reuse it across all 1,681 hops instead of reloading the
    checkpoint 5-7 times per hop. Same objects-per-image topology, same numbers; the only
    cost is a handful of copies of the model weights, which is negligible next to MACE's
    activation memory.

Data staging -- IMPORTANT
-------------------------
`litraj.data.download_dataset` pulls a 65 MB zip over the network. Meluxina compute nodes
are offline, so the dataset MUST be staged on a login node with `--prefetch` first (see
submit_meluxina_litraj_nebdft2k.sh). The run and aggregate steps then read --cache-dir
offline via litraj.data.load_index + ase.io.read, which is exactly what
litraj.data.load_data does internally, only lazily (load_data eagerly reads all 3,362
trajectory files into RAM before you touch the first item).

Parallelism model (identical to run_neb_bench.py)
-------------------------------------------------
Each hop is independent, so we build one work-list and give each SLURM task a disjoint
stride (SLURM_PROCID/SLURM_NTASKS), one pinned GPU per task (SLURM_LOCALID). Each item
writes its own JSON result; a `.done` marker makes reruns/requeues skip finished items.
Run once with --aggregate afterwards.

Environment (set by srun)
-------------------------
SLURM_PROCID  -> global rank   (which stride this task owns)
SLURM_NTASKS  -> world size    (stride length)
SLURM_LOCALID -> local rank    (which of the node's GPUs to pin)
"""

from __future__ import annotations

import argparse
import json
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

# ----------------------------------------------------------------------------
# Elements the model is allowed to see. Atomic numbers 1-56 (H..Ba) and 72-83
# (Hf..Bi). EXCLUDES the lanthanides (57-71) and everything from Po (84) up -- the
# f-block the OMat24 model was not trained on. IDENTICAL to ALLOWED_Z in
# mlip_arena/run_neb_bench.py / run_stability_bench.py / run_matcalc_bench.py so every
# benchmark judges the model on the same chemistry.
#
# nebDFT2k contains 45 elements, of which exactly two are out of support: Eu and La.
# They appear in 25 of the 1,681 hops.
# ----------------------------------------------------------------------------
ALLOWED_Z = frozenset(range(1, 57)) | frozenset(range(72, 84))

# MP's (and therefore OMat24's) Hubbard-U element set. nebDFT2k's own `has_specific_TM`
# index column is True IFF the chemsys intersects this set -- verified row-by-row against
# the shipped nebDFT2k_index.csv (646 False / 1035 True, zero disagreements). Kept here so
# the flag can be recomputed if a future dataset revision drops the column.
U_ELEMENTS = frozenset({"Co", "Cr", "Fe", "Mn", "Mo", "Ni", "V", "W"})

DATASET = "nebDFT2k"

# Published protocol constants (notebooks/nebDFT2k_benchmark_MACE_MP_small.ipynb).
PUB_K = 5.0                # NEB spring constant, eV/A^2
PUB_METHOD = "improvedtangent"
PUB_CLIMB = True
PUB_N_STEPS = 100          # step cap for the target end-point and for the band
PUB_FMAX = 0.1             # convergence GATE (and the target end-point's fmax)
PUB_BAND_FMAX = 0.05       # ASE's FIRE default, which the notebook relies on for the band


# ============================================================================
# Driver
# ============================================================================
def get_calculator(model_path: str, device: str, dtype: str, cueq: bool = False, rank: int = 0):
    """Wrap the user's MACE checkpoint in its native ASE calculator.

    float64 by default, for the same reason as run_neb_bench.py: a migration barrier is a
    small ENERGY DIFFERENCE along a relaxed path, and float32 rounding noise perturbs both
    the relaxation and the barrier read-off. The LiTraj notebook also uses float64.

    cueq=True routes the equivariant tensor products through cuEquivariance's fused CUDA
    kernels (identical weights). We probe the import explicitly and warn LOUDLY on fallback
    so a run that quietly lost its speedup is visible in the logs.
    """
    from mace.calculators import MACECalculator

    kwargs = dict(model_paths=model_path, device=device, default_dtype=dtype)
    if cueq:
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


class CalculatorPool:
    """A growing pool of INDEPENDENT calculator instances, one per NEB image.

    ase.mep.NEB refuses to run if two images share a Calculator object (it compares the
    calculators as a set), which is why both published notebooks build one per image. Doing
    that literally would reload the checkpoint 5-7 times for every one of 1,681 hops; this
    pool builds them once per rank and hands the same objects to every hop. The band's
    topology -- one distinct calculator per image -- is identical either way.
    """

    def __init__(self, factory):
        self._factory = factory
        self._pool = []

    def take(self, n: int):
        while len(self._pool) < n:
            self._pool.append(self._factory())
        return self._pool[:n]


def _import_neb():
    """ase.mep.NEB is the current home; ase.neb (what the notebook imports) is a deprecated
    shim for it. Prefer the current path, fall back to the notebook's, so this script works
    across the ASE versions in the arena venv."""
    try:
        from ase.mep import NEB
    except ImportError:
        from ase.neb import NEB  # noqa: F401  (older ASE)
    return NEB


def offending_element(symbol: str) -> bool:
    """True if `symbol`'s atomic number is outside ALLOWED_Z (out of model support)."""
    from ase.data import atomic_numbers

    return atomic_numbers[symbol] not in ALLOWED_Z


def chemsys_unsupported(chemsys: str) -> bool:
    """True if any element in a `Li-Fe-P-O`-style chemsys string is out of support."""
    return any(offending_element(e) for e in str(chemsys).split("-"))


def chemsys_has_u_element(chemsys: str) -> bool:
    """True if the chemsys contains one of MP/OMat24's Hubbard-U elements."""
    return bool(U_ELEMENTS & set(str(chemsys).split("-")))


def run_neb_published(images, pool: "CalculatorPool", args, submitdir: Path | None = None):
    """The LiTraj uMLIP-NEB recipe, transcribed from
    notebooks/nebDFT2k_benchmark_MACE_MP_small.ipynb::optimize().

    `images` is the BVSE-NEB preconditioned band from `<edge_id>_init.xyz`, already the
    right length -- the notebook does NOT re-interpolate, and neither do we.

    Returns (neb, diagnostics dict).
    """
    from ase.optimize import FIRE

    NEB = _import_neb()

    # One distinct calculator per image, as in the notebook and as ase.mep.NEB requires.
    for image, c in zip(images, pool.take(len(images))):
        image.calc = c

    logs = {}
    if submitdir is not None:
        submitdir.mkdir(parents=True, exist_ok=True)
        logs = {
            "source": dict(logfile=str(submitdir / "qn_source.log"),
                           trajectory=str(submitdir / "qn_source.traj")),
            "target": dict(logfile=str(submitdir / "qn_target.log"),
                           trajectory=str(submitdir / "qn_target.traj")),
            "band": dict(logfile=str(submitdir / "qn.log"),
                         trajectory=str(submitdir / "neb.traj")),
        }
    else:
        # ASE writes optimiser logs to stdout when logfile is None; '-' would spam the
        # SLURM log with 1,681 x 3 optimiser tables, so silence them by default.
        logs = {k: dict(logfile=os.devnull, trajectory=None)
                for k in ("source", "target", "band")}

    # --- end-points -----------------------------------------------------------------
    # NOTE the asymmetry: the notebook calls run() with no arguments on the source (ASE
    # defaults: fmax=0.05, unbounded steps) and run(fmax=0.1, steps=100) on the target.
    # Reproduced verbatim unless --symmetric-endpoints is given. See module docstring.
    qn_source = FIRE(images[0], **logs["source"])
    if args.symmetric_endpoints:
        qn_source.run(fmax=args.endpoint_fmax, steps=args.endpoint_steps)
    else:
        qn_source.run()
    n_src = qn_source.get_number_of_steps()

    qn_target = FIRE(images[-1], **logs["target"])
    if args.symmetric_endpoints:
        qn_target.run(fmax=args.endpoint_fmax, steps=args.endpoint_steps)
    else:
        qn_target.run(fmax=args.fmax, steps=args.n_steps)
    n_tgt = qn_target.get_number_of_steps()

    # --- band -----------------------------------------------------------------------
    neb = NEB(images, parallel=False, k=args.k, climb=PUB_CLIMB, method=PUB_METHOD)
    qn = FIRE(neb, **logs["band"])
    if getattr(args, "band_fmax", None) is None:
        qn.run(steps=args.n_steps)  # no fmax -> ASE default 0.05, as in the notebook
    else:
        qn.run(fmax=args.band_fmax, steps=args.n_steps)
    n_band = qn.get_number_of_steps()

    # --- the published convergence gate ----------------------------------------------
    # abs(...).max() is the max force COMPONENT (not the vector norm) -- the notebook's
    # choice, kept so the gate admits/rejects exactly the same items.
    max_neb_force = float(abs(neb.get_forces()).max())
    max_source_force = float(abs(neb.images[0].get_forces()).max())
    max_target_force = float(abs(neb.images[-1].get_forces()).max())
    worst = max(max_neb_force, max_source_force, max_target_force)

    diag = {
        "steps_source": int(n_src),
        "steps_target": int(n_tgt),
        "steps_band": int(n_band),
        "fmax_band": max_neb_force,
        "fmax_source": max_source_force,
        "fmax_target": max_target_force,
        "fmax_worst": worst,
        "converged": bool(worst < args.fmax),
        "flag_step_limit": bool(n_band >= args.n_steps),
    }
    return neb, diag


# ---- caching (login-node prefetch <-> offline compute-node run) -------------
def index_path(cache_dir: Path) -> Path:
    return cache_dir / DATASET / f"{DATASET}_index.csv"


def load_index(cache_dir: Path) -> pd.DataFrame:
    """The nebDFT2k index, read through litraj's own loader so the schema stays whatever
    the package says it is. Columns: material_id, edge_id, chemsys, has_specific_TM,
    em_bvse, em_dft, _split."""
    from litraj.data import load_index as _load_index

    if not index_path(cache_dir).exists():
        raise FileNotFoundError(
            f"{index_path(cache_dir)} not found. Run `--prefetch` on a login node with "
            f"internet first (see submit_meluxina_litraj_nebdft2k.sh)."
        )
    return _load_index(DATASET, str(cache_dir))


def read_band(cache_dir: Path, edge_id: str, which: str):
    """One band from the cache. `which` is 'init' (BVSE-NEB guess) or 'relaxed' (DFT-NEB
    reference). Same file layout litraj.data.load_data uses, read lazily per item."""
    from ase.io import read

    return read(str(cache_dir / DATASET / f"{edge_id}_{which}.xyz"), index=":")


def do_prefetch(args) -> None:
    """Login-node step: pull the 65 MB nebDFT2k zip via litraj's own downloader and unzip
    it into --cache-dir. Compute nodes then read these files offline."""
    from litraj.data import download_dataset

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    download_dataset(DATASET, str(cache_dir), unzip=True)

    idx = load_index(cache_dir)
    n_missing = sum(
        0 if (cache_dir / DATASET / f"{e}_init.xyz").exists()
        and (cache_dir / DATASET / f"{e}_relaxed.xyz").exists() else 1
        for e in idx.edge_id
    )
    print(f"[prefetch] {DATASET}: {len(idx)} hops indexed, {n_missing} with a missing "
          f"trajectory file", flush=True)
    print(f"[prefetch] splits: {idx['_split'].value_counts().to_dict()}", flush=True)
    print(f"[prefetch] has_specific_TM: "
          f"{idx['has_specific_TM'].value_counts().to_dict()}", flush=True)
    if n_missing:
        print(f"[prefetch] WARNING: {n_missing} indexed hops have no trajectory on disk; "
              f"they will be skipped by the run step.", file=sys.stderr)


def build_worklist(cache_dir: Path, args):
    """edge_id items after split / element / TM filtering, before sharding. All filtering
    happens HERE (identically on every rank) so the strides stay disjoint and reproducible.
    Returns (work, dropped) where work rows are dicts."""
    idx = load_index(cache_dir)

    # Recompute the TM flag from chemsys if a future dataset revision drops the column.
    if "has_specific_TM" not in idx.columns:
        idx["has_specific_TM"] = idx["chemsys"].map(chemsys_has_u_element)

    if args.split != "all":
        idx = idx[idx["_split"] == args.split]

    idx = idx.copy()
    idx["has_unsupported"] = idx["chemsys"].map(chemsys_unsupported)

    dropped = []
    if args.element_policy in ("drop", "supported"):
        for _, r in idx[idx["has_unsupported"]].iterrows():
            dropped.append((r.edge_id, r.chemsys, "unsupported_element"))
        idx = idx[~idx["has_unsupported"]]

    if args.tm_policy == "drop":
        for _, r in idx[idx["has_specific_TM"]].iterrows():
            dropped.append((r.edge_id, r.chemsys, "hubbard_u_element"))
        idx = idx[~idx["has_specific_TM"]]

    if args.limit is not None:
        idx = idx.iloc[: args.limit]

    work = idx.to_dict("records")
    return work, dropped


def result_path(run_dir: Path, model_name: str, edge_id: str) -> Path:
    return run_dir / f"{model_name}_{edge_id}.json"


class _RunTimeout(BaseException):
    """Per-run wall-clock timeout. Subclasses BaseException so it propagates cleanly out of
    the ASE/NEB stack even past bare `except Exception`."""


@contextmanager
def wall_timeout(seconds: float):
    """SIGALRM wall-clock guard around a synchronous call (main-thread only)."""
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
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def run_one(row, pool, cache_dir: Path, run_dir: Path, model_name: str, args) -> str:
    """One hop. Returns done / timeout / crash / oom / skip / missing.

    A NEB is NOT trajectory-resumable, so each of the --retries+1 attempts starts FRESH.
    A `.done` marker (written once attempts are spent) makes SLURM requeues skip the item.
    """
    from ase.io import write

    from litraj.metrics import PA_MAE
    from litraj.utils import get_barrier

    edge_id = row["edge_id"]
    out = result_path(run_dir, model_name, edge_id)
    marker = out.with_suffix(".done")
    if marker.exists():
        return "skip"

    try:
        traj_init = read_band(cache_dir, edge_id, "init")
        traj_ref = read_band(cache_dir, edge_id, "relaxed")
    except Exception:
        (run_dir / f".failed_{out.stem}.txt").write_text(traceback.format_exc())
        marker.write_text("done\n")
        return "missing"

    submitdir = (run_dir / "opt" / edge_id) if args.save_traj else None

    attempts = max(1, args.retries + 1)
    last = None
    result = None
    for _attempt in range(attempts):
        result = None   # a timeout can fire part-way through building `result`; never keep
                        # a half-filled dict from a failed attempt
        try:
            with wall_timeout(args.per_run_timeout):
                images = [im.copy() for im in traj_init]
                neb, diag = run_neb_published(images, pool, args, submitdir)

                em_pred = float(get_barrier(neb.images))
                profile = [float(im.get_potential_energy()) for im in neb.images]

                # Path-averaged geometry error against the DFT-NEB band, using LiTraj's own
                # implementation. Only defined when the two bands have equal image counts,
                # which the dataset guarantees (init and relaxed are the same length).
                pa_mae = None
                if len(traj_ref) == len(neb.images):
                    try:
                        pa_mae = float(PA_MAE(traj_ref, neb.images))
                    except Exception:
                        pa_mae = None

                result = dict(diag)
                result.update({
                    "em_pred": em_pred,
                    "profile_pred": profile,
                    "pa_mae": pa_mae,
                    "n_images": len(neb.images),
                    "natoms": len(neb.images[0]),
                })
            break
        except _RunTimeout:
            last = "timeout"
        except Exception:
            tb = traceback.format_exc()
            (run_dir / f".failed_{out.stem}.txt").write_text(tb)
            low = tb.lower()
            last = "oom" if ("out of memory" in low or "outofmemoryerror" in low) else "crash"
        finally:
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    marker.write_text("done\n")

    if result is not None:
        # em_dft is tabulated in the index; recompute it from the reference band as a
        # self-check. They agree to ~1e-6 eV on the shipped data -- a mismatch means the
        # cache is inconsistent with the index, which is worth knowing before trusting a MAE.
        try:
            em_ref_recomputed = float(get_barrier(traj_ref))
        except Exception:
            em_ref_recomputed = float("nan")

        result.update({
            "edge_id": edge_id,
            "material_id": row.get("material_id"),
            "chemsys": row.get("chemsys"),
            "split": row.get("_split"),
            "em_dft": float(row["em_dft"]),
            "em_bvse": float(row["em_bvse"]) if pd.notna(row.get("em_bvse")) else None,
            "em_dft_recomputed": em_ref_recomputed,
            "has_specific_TM": bool(row["has_specific_TM"]),
            "has_unsupported": bool(chemsys_unsupported(row.get("chemsys", ""))),
        })
        out.write_text(json.dumps(result))

        if args.save_band:
            band_dir = run_dir / "bands"
            band_dir.mkdir(parents=True, exist_ok=True)
            # Same naming as the notebook. Unlike the notebook we write the band whether or
            # not it converged (the JSON carries the flag); the metrics still gate on it.
            write(str(band_dir / f"{edge_id}_relaxed_{model_name}.xyz"),
                  neb.images, format="extxyz")
        return "done"

    if last == "oom":
        (run_dir / f".oom_{out.stem}.txt").write_text("cuda-oom\n")
    return last or "crash"


def do_run(args) -> None:
    import torch

    rank = int(os.environ.get("SLURM_PROCID", 0))
    world = int(os.environ.get("SLURM_NTASKS", 1))
    local = int(os.environ.get("SLURM_LOCALID", 0))

    if torch.cuda.is_available():
        device = f"cuda:{local}" if torch.cuda.device_count() > local else "cuda:0"
    else:
        device = "cpu"
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

    run_dir = Path(args.outdir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    print(f"[rank {rank}/{world}] host={socket.gethostname()} device={device} "
          f"dtype={args.dtype} cueq={args.cueq} split={args.split} "
          f"element_policy={args.element_policy} tm_policy={args.tm_policy} "
          f"k={args.k} n_steps={args.n_steps} fmax={args.fmax} "
          f"band_fmax={args.band_fmax or 'ASE default 0.05'} "
          f"symmetric_endpoints={args.symmetric_endpoints}", flush=True)

    work, dropped = build_worklist(cache_dir, args)
    if rank == 0:
        if dropped:
            pd.DataFrame(dropped, columns=["edge_id", "chemsys", "reason"]).to_csv(
                run_dir / "dropped_structures.csv", index=False)
            n_el = sum(1 for _e, _c, r in dropped if r == "unsupported_element")
            n_u = sum(1 for _e, _c, r in dropped if r == "hubbard_u_element")
            print(f"[rank 0] excluded {len(dropped)} hop(s) -> dropped_structures.csv "
                  f"({n_el} out-of-support element, {n_u} Hubbard-U chemistry)", flush=True)
        n_tm = sum(1 for r in work if r["has_specific_TM"])
        print(f"[rank 0] work-list: {len(work)} hop(s); {n_tm} contain a Hubbard-U element "
              f"(reference is plain PBE for all of them)", flush=True)

    shard = work[rank::world]
    print(f"[rank {rank}] {len(shard)}/{len(work)} NEB run(s) in this shard", flush=True)

    # Built lazily and reused for every hop; see CalculatorPool.
    pool = CalculatorPool(lambda: get_calculator(args.model, device, args.dtype,
                                                 cueq=args.cueq, rank=rank))

    tally = {"done": 0, "skip": 0, "timeout": 0, "crash": 0, "oom": 0, "missing": 0}
    for row in shard:
        t0 = time.time()
        status = run_one(row, pool, cache_dir, run_dir, args.model_name, args)
        tally[status] = tally.get(status, 0) + 1
        if status != "skip":
            print(f"[rank {rank}] {row['edge_id']}: {status} in "
                  f"{(time.time() - t0) / 60:.2f} min", flush=True)

    print(f"[rank {rank}] DONE: " + ", ".join(f"{k}={v}" for k, v in tally.items()), flush=True)


def do_aggregate(args) -> None:
    """Glob every rank's JSON results into one parquet and print the published metrics.
    Run once (no srun)."""
    from litraj.metrics import get_metrics

    run_dir = Path(args.outdir)

    rows = []
    for f in run_dir.glob(f"{args.model_name}_*.json"):
        try:
            rows.append(json.loads(f.read_text()))
        except Exception as e:
            print(f"[aggregate] error reading {f}: {e}", file=sys.stderr)
    if not rows:
        print(f"[aggregate] no result JSONs found in {run_dir}", flush=True)
        return
    df = pd.DataFrame(rows)

    oom_stems = {m.name[len(".oom_"):-len(".txt")] for m in run_dir.glob(".oom_*.txt")}
    df["oom_terminated"] = df["edge_id"].map(lambda e: f"{args.model_name}_{e}" in oom_stems)

    # Items that were ATTEMPTED but produced no JSON (timeout / crash / OOM / missing file)
    # still carry a `.done` marker, so a requeue skips them and they vanish from the metrics.
    # Report them: a MAE over the subset that happened to finish is biased toward easy hops.
    done_stems = {m.stem for m in run_dir.glob(f"{args.model_name}_*.done")}
    json_stems = {f.stem for f in run_dir.glob(f"{args.model_name}_*.json")}
    missing = sorted(done_stems - json_stems)
    if missing:
        print(f"[aggregate] WARNING: {len(missing)} attempted hop(s) produced no result "
              f"(timeout/crash/OOM) and are absent from every metric below. See "
              f".failed_*.txt in {run_dir}.", file=sys.stderr)

    # Reference self-check: index em_dft vs the barrier recomputed from the DFT band.
    if "em_dft_recomputed" in df.columns:
        d = (df["em_dft"] - df["em_dft_recomputed"]).abs()
        if d.notna().any() and d.max() > 1e-4:
            print(f"[aggregate] WARNING: tabulated em_dft and the barrier recomputed from "
                  f"<edge_id>_relaxed.xyz disagree by up to {d.max():.4f} eV -- the cache "
                  f"may not match the index.", file=sys.stderr)

    out = run_dir / f"{args.model_name}-litraj-nebdft2k.parquet"
    df.to_parquet(out, index=False)

    print(f"\n{'=' * 78}\n[aggregate] LiTraj nebDFT2k: {len(df)} hop(s) -> {out.name}"
          f"\n{'=' * 78}", flush=True)

    n_conv = int(df["converged"].sum())
    n_step = int(df["flag_step_limit"].sum())
    n_oom = int(df["oom_terminated"].sum())
    print(f"  converged under the published gate (max|F|_inf < {args.fmax} eV/A): "
          f"{n_conv}/{len(df)}", flush=True)
    print(f"  hit the {args.n_steps}-step band budget: {n_step}   |   OOM-terminated: "
          f"{n_oom}   (all kept in the parquet)", flush=True)
    print(f"  NOTE: the published notebook DISCARDS non-converged bands, so the "
          f"'converged only' block below is the paper-comparable one.", flush=True)

    def emit(title, d):
        d = d.dropna(subset=["em_dft", "em_pred"])
        if len(d) < 3:
            print(f"  {title.rstrip(':')}: n={len(d)} (too few for regression metrics)",
                  flush=True)
            return
        m = get_metrics(d["em_dft"].to_numpy(), d["em_pred"].to_numpy())
        signed = (d["em_pred"] - d["em_dft"])
        pa = d["pa_mae"].dropna()
        print(f"  {title}", flush=True)
        print(f"    n={len(d):<5} MAE {m['mae']:.3f}  RMSE {m['rmse']:.3f}  "
              f"Rp {m['Rp']:.3f}  slope {m['slope']:.3f}  R2 {m['R2']:.3f}", flush=True)
        print(f"          signed-mean {signed.mean():+.3f}  median |err| "
              f"{signed.abs().median():.3f}  over-predicted {int((signed > 0).sum())}/{len(d)}"
              + (f"  PA-MAE {pa.mean():.3f} A (n={len(pa)})" if len(pa) else ""), flush=True)

    conv = df[df["converged"].fillna(False).astype(bool)]
    for label, d in (("ALL items (incl. non-converged)", df),
                     ("CONVERGED only  [paper-comparable]", conv)):
        print(f"\n  --- {label} ---", flush=True)
        emit("migration barrier E_m vs DFT-NEB (PBE, no +U) [eV]:", d)
        # The split that matters for an OMat24 model: the reference is plain PBE for every
        # hop, but the model carries MP's Hubbard U on Co/Cr/Fe/Mn/Mo/Ni/V/W chemistries.
        if "has_specific_TM" in d.columns and d["has_specific_TM"].nunique() > 1:
            emit("  w/o Hubbard-U elements (model and reference agree on the functional):",
                 d[~d["has_specific_TM"].astype(bool)])
            emit("  with Hubbard-U elements (model is PBE+U, reference is PBE):",
                 d[d["has_specific_TM"].astype(bool)])

    # BVSE-NEB is the dataset's own empirical-pair-potential baseline; beating it is the
    # floor, not the target. Reported on the same items so the comparison is like-for-like.
    if "em_bvse" in conv.columns and conv["em_bvse"].notna().any():
        d = conv.dropna(subset=["em_dft", "em_bvse"])
        if len(d) >= 3:
            m = get_metrics(d["em_dft"].to_numpy(), d["em_bvse"].to_numpy())
            print(f"\n  BASELINE, BVSE-NEB on the same converged items: n={len(d)} "
                  f"MAE {m['mae']:.3f}  RMSE {m['rmse']:.3f}  Rp {m['Rp']:.3f}", flush=True)

    worst = conv.dropna(subset=["em_dft", "em_pred"]).copy()
    if len(worst):
        worst["abs_err"] = (worst["em_pred"] - worst["em_dft"]).abs()
        cols = [c for c in ("edge_id", "chemsys", "em_dft", "em_pred", "abs_err",
                            "pa_mae", "steps_band", "fmax_worst", "has_specific_TM")
                if c in worst.columns]
        top = worst.nlargest(min(20, len(worst)), "abs_err")[cols]
        print(f"\n  20 WORST converged hops by |E_m error|:", flush=True)
        print("    " + top.to_string(index=False, float_format=lambda x: f"{x:8.3f}")
              .replace("\n", "\n    "), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="Path to MACE .model checkpoint (required for the run).")
    p.add_argument("--model-name", default="mace-omat-custom",
                   help="Used for result/parquet filenames; keep stable across run + aggregate.")
    p.add_argument("--outdir", default="results_litraj_nebdft2k", help="Result + parquet output dir.")
    p.add_argument("--cache-dir", default="litraj_cache",
                   help="Where --prefetch unzips nebDFT2k/, and where run/aggregate read it.")
    p.add_argument("--dtype", default="float64", choices=["float32", "float64"],
                   help="float64 by default: barriers are small energy differences, and the "
                        "published notebook also uses float64.")
    p.add_argument("--cueq", action="store_true",
                   help="Route MACE tensor products through cuEquivariance fused CUDA kernels. "
                        "Warns + falls back to e3nn if the import fails.")
    p.add_argument("--split", choices=["all", "train", "val", "test"], default="all",
                   help="nebDFT2k data split. 'all' (default) matches the paper, which scores "
                        "every trajectory regardless of split because the target is the "
                        "downstream NEB task, not held-out property prediction.")
    p.add_argument("--element-policy", choices=["flag", "drop", "keep", "supported"],
                   default="supported",
                   help="Elements outside OMat24 support (Z in 1-56,72-83; nebDFT2k contains "
                        "Eu and La in 25 hops). supported/drop: remove those hops (default). "
                        "flag/keep: run them, has_unsupported is recorded either way.")
    p.add_argument("--tm-policy", choices=["flag", "drop", "keep"], default="flag",
                   help="Hops whose chemsys contains an MP/OMat24 Hubbard-U element "
                        "(Co/Cr/Fe/Mn/Mo/Ni/V/W) while the reference is plain PBE. "
                        "flag (default): run all, report the two splits separately (the "
                        "published layout). drop: score only the 646 U-free hops, where the "
                        "levels of theory actually agree. keep: single pooled metric.")
    p.add_argument("--k", type=float, default=PUB_K,
                   help=f"NEB spring constant in eV/A^2 (published: {PUB_K}).")
    p.add_argument("--n-steps", type=int, default=PUB_N_STEPS,
                   help=f"Step cap for the band and the target end-point (published: {PUB_N_STEPS}).")
    p.add_argument("--fmax", type=float, default=PUB_FMAX,
                   help=f"Convergence GATE in eV/A, on the max force component of band + both "
                        f"end-points (published: {PUB_FMAX}). Also the target end-point's fmax. "
                        f"The band itself relaxes to ASE's FIRE default of {PUB_BAND_FMAX}.")
    p.add_argument("--band-fmax", type=float, default=None,
                   help="Band convergence in eV/A. Default None reproduces the notebook, which "
                        "calls run(steps=...) with no fmax and so inherits ASE's FIRE default "
                        "of 0.05. Pass 0.1 to match LiTraj's own DFT driver "
                        "(scripts/ase_run_vasp.py), which relaxed the reference band at 0.1.")
    p.add_argument("--symmetric-endpoints", action="store_true",
                   help="SENSITIVITY CHECK ONLY, not paper-comparable. Relax BOTH end-points "
                        "with --endpoint-fmax/--endpoint-steps instead of reproducing the "
                        "notebook's asymmetric source (ASE defaults) / target (0.1, 100) pair.")
    p.add_argument("--endpoint-fmax", type=float, default=0.05,
                   help="Used only with --symmetric-endpoints.")
    p.add_argument("--endpoint-steps", type=int, default=100,
                   help="Used only with --symmetric-endpoints.")
    p.add_argument("--per-run-timeout", type=float, default=1200.0,
                   help="Wall-clock seconds PER ATTEMPT before a hop is abandoned. NEB is NOT "
                        "resumable, so attempts start fresh. This is also the only backstop "
                        "against the notebook's uncapped source relaxation. <=0 disables.")
    p.add_argument("--retries", type=int, default=1,
                   help="Extra fresh attempts after the first if a run times out or crashes.")
    p.add_argument("--limit", type=int, default=None,
                   help="Calibration: cap the work-list to the first N hops.")
    p.add_argument("--save-band", action="store_true", default=True,
                   help="Write the relaxed band to bands/<edge_id>_relaxed_<model>.xyz "
                        "(default on; the notebook does this for converged bands only).")
    p.add_argument("--no-save-band", dest="save_band", action="store_false")
    p.add_argument("--save-traj", action="store_true",
                   help="Also write per-hop optimiser logs + trajectories under opt/<edge_id>/. "
                        "Off by default: 1,681 x 3 files.")
    p.add_argument("--prefetch", action="store_true",
                   help="Login-node step: download + unzip nebDFT2k into --cache-dir, then exit.")
    p.add_argument("--aggregate", action="store_true",
                   help="Skip compute; glob JSON results into parquet + print metrics.")
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
