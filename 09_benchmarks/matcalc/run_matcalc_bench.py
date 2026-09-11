#!/usr/bin/env python3
"""
Run the packaged MatCalc-Bench suite for a custom MACE model, sharded across
all SLURM tasks (one task per GPU). Each task processes a disjoint stride of
each benchmark's structure list and writes a per-rank CSV. Use merge_results.py
afterwards to concatenate and score.

Parallelism model
-----------------
MACE inference is the bottleneck and a single CUDA context cannot be shared
across joblib worker processes, so each task runs its shard serially (n_jobs=1)
on one pinned GPU. Throughput scales ~linearly with the number of tasks, since
the four benchmarks are embarrassingly parallel over structures.

Environment (set by srun)
-------------------------
SLURM_PROCID   -> global rank   (which stride this task owns)
SLURM_NTASKS   -> world size    (stride length)
SLURM_LOCALID  -> local rank    (which of the node's 4 GPUs to pin)
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from matcalc import EnergeticsCalc, RelaxCalc
from matcalc._stability import ELEMENTAL_REFS_DIR
from matcalc.benchmark import (
    ElasticityBenchmark,
    EquilibriumBenchmark,
    PhononBenchmark,
    SofteningBenchmark,
)
from matminer.featurizers.site import CrystalNNFingerprint
from matminer.featurizers.structure import SiteStatsFingerprint
from pymatgen.core import Structure
from monty.serialization import loadfn

# Elements the model is allowed to see. Atomic numbers 1-56 (H..Ba) and 72-83
# from Po (84) up -- i.e. the rare earths and the actinides the model was not
# trained to handle. Structures containing any other element are dropped before
# they are ever relaxed (see filter_allowed_inplace); merge_results.py applies
# the same cut at scoring time as a backstop (notably for Softening, which is
# indexed by material_id and cannot be filtered up front here).
ALLOWED_Z = frozenset(range(1, 57)) | frozenset(range(72, 84))


def get_calculator(model_path: str, device: str, dtype: str):
    """Wrap the user's MACE checkpoint in its native ASE calculator.

    float64 is the default on purpose: elasticity and phonon properties are
    finite differences of stress/energy, and float32 rounding noise propagates
    into the second derivatives (soft/imaginary modes, noisy elastic tensors).
    """
    from mace.calculators import MACECalculator

    return MACECalculator(model_paths=model_path, device=device, default_dtype=dtype)


def shard_inplace(bench, rank: int, world: int) -> None:
    """Restrict a Benchmark (or SofteningBenchmark) to this rank's stride."""
    if hasattr(bench, "structures"):  # EquilibriumBenchmark / Elasticity / Phonon
        bench.structures = bench.structures[rank::world]
        bench.ground_truth = bench.ground_truth[rank::world]
    elif hasattr(bench, "material_ids"):  # SofteningBenchmark
        bench.material_ids = bench.material_ids[rank::world]
    else:
        raise TypeError(f"Don't know how to shard {type(bench).__name__}")


def n_items(bench) -> int:
    if hasattr(bench, "structures"):
        return len(bench.structures)
    return len(bench.material_ids)


# Built-in matcalc reference tables that carry FIXED tabulated DFT energies
# reference *structures* only, with no energies, so it can only be used by relaxing
# them with the model.
TABULATED_REF_TABLES = frozenset({"MatPES-PBE", "MatPES-r2SCAN"})

# Elements whose 0 K ground state is a molecular / van-der-Waals solid (the
# diatomics H2/N2/O2/F2/Cl2/Br2/I2 and the noble gases). An OMat24-trained model
# never saw these low-density structures and relaxes them to nonsense, so their
# reference energy is READ from the tabulated DFT table instead of computed by the
# model. Every OTHER element forms a bulk solid the model handles well, so its
# reference IS computed self-consistently by relaxing with the model -- which cancels
# the model's arbitrary absolute-energy offset exactly (see class docstring).
# NOTE: this is a chemistry judgement. Hard bulk solids the model may also botch
# (e.g. boron) are NOT in this set; add them via --table-ref-elements if their
# compounds show large formation-energy errors.
DEFAULT_TABLE_REF_ELEMENTS = frozenset(
    {"H", "N", "O", "F", "Cl", "Br", "I", "He", "Ne", "Ar", "Kr", "Xe", "Rn"}
)


class EquilibriumBenchmarkHybridRefs(EquilibriumBenchmark):
    """EquilibriumBenchmark with HYBRID elemental references for formation energy.

    Stock matcalc's ``EquilibriumBenchmark.get_prop_calc`` calls
    ``_prepare_elemental_refs``, which relaxes EVERY element's MP-PBE reference
    structure with the model to obtain E_i. That is theoretically clean --
    computing both the compound and its references with the same model cancels the
    model's arbitrary per-element energy offset, so the formation energy is
    invariant to the absolute-energy scale. Its only failure is physical: an
    OMat24-trained model cannot relax the molecular O2/N2/F2 reference crystals it
    never saw, producing nonsensical references (observed: fluorides at +326 eV/atom).

    This subclass keeps the self-consistent (model-relaxed) reference for every
    element that forms a bulk solid, and substitutes a tabulated DFT reference only
    for the molecular elements in ``table_ref_elements``. Trade-off: the substitution
    re-introduces a small residual DFT-footing offset (OMat24 vs ``eform_refs``) for
    those few elements only -- tens of meV/atom, both being PBE, versus the multi-eV
    artifact it removes. The compound-side +U treatment in TM oxides/fluorides is a
    separate issue this does not address (it would need OMat24Compatibility corrections).
    """

    def __init__(self, *args, eform_refs: str = "MatPES-PBE",
                 table_ref_elements=DEFAULT_TABLE_REF_ELEMENTS,
                 all_tabulated: bool = False,
                 optimizer: str = "FIRE", **kwargs):
        super().__init__(*args, **kwargs)
        self.eform_refs_name = eform_refs if isinstance(eform_refs, str) else "custom"
        self.eform_table = self._load_table(eform_refs)
        self.table_ref_elements = frozenset(table_ref_elements)
        # all_tabulated: read EVERY element's reference from the tabulated DFT table
        # de-facto method and is immune to the model-relaxed-reference collapse that
        # poisons Eform for whichever elemental allotrope a given model botches
        self.all_tabulated = all_tabulated
        # Consumed here (not forwarded to super) so it never leaks into self.kwargs;
        # applied explicitly to both the compound and reference relaxations below.
        self.optimizer = optimizer

    @staticmethod
    def _load_table(eform_refs):
        """Load the tabulated DFT reference table as a dict (used for the molecular
        elements). Built-in name -> matcalc/elemental_refs/<name>-Element-Refs.json.gz;
        otherwise a path to a custom refs file (e.g. OMat24-DFT references)."""
        if isinstance(eform_refs, str) and eform_refs == "MP-PBE":
            raise ValueError(
                "MP-PBE has no tabulated energies (structures only). Use one of "
                f"{sorted(TABULATED_REF_TABLES)} or a custom refs file path."
            )
        if isinstance(eform_refs, str) and eform_refs in TABULATED_REF_TABLES:
            table = loadfn(ELEMENTAL_REFS_DIR / f"{eform_refs}-Element-Refs.json.gz")
        else:
            table = loadfn(eform_refs)
        missing = [el for el, v in table.items() if "energy_per_atom" not in v]
        if missing:
            raise ValueError(
                f"Reference table {eform_refs!r} lacks 'energy_per_atom' for: "
                f"{', '.join(missing[:10])}{'...' if len(missing) > 10 else ''}"
            )
        return table

    def _build_hybrid_refs(self, calculator):
        """Build {element: {'energy_per_atom': E_i}} for every element in this shard.

        Each element gets a PRIMARY source and a FALLBACK:
          * all_tabulated -> tabulated DFT table for every element (fallback: model);
          * else (hybrid) -> tabulated for the molecular `table_ref_elements`, model-
            relaxed for all other (bulk) elements (fallback: the other source).
        The model-relaxed path only runs when actually needed, so in all_tabulated
        mode no reference is ever relaxed (unless an element is absent from the table).
        Runs once per rank over the elements present in this shard's structures."""
        needed = {el.symbol for s in self.structures for el in s.composition.elements}
        mp_structs = loadfn(ELEMENTAL_REFS_DIR / "MP-PBE-Element-Refs.json.gz")
        # perturb_distance defaults to None here: references are relaxed unperturbed,
        # matching stock matcalc's _prepare_elemental_refs.
        relaxer = RelaxCalc(calculator, fmax=0.05, optimizer=self.optimizer)

        def model_ref(el):
            """Model-relaxed min energy_per_atom for `el`, or None on failure."""
            info = mp_structs.get(el)
            if info is None:
                return None
            s = info["structure"]
            s_list = [s] if not isinstance(s, list) else s
            energies = [r["energy"] / r["final_structure"].num_sites
                        for r in relaxer.calc_many(s_list) if r is not None]
            return min(energies) if energies else None

        def table_ref(el):
            """Tabulated DFT energy_per_atom for `el`, or None if absent from table."""
            info = self.eform_table.get(el)
            if info is None or "energy_per_atom" not in info:
                return None
            epa = info["energy_per_atom"]
            return min(epa) if isinstance(epa, list) else epa

        rank = os.environ.get("SLURM_PROCID", "0")
        hybrid, from_model, from_table = {}, [], []
        for el in sorted(needed):
            prefer_table = self.all_tabulated or (el in self.table_ref_elements)
            primary, fallback = ((table_ref, model_ref) if prefer_table
                                 else (model_ref, table_ref))
            e, src = primary(el), (from_table if prefer_table else from_model)
            if e is None:
                e, src = fallback(el), (from_model if prefer_table else from_table)
                if e is not None:
                    print(f"[rank {rank}] eform refs: {el} unavailable from "
                          f"{'table' if prefer_table else 'model'}, fell back to "
                          f"{'model' if prefer_table else 'table'}", flush=True)
            if e is None:
                print(f"[rank {rank}] eform refs: WARNING no reference for {el!r} "
                      f"(neither table nor model); its compounds will NaN-fail",
                      flush=True)
                continue
            hybrid[el] = {"energy_per_atom": e}
            src.append(el)
        print(f"[rank {rank}] eform refs: model-relaxed {sorted(from_model)}", flush=True)
        print(f"[rank {rank}] eform refs: tabulated[{self.eform_refs_name}] "
              f"{sorted(from_table)}", flush=True)
        return hybrid

    def get_prop_calc(self, calculator, **kwargs):
        # use_gs_reference=True makes EnergeticsCalc.calc read energy_per_atom straight
        # is decided in _build_hybrid_refs.
        return EnergeticsCalc(
            calculator,
            elemental_refs=self._build_hybrid_refs(calculator),
            use_gs_reference=True,
            fmax=0.05,
            perturb_distance=0.1,
            optimizer=self.optimizer,
            **kwargs,
        )

    def run(self, calculator, model_name, *, n_jobs=-1, checkpoint_file=None,
            checkpoint_freq=1000, delete_checkpoint_on_finish=True,
            include_full_results=False, **kwargs):
        """Override to compute the fingerprint distance NaN-safely.

        Stock ``EquilibriumBenchmark.run`` appends the ``d_<model>`` fingerprint
        distance with the guard ``if model is not None`` -- but a failed relaxation
        is stored as ``float('nan')``, and ``nan is not None`` is True, so it calls
        ``ssf.featurize(nan)`` and dies with "'float' object has no attribute
        'sites'". That crash takes out the fingerprint step for the ENTIRE shard, so
        every rank with even one failed structure loses its whole ``d_`` column (in
        the first run: 6/8 ranks -> only 27% fingerprint coverage). The crash is
        independent of the reference fix and would recur on every rerun.

        Here we call the grandparent ``Benchmark.run`` for the energetics, then build
        ``d_<model>`` ourselves with the SAME matcalc featurizer config but a guard
        that maps a non-Structure (a NaN failure) to NaN instead of featurizing it.
        Failures are excluded from the metric; every successful structure is scored.
        """
        results_df = super(EquilibriumBenchmark, self).run(
            calculator, model_name,
            n_jobs=n_jobs, checkpoint_file=checkpoint_file,
            checkpoint_freq=checkpoint_freq,
            delete_checkpoint_on_finish=delete_checkpoint_on_finish,
            include_full_results=include_full_results, **kwargs,
        )

        ssf = SiteStatsFingerprint(
            CrystalNNFingerprint.from_preset("ops", distance_cutoffs=None, x_diff_weight=0),
            stats=("mean", "std_dev", "minimum", "maximum"),
        )

        def _dist(model_s, dft_s):
            if not isinstance(model_s, Structure) or not isinstance(dft_s, Structure):
                return np.nan  # failed relaxation (NaN) -> excluded, not featurized
            try:
                return float(np.linalg.norm(
                    np.array(ssf.featurize(model_s)) - np.array(ssf.featurize(dft_s))))
            except Exception:
                return np.nan

        results_df[f"d_{model_name}"] = [
            _dist(m, d) for m, d in
            zip(results_df[f"structure_{model_name}"], results_df["structure_DFT"])
        ]
        return results_df


def build_benchmarks(limit: int | None, eform_refs: str = "MatPES-PBE",
                     table_ref_elements=DEFAULT_TABLE_REF_ELEMENTS,
                     all_tabulated: bool = False,
                     relax_deformed_structures: bool = True,
                     optimizer: str = "FIRE"):
    """Instantiate the four packaged benchmarks. `limit` (calibration) takes a
    random n_samples subset BEFORE sharding so a quick timing run is cheap.
    `eform_refs` is the tabulated DFT table used for the molecular `table_ref_elements`;
    all other elements' references are relaxed by the model (self-consistent) -- unless
    `all_tabulated` is set, in which case EVERY element's reference is read from the
    table and nothing is model-relaxed (robust; matches the paper's de-facto method).
    `relax_deformed_structures` (default True) relaxes the internal atomic positions of
    each strained cell before reading its stress -> RELAXED-ION elastic constants,
    matching the Materials Project DFT reference (which relaxes ionic DOF at every
    deformation). matcalc's own default is False (CLAMPED-ION), which is systematically
    too stiff -- worse for shear (G) than bulk (K) via the internal-strain/Kleinman
    effect -- so it inflates K and especially G vs the relaxed-ion MP ground truth.
    `optimizer` is the ASE optimizer for every relaxation (default matcalc's "FIRE").
    The three benchmarks that relax take it via different channels; Softening does no
    relaxation (forces only) so it is unaffected."""
    kw = {"n_samples": limit} if limit else {}
    return {
        # WBM PBE: fingerprint dist + Eform (hybrid model/tabulated references).
        "Equilibrium": EquilibriumBenchmarkHybridRefs(
            eform_refs=eform_refs, table_ref_elements=table_ref_elements,
            all_tabulated=all_tabulated, optimizer=optimizer, **kw),
        # ElasticityCalc has no `optimizer` arg -> pass it to the internal RelaxCalc
        # the per-deformation internal-coordinate relaxation).
        #
        # CORRECTED COMMENT (repo change; the code below is unchanged). The previous
        # text said the shear branch was reduced from matcalc's defaults and that fmax
        # was tightened from 0.1. Checked against matcalc 0.5.1, none of that holds:
        #
        #   _elasticity.py:51-52   norm_strains  default (-0.01, -0.005, 0.005, 0.01)
        #                          shear_strains default (-0.06, -0.03, 0.03, 0.06)
        #   benchmark.py:550       ElasticityBenchmark default fmax = 0.05
        #
        # The settings below therefore match the matcalc defaults on all three axes:
        # same +/-1% normal envelope, same +/-6% shear envelope, same fmax. The only
        # change is SAMPLING DENSITY, 4 -> 6 points per axis. That is arguably the
        # right choice for cross-model comparability -- but the catastrophic G_vrh
        # outliers must NOT be attributed to a narrowed strain window, because there
        # is none.
        #
        # Still true: `fmax` here feeds BOTH the parent cell relax and every
        # deformed-ion relax, and must NOT also appear in relax_calc_kwargs (RelaxCalc
        # would get a duplicate keyword).
        #
        # See CHANGES_FROM_AS_RUN.md (K2, K2b) -- in particular, ElasticityCalc
        # computes `r2_score_mean` and warns below r2_min=0.95, but the benchmark
        # layer discards it, so fit quality never reaches the CSV.
        "Elasticity": ElasticityBenchmark(
            norm_strains=(-0.01, -0.00666667, -0.00333333,0.00333333, 0.00666667, 0.01),
            shear_strains=(-0.06, -0.04, -0.02, 0.02, 0.04, 0.06),
            fmax=0.05,                                        # == the matcalc 0.5.1 default
            relax_calc_kwargs={"optimizer": optimizer},
            relax_deformed_structures=relax_deformed_structures,
            **kw),   # MP binaries: K_VRH,G_VRH
        # PhononCalc takes `optimizer` directly (pre-phonon relaxation).
        "Phonon": PhononBenchmark(optimizer=optimizer, **kw),    # Alexandria: C_V
        "Softening": SofteningBenchmark(**kw),       # WBM high-energy states: force slope
    }


def filter_allowed_inplace(bench):
    """Drop every structure containing an element outside ALLOWED_Z, in place,
    keeping `structures` and `ground_truth` aligned. Runs identically on every
    rank (before sharding) so the strides stay disjoint and reproducible.

    Returns (n_before, dropped) where `dropped` is a list of
    (reduced_formula, offending_symbols) for logging. Returns None for a
    benchmark that exposes only material_ids (SofteningBenchmark): it has no
    structures to inspect here, so its element cut happens in merge_results.py.
    """
    if not hasattr(bench, "structures"):
        return None
    structures = list(bench.structures)
    keep_idx, dropped = [], []
    for i, s in enumerate(structures):
        bad = sorted({el.symbol for el in s.composition.elements if el.Z not in ALLOWED_Z})
        if bad:
            dropped.append((s.composition.reduced_formula, ",".join(bad)))
        else:
            keep_idx.append(i)
    n_before = len(structures)
    bench.structures = [structures[i] for i in keep_idx]
    gt = bench.ground_truth
    if isinstance(gt, pd.DataFrame):
        bench.ground_truth = gt.iloc[keep_idx].reset_index(drop=True)
    else:
        bench.ground_truth = [gt[i] for i in keep_idx]
    return n_before, dropped


def count_failures(df: pd.DataFrame, model_name: str):
    """Rows the model failed to evaluate -- every numeric prediction column is
    NaN (a diverged/non-finite relaxation, which matcalc stores as NaN). Returns
    (n_failed, failed_subframe). Structure-object columns are non-numeric and
    coerce to all-NaN, so they are dropped before the test to avoid false hits.
    """
    pred_cols = [c for c in df.columns if model_name in c]
    if not pred_cols:
        return 0, df.iloc[0:0]
    num = df[pred_cols].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    if num.shape[1] == 0:
        return 0, df.iloc[0:0]
    failed_mask = num.isna().all(axis=1)
    return int(failed_mask.sum()), df[failed_mask]


def salvage_from_checkpoint(ckpt_path: Path, bench=None) -> pd.DataFrame | None:
    """Rebuild a results DataFrame from matcalc's checkpoint after the scorer
    crashed. The checkpoint is the full per-row results list (monty JSON) with
    ground-truth + predicted columns already populated and NaN for failures, so
    it is exactly what bench.run() would have returned minus any benchmark-
    specific post-processing column (e.g. Equilibrium's `d_<model>` fingerprint
    distance, which is what crashes). Good enough for merge_results.py, which
    scores paired `<x>`/`<x>_DFT` columns and ignores the rest.

    The checkpoint holds the RAW pre-rename column keys (matcalc saves it inside
    the compute loop, before Benchmark.run() applies property_rename_map at the
    very end). A clean rank therefore returns e.g. `Eform_*` while a salvaged rank
    would return `formation_energy_per_atom_*`, splitting the same metric across
    two column names when the shards are merged. Re-apply the benchmark's own
    rename map here so salvaged shards line up exactly with clean ones."""
    if not ckpt_path.exists():
        return None
    raw = ckpt_path.read_text().strip()
    if not raw:
        return None
    try:
        rows = json.loads(raw)               # stdlib json parses NaN/Infinity
    except json.JSONDecodeError:
        rows = json.loads(raw.replace("NaN", "null"))
    if not isinstance(rows, list) or not rows:
        return None
    df = pd.DataFrame(rows)

    # Mirror Benchmark.run()'s final rename step (which the crash skipped). Same
    # substring-replace semantics as matcalc, so DFT and prediction columns both
    # get renamed identically (formation_energy_per_atom -> Eform, etc.).
    rename_map = getattr(bench, "property_rename_map", None)
    if rename_map:
        def _rename(col: str) -> str:
            for k, v in rename_map.items():
                col = col.replace(k, v)
            return col
        df = df.rename(columns=_rename)
    return df


def failed_label(failed: pd.DataFrame) -> str:
    """Compact 'id (formula)' list of failed rows for the log."""
    id_col = next((c for c in ("material_id", "mp_id", "formula") if c in failed.columns), None)
    if id_col is None:
        return f"{len(failed)} rows"
    if "formula" in failed.columns and id_col != "formula":
        items = [f"{r[id_col]} ({r['formula']})" for _, r in failed.iterrows()]
    else:
        items = [str(v) for v in failed[id_col]]
    return ", ".join(items)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to MACE .model checkpoint")
    p.add_argument("--model-name", default="mace-omat-custom")
    p.add_argument("--outdir", default="results")
    p.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    p.add_argument("--limit", type=int, default=None,
                   help="Calibration: cap each benchmark to N structures (random).")
    p.add_argument("--only", nargs="*", default=None,
                   help="Subset of {Equilibrium,Elasticity,Phonon,Softening}.")
    p.add_argument("--eform-refs", default="MatPES-PBE",
                   help="Tabulated DFT elemental reference table used ONLY for the molecular "
                        "elements (--table-ref-elements) in the Equilibrium formation energy: "
                        "a built-in matcalc table ({MatPES-PBE,MatPES-r2SCAN}) or a path to a "
                        "custom refs file (e.g. OMat24-DFT). Should be on the same DFT footing "
                        "as the model for those elements (both PBE here).")
    p.add_argument("--table-ref-elements", nargs="*",
                   default=sorted(DEFAULT_TABLE_REF_ELEMENTS),
                   help="Element symbols whose formation-energy reference is READ from the "
                        "--eform-refs table instead of relaxed by the model (their 0K ground "
                        "state is a molecular/vdW solid the model can't relax). Defaults to "
                        "H N O F Cl Br I + noble gases; all OTHER elements are relaxed by the "
                        "model (scale-invariant). Pass your own list to override (e.g. add B), "
                        "or pass the flag with no elements to disable tabulation entirely.")
    p.add_argument("--optimizer", default="FIRE",
                   help="ASE optimizer for all relaxations (Equilibrium/Elasticity/Phonon; "
                        "Softening does none). Any ase.optimize name, e.g. FIRE (default), "
                        "BFGS, LBFGS, FIRE2. NOTE: BFGS/LBFGS build a Hessian and can take "
                        "large steps that DIVERGE far from equilibrium -- FIRE is usually the "
                        "more robust choice for the pathological relaxations seen here.")
    p.add_argument("--all-tabulated", action="store_true",
                   help="Read EVERY element's formation-energy reference from the "
                        "--eform-refs DFT table; do NOT model-relax any elemental "
                        "reference (overrides --table-ref-elements). This is the robust, "
                        "paper-comparable mode: it eliminates the model-relaxed-reference "
                        "collapse that poisons Eform for whichever elemental allotrope a "
                        "given model botches (boron for the full model, phosphorus for the "
                        "ncov60 subset). Recommended for Eform benchmarking.")
    p.add_argument("--clamped-ion", action="store_true",
                   help="Elasticity ONLY: compute stress on each strained cell WITHOUT "
                        "relaxing its internal atomic positions (clamped-ion elastic "
                        "constants). This is matcalc's own default, but it is "
                        "systematically too STIFF -- worse for shear G than bulk K "
                        "(internal-strain/Kleinman softening is omitted) -- and it is "
                        "INCONSISTENT with the Materials Project DFT reference, which "
                        "relaxes ionic DOF at every deformation. By default this script "
                        "relaxes the deformed cells (RELAXED-ION), matching the MP ground "
                        "truth. Pass --clamped-ion only for a fast/approximate run.")
    args = p.parse_args()

    rank = int(os.environ.get("SLURM_PROCID", 0))
    world = int(os.environ.get("SLURM_NTASKS", 1))
    local = int(os.environ.get("SLURM_LOCALID", 0))

    # Under SLURM cgroup GPU isolation (Meluxina) each task sees only its own GPU
    # as cuda:0, so torch.cuda.device_count() == 1 per task -> index by local rank
    # ONLY if this task can actually see all the node's GPUs; otherwise use cuda:0.
    if torch.cuda.is_available():
        device = f"cuda:{local}" if torch.cuda.device_count() > local else "cuda:0"
    else:
        device = "cpu"
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    table_ref = frozenset(args.table_ref_elements)

    relax_deformed = not args.clamped_ion
    print(f"[rank {rank}/{world}] host={socket.gethostname()} device={device} "
          f"dtype={args.dtype} optimizer={args.optimizer} eform_refs={args.eform_refs} "
          f"all_tabulated={args.all_tabulated} "
          f"table_ref_elements={'ALL' if args.all_tabulated else sorted(table_ref)} "
          f"elastic_ions={'clamped' if args.clamped_ion else 'relaxed'}",
          flush=True)

    calc = get_calculator(args.model, device, args.dtype)

    benchmarks = build_benchmarks(args.limit, args.eform_refs, table_ref,
                                  args.all_tabulated, relax_deformed, args.optimizer)
    if args.only:
        benchmarks = {k: v for k, v in benchmarks.items() if k in args.only}

    crashed = []        # benchmarks whose CSV could not be produced at all
    tally = []          # (name, n_attempted, n_failed) for the end-of-rank summary
    for name, bench in benchmarks.items():
        out_csv = outdir / f"{name}_{args.model_name}_rank{rank}of{world}.csv"
        ckpt = outdir / f".ckpt_{name}_rank{rank}of{world}.csv"

        if out_csv.exists():
            print(f"[rank {rank}] {name}: already done -> {out_csv.name}, skip", flush=True)
            continue

        # Element cut FIRST, identically on every rank, then shard the survivors.
        filt = filter_allowed_inplace(bench)
        if filt is not None:
            n_before, dropped = filt
            if dropped:
                ex = "; ".join(f"{f} [{b}]" for f, b in dropped[:8])
                more = "" if len(dropped) <= 8 else f" (+{len(dropped) - 8} more)"
                print(f"[rank {rank}] {name}: dropped {len(dropped)}/{n_before} "
                      f"structures with disallowed elements: {ex}{more}", flush=True)
        else:
            print(f"[rank {rank}] {name}: no structures to element-filter here "
                  f"(material_id-indexed); cut deferred to merge_results.py", flush=True)

        shard_inplace(bench, rank, world)
        n = n_items(bench)
        if n == 0:
            print(f"[rank {rank}] {name}: empty shard, skip", flush=True)
            continue

        print(f"[rank {rank}] {name}: {n} structures starting", flush=True)
        t0 = time.time()

        df = None
        try:
            # delete_checkpoint_on_finish=False: keep the checkpoint so a crash in
            # matcalc's post-relaxation scoring (see below) can be salvaged. We
            # delete it ourselves once the CSV is safely written.
            # SofteningBenchmark is NOT a Benchmark subclass: its run() takes only
            # include_full_results) -- it has neither n_jobs NOR
            # delete_checkpoint_on_finish, and it never deletes its own checkpoint.
            # Passing either kwarg raises TypeError before any work is done, so the
            # whole shard is lost. Give it a trimmed kwarg set. model_name IS a
            # required positional, same as the other benchmarks.
            common = dict(checkpoint_file=str(ckpt), checkpoint_freq=10,
                          delete_checkpoint_on_finish=False)
            if name == "Softening":
                soft_common = dict(checkpoint_file=str(ckpt), checkpoint_freq=10)
                df = bench.run(calc, args.model_name, **soft_common)
            else:
                df = bench.run(calc, args.model_name, n_jobs=1, **common)
        except Exception:
            # A single pathological structure must NOT lose this rank's whole shard.
            # Known failure mode: the relaxation diverges to a non-finite structure
            # NaN to ssf.featurize() -- its guard only checks `is not None`, so the
            # NaN slips through and it dies with "'float' object has no attribute
            # 'sites'". The expensive relaxations are already in the checkpoint, so
            # rebuild the DataFrame from it (minus the post-processing column that
            # crashed) instead of throwing the shard away.
            import traceback
            tb = traceback.format_exc()
            (outdir / f".failed_{name}_rank{rank}of{world}.txt").write_text(tb)
            df = salvage_from_checkpoint(ckpt, bench)
            if df is None or df.empty:
                print(f"[rank {rank}] {name}: CRASHED after {(time.time()-t0)/60:.1f} min "
                      f"and nothing salvageable from checkpoint -- shard lost\n{tb}",
                      flush=True)
                crashed.append(name)
                continue
            print(f"[rank {rank}] {name}: scorer crashed; salvaged {len(df)} relaxed "
                  f"rows from checkpoint (see .failed_{name}_rank{rank}of{world}.txt "
                  f"for the traceback)", flush=True)

        df.to_csv(out_csv, index=False)
        ckpt.unlink(missing_ok=True)        # success path: checkpoint no longer needed

        n_failed, failed = count_failures(df, args.model_name)
        tally.append((name, len(df), n_failed))
        dt = time.time() - t0
        msg = (f"[rank {rank}] {name}: wrote {len(df)} rows in {dt/60:.1f} min "
               f"({dt/max(n,1):.1f} s/structure) -> {out_csv.name}; "
               f"{n_failed} failed (NaN, excluded from scoring)")
        if n_failed:
            msg += f": {failed_label(failed)}"
        print(msg, flush=True)

    # Exit 0 even when something failed: a non-zero exit would (under the default
    # SLURM step behaviour) tear down every sibling rank. Surface it all instead.
    print(f"[rank {rank}] SUMMARY:", flush=True)
    for name, n_att, n_fail in tally:
        print(f"[rank {rank}]   {name}: {n_att} written, {n_fail} failed", flush=True)
    if crashed:
        print(f"[rank {rank}]   UNSALVAGEABLE benchmarks (no CSV): {', '.join(crashed)}",
              flush=True)
    total_fail = sum(f for _, _, f in tally)
    print(f"[rank {rank}] DONE: {len(tally)} benchmark CSV(s) written, "
          f"{total_fail} failed structure(s) total, {len(crashed)} benchmark(s) lost",
          flush=True)


if __name__ == "__main__":
    main()
