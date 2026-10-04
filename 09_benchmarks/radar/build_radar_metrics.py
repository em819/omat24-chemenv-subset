#!/usr/bin/env python3
"""
Stage 09 / summary -- the twelve spokes of the radar figure (Fig. 'radar',
radar_benchmarks_v2) from the outputs of the four benchmark analyses.

Environment: any of the stage-09 envs (numpy, pandas only); envD-mliparena
is the one the three summaries below run in. CPU only.

Inputs (all written by repository scripts):
  --matcalc     matcalc_summary.csv of matcalc/summarize_matcalc.py, run with
                --unphysical drop (the manuscript's convention)
  --stability   stability_npt_summary.csv of stability/summarize_stability.py
  --nebdft2k    nebdft2k_table.csv of nebdft2k/table_nebdft2k.py
  --matpes-tables   the --aggregate_out directory of
                matpes/eval_with_report_efs.py --aggregate (committed copy:
                matpes/paper_tables/): peratom_force_mae_bulk.csv,
                peratom_force_mae_non_bulk.csv, paper_tables.json
  --matpes-runs the per-model `*_paper` directories of
                eval_with_report_efs.py (per_atom_forces.npz,
                per_structure.npz), for the non-bulk median, which the
                aggregate tables do not keep

Spokes, per model (reference model in brackets in the manuscript figure):
  F, bulk        pooled per-atom force MAE over the bulk domain, component
                 convention (MACE's MAE_F), meV/A -- table row 'comp'
  F, non-bulk    MEDIAN over the non-bulk atoms of the per-atom component-
                 averaged error (|dFx|+|dFy|+|dFz|)/3, meV/A; the mean of the
                 same quantity is the table's pooled MAE, which is checked,
                 as is the atom count
  E_form         mean |E_form error|, eV/atom
  fingerprint    mean fingerprint distance
  softening      mean force underestimation, 100 (1 - mean softening scale), %
  K, G, C_V      mean absolute percentage error (MAPE)
  MD completion  mean completed fraction of the NPT trajectory, % (all
                 protocol runs; unsnapped normalized_final_step)
  MD stability   % of NPT runs not physically diverged (CUDA OOM NOT counted
                 as diverged, as in the published figure; the variant with OOM
                 counted is printed)
  E_a (no U), E_a (+U)   nebDFT2k barrier MAE per Hubbard-U split, eV
Random replicas: MatPES and nebDFT2k entries are the mean over the seeds (the
tables' seed_mean / Table II); MatCalc and stability entries pool the seeds,
as the corresponding figures do.

Output: radar_metrics.csv, one row per spoke in plotting order with columns
label, better, <models...>; better = lower (an error: the radar plots
reference/model) or higher (a success rate: model/reference). Same layout as
the CSV the radar plotting script reads.

Backs: Fig. 'radar' and its caption ("x0.7 on the softening axis corresponds
to -2.8% for MACE-OMAT-0 versus -3.9 to -4.2% for the subset-trained models";
"For non-bulk structures, the median per-atom force error is shown"), and
the conclusions' "bulk force MAE from 149--154 to 126 meV/A and the
formation-energy MAE from 63--70 to 56 meV/atom".

    python build_radar_metrics.py \\
        --matcalc matcalc_summary/matcalc_summary.csv \\
        --stability stability_summary/stability_npt_summary.csv \\
        --nebdft2k nebdft2k_table/nebdft2k_table.csv \\
        --matpes-tables ../matpes/paper_tables \\
        --matpes-runs v14_paper omat0_paper direct_paper \\
                      rdm1_paper rdm2_paper rdm3_paper rdm4_paper \\
        --matpes-labels v14 OMAT0 DIRECT rdm1 rdm2 rdm3 rdm4 \\
        --models MACE-OMAT-0 EnvTok DIRECT random \\
        --matpes-columns OMAT0 v14 DIRECT rdm1,rdm2,rdm3,rdm4 \\
        --reference MACE-OMAT-0 --out radar_metrics.csv
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================
# >>> USER CONFIGURATION -- pass these on the command line <<<
#
#   --matcalc / --stability / --nebdft2k   REQUIRED. The summary CSVs above.
#                Their model names (column 'group') must be the --models.
#
#   --matpes-tables  REQUIRED. Directory with the aggregate tables.
#
#   --matpes-runs    REQUIRED. One `*_paper` directory per MatPES model, and
#   --matpes-labels  one label per directory, the labels the aggregate was run
#                    with (submit_aggregate.sh LABELS: v14 OMAT0 DIRECT rdm1..4).
#
#   --models         REQUIRED. The models of the radar, in column order, by
#                    their names in the three summaries (e.g. MACE-OMAT-0
#                    EnvTok DIRECT random).
#
#   --matpes-columns REQUIRED. For every --models entry, its MatPES label; a
#                    comma-separated list of labels for a replica model, whose
#                    value is then the mean over those labels.
#
#   --reference      The --models entry the radar is normalised to (printout
#                    only). Default: the first --models entry.
#
#   --out            Output CSV path. Default radar_metrics.csv.
# ============================================================================

SPOKES = [  # label, better, unit for the printout
    ("Force MAE, bulk", "lower", "meV/A"),
    ("Force MAE, non-bulk", "lower", "meV/A"),
    ("Formation energy MAE", "lower", "eV/atom"),
    ("Fingerprint distance", "lower", ""),
    ("Force softening", "lower", "%"),
    ("Bulk modulus K MAE", "lower", "%"),
    ("Shear modulus G MAE", "lower", "%"),
    ("Heat capacity C_V MAE", "lower", "%"),
    ("Trajectory completed", "higher", "%"),
    ("Runs without divergence", "higher", "%"),
    ("Barrier MAE, w/o Hubbard-U", "lower", "eV"),
    ("Barrier MAE, with Hubbard-U", "lower", "eV"),
]
MATCALC_SPOKE = {"Formation energy MAE": "eform", "Fingerprint distance": "fpdist",
                 "Force softening": "softening", "Bulk modulus K MAE": "kvrh_abs",
                 "Shear modulus G MAE": "gvrh_abs", "Heat capacity C_V MAE": "cv_abs"}


def read_table(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """An aggregate table: '#' comment lines, then CSV indexed by 'convention'."""
    lines = path.read_text().splitlines(keepends=True)
    comments = [l for l in lines if l.startswith("#")]
    df = pd.read_csv(io.StringIO("".join(l for l in lines if not l.startswith("#"))))
    return df.set_index("convention"), comments


def table_atoms(comments: list[str], path: Path) -> int:
    for l in comments:
        if "atoms (sum of natoms" in l:
            return int(l.split(",")[1].split()[0])
    sys.exit(f"ERROR: {path}: no '... atoms (sum of natoms ...)' header line")


def nonbulk_mask(st, detector: str | None) -> np.ndarray:
    """Non-bulk structures exactly as the aggregate tables define them."""
    if detector is None:                       # tables used the stored split
        return st["domain"] == "non_bulk"
    if detector == "dim":                      # overcompressed screen kept as stored
        return (st["dim"] < 3) & (st["domain"] != "overcompressed")
    sys.exit(f"ERROR: the tables were built with --aggregate_non_bulk_detector "
             f"{detector!r}; only 'dim' (the paper's) and the stored split are "
             f"implemented here.")


def nonbulk_median(run_dir: Path, detector, n_expected: int, mae_expected: float,
                   label: str) -> float:
    for f in ("per_structure.npz", "per_atom_forces.npz"):
        if not (run_dir / f).is_file():
            sys.exit(f"ERROR: {run_dir / f} not found; --matpes-runs must be the *_paper "
                     f"directories of eval_with_report_efs.py")
    with np.load(run_dir / "per_structure.npz") as st, \
            np.load(run_dir / "per_atom_forces.npz") as pa:
        counts = np.diff(pa["struct_offset"])
        if counts.size != st["sid"].size:
            sys.exit(f"ERROR: {run_dir}: per_atom_forces.npz and per_structure.npz "
                     f"disagree on the number of structures")
        sel = np.repeat(nonbulk_mask(st, detector), counts)
        e = np.abs(pa["d_force"][sel].astype(np.float64)).mean(axis=1) * 1000.0
    e[~np.isfinite(e)] = np.inf                # a divergent atom ranks last
    if e.size != n_expected:
        sys.exit(f"ERROR: {label}: {e.size} non-bulk atoms, the table has {n_expected}")
    if np.isfinite(e).all() and abs(e.mean() / mae_expected - 1) > 1e-5:
        sys.exit(f"ERROR: {label}: non-bulk mean {e.mean():.6g} meV/A differs from the "
                 f"table's {mae_expected:.6g}; wrong *_paper directory?")
    return float(np.median(e))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matcalc", type=Path, required=True)
    ap.add_argument("--stability", type=Path, required=True)
    ap.add_argument("--nebdft2k", type=Path, required=True)
    ap.add_argument("--matpes-tables", type=Path, required=True)
    ap.add_argument("--matpes-runs", nargs="+", type=Path, required=True)
    ap.add_argument("--matpes-labels", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--matpes-columns", nargs="+", required=True)
    ap.add_argument("--reference")
    ap.add_argument("--out", type=Path, default=Path("radar_metrics.csv"))
    args = ap.parse_args()
    if len(args.matpes_runs) != len(args.matpes_labels):
        ap.error("--matpes-labels must have one entry per --matpes-runs entry")
    if len(args.matpes_columns) != len(args.models):
        ap.error("--matpes-columns must have one entry per --models entry")
    args.reference = args.reference or args.models[0]
    if args.reference not in args.models:
        ap.error("--reference must be one of --models")
    return args


def main() -> None:
    args = parse_args()
    vals = {s[0]: {} for s in SPOKES}
    mp_cols = {m: c.split(",") for m, c in zip(args.models, args.matpes_columns)}

    # ---- the three summaries must all carry every --models entry ---------------
    for path in (args.matcalc, args.stability, args.nebdft2k):
        missing = set(args.models) - set(pd.read_csv(path)["group"])
        if missing:
            sys.exit(f"ERROR: {path} has no model(s) {sorted(missing)}; the names in "
                     f"--models must be the 'group' names of the summaries")

    # ---- MatPES ---------------------------------------------------------------
    bulk, _ = read_table(args.matpes_tables / "peratom_force_mae_bulk.csv")
    nb_path = args.matpes_tables / "peratom_force_mae_non_bulk.csv"
    nonbulk, nb_comments = read_table(nb_path)
    meta = json.loads((args.matpes_tables / "paper_tables.json").read_text())
    detector = (meta.get("rederived_domains") or {}).get("detector")
    n_nb = table_atoms(nb_comments, nb_path)
    run_of = dict(zip(args.matpes_labels, args.matpes_runs))
    med = {}
    for lab in sorted({l for ls in mp_cols.values() for l in ls}):
        if lab not in run_of:
            sys.exit(f"ERROR: MatPES label {lab!r} has no --matpes-runs entry")
        med[lab] = nonbulk_median(run_of[lab], detector, n_nb,
                                  float(nonbulk.loc["comp", lab]), lab)
    for m, labs in mp_cols.items():
        vals["Force MAE, bulk"][m] = float(np.mean([bulk.loc["comp", l] for l in labs]))
        vals["Force MAE, non-bulk"][m] = float(np.mean([med[l] for l in labs]))
        if len(labs) > 1 and "seed_mean" in bulk.columns:
            sm = float(bulk.loc["comp", "seed_mean"])
            if abs(vals["Force MAE, bulk"][m] / sm - 1) > 1e-6:
                print(f"  WARNING: mean of {labs} ({vals['Force MAE, bulk'][m]:.6g}) is not "
                      f"the table's seed_mean ({sm:.6g}): other replicas in the table?")

    # ---- MatCalc --------------------------------------------------------------
    mc = pd.read_csv(args.matcalc)
    if "unphysical_mode" in mc and set(mc["unphysical_mode"]) != {"drop"}:
        sys.exit(f"{args.matcalc} was built with --unphysical "
                 f"{sorted(set(mc['unphysical_mode']))}; the radar uses 'drop'")
    mc = mc[mc["scope"] == "pooled"].set_index(["group", "quantity"])
    for spoke, q in MATCALC_SPOKE.items():
        for m in args.models:
            vals[spoke][m] = float(mc.loc[(m, q), "mean"])

    # ---- MLIP-Arena stability -------------------------------------------------
    st = pd.read_csv(args.stability).set_index("group")
    for m in args.models:
        vals["Trajectory completed"][m] = float(st.loc[m, "mean_completion_pct"])
        vals["Runs without divergence"][m] = 100.0 - float(st.loc[m, "phys_div_pct"])

    # ---- nebDFT2k -------------------------------------------------------------
    nb = pd.read_csv(args.nebdft2k).set_index(["group", "split"])
    for m in args.models:
        vals["Barrier MAE, w/o Hubbard-U"][m] = float(nb.loc[(m, "w/o Hubbard-U"), "mae"])
        vals["Barrier MAE, with Hubbard-U"][m] = float(nb.loc[(m, "with Hubbard-U"), "mae"])

    # ---- output ---------------------------------------------------------------
    rows = [{"label": lab, "better": better,
             **{m: f"{vals[lab][m]:.6g}" for m in args.models}}
            for lab, better, _ in SPOKES]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)

    ref = args.reference
    print(f"\nRadar spokes (value; ratio to {ref}: ref/model for errors, "
          f"model/ref for success rates)")
    print(f"{'spoke':<30}" + "".join(f"{m[:18]:>20}" for m in args.models))
    for lab, better, unit in SPOKES:
        r = vals[lab][ref]
        cells = []
        for m in args.models:
            v = vals[lab][m]
            ratio = r / v if better == "lower" else v / r
            cells.append(f"{v:>10.4g} (x{ratio:.2f})")
        print(f"{lab[:29]:<30}" + "".join(f"{c:>20}" for c in cells) + f"  {unit}")
    oom = 100.0 - st["phys_div_incl_oom_pct"]
    print("\nMD stability with CUDA OOM counted as diverged: "
          + ", ".join(f"{m} {oom[m]:.4g}%" for m in args.models))
    print(f"MatPES non-bulk domain: detector {detector!r}, {n_nb} atoms; "
          f"per-label medians: " + ", ".join(f"{k} {v:.4g}" for k, v in med.items()))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
