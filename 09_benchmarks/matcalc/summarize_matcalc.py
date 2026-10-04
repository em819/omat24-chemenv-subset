#!/usr/bin/env python3
"""
Stage 09 / MatCalc-Bench -- error distributions per model (Fig. 2, Fig. S4) and
the numbers the text quotes from them, from the CSVs merge_results.py writes.

Environment: envC-matcalc (numpy, pandas). CPU only, no model.

Input, per run: the --outdir of run_matcalc_bench.py after merge_results.py
(submit_matcalc.sh: $OUTDIR, default <submit dir>/results), holding one
<Benchmark>_<model_name>_FULL.csv per benchmark:
  Equilibrium   Eform_DFT, Eform_<model>, d_<model>
  Elasticity    K_vrh_DFT, G_vrh_DFT, K_vrh_<model>, G_vrh_<model>
  Phonon        CV_DFT, CV_<model>
  Softening     softening_scale_<model>
(merge_results.py has already removed the rows with out-of-support elements,
identically for every model; they are not part of any denominator here.)

Quantities, one value per structure:
  eform          |E_form(pred) - E_form(DFT)|            eV/atom
  fpdist         fingerprint distance d                   arb. units
  softening      1 - softening scale, in %                (> 0: forces too small)
  kvrh, gvrh, cv signed error 100 (pred - DFT) / DFT      %
  kvrh_abs, ...  absolute value of the same               %
Excluded from a quantity: a non-finite prediction (failed relaxation or
evaluation) and, for K, G and C_V with --unphysical drop (default, as in the
manuscript's violins), a non-positive prediction. A non-positive DFT value
would also be excluded (the percentage is undefined); none occurs.
excluded_pct = excluded / rows of the CSV, pooled over seeds as sum / sum.

Statistics per quantity: n, excluded %, median, mean, p5, p25, p75, p95
(numpy linear interpolation), pooled over the seeds of a model and per seed.
Random-seed replicas (labels starting with rdm / seed / random) are pooled per
--groups entry (default 'random'); pass --groups to compare several replica
sets (the SI degradation figure).

Backs (main text 'Near-equilibrium properties', Fig. 2; SI S4.2, Fig. S4):
  "For K, EnvTok gives the median signed error closest to zero, almost two
   percentage points below that of MACE-OMAT-0 [...] DIRECT produces the
   broadest distribution, with a mean error exceeding those of all other
   models by 6-7 percentage points. For G, subset training shifts the median
   error from 1.4% to 5.6-7.2 percentage points below zero. [...] the most
   negative median error (-7.2%)."
  "The mean force underestimation increases from 2.8% for MACE-OMAT-0 to 3.9%
   for random selection, 4.2% for DIRECT and EnvTok."
  "the mean C_V errors follow the same ordering as the mean force softening
   [...] EnvTok's 95th-percentile C_V error exceeds that of MACE-OMAT-0 by
   only about 2.5 percentage points."
  Conclusions: "the formation-energy MAE from 63-70 to 56 meV/atom".
  SI S4.2: "the shear modulus is increasingly underestimated (median -5.6%,
   -10.2% and -14.7%)"; per-seed medians scatter vs shift between sizes.
  Fig. 2 / Fig. S4: the excluded-% annotations and the totals per run.

    python summarize_matcalc.py \\
        --runs omat/results v14/results direct/results \\
               rdm1/results rdm2/results rdm3/results rdm4/results \\
        --labels MACE-OMAT-0 EnvTok DIRECT rdm1 rdm2 rdm3 rdm4 \\
        --out-dir matcalc_summary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================
# >>> USER CONFIGURATION -- pass these on the command line <<<
#
#   --runs       REQUIRED. One directory per benchmark run: the --outdir of
#                run_matcalc_bench.py, holding the four <Benchmark>_*_FULL.csv
#                written by merge_results.py.
#
#   --labels     One label per --runs entry, same order. A label starting with
#                rdm / seed / random marks a random-seed replica. Default: the
#                parent directory names.
#
#   --groups     Optional. One model name per --runs entry, same order, for
#                pooling runs into one model explicitly, e.g. three sets of
#                random replicas of different size. Default: replica labels go
#                to the group 'random', every other run is its own group.
#
#   --unphysical drop (default) excludes non-positive K, G and C_V predictions
#                as failed calculations, as the manuscript's figures do; keep
#                scores them as (large) errors. Non-finite values are excluded
#                either way.
#
#   --out-dir    Where the CSV is written (created if missing):
#                matcalc_summary.csv   one row per (model, seed or 'pooled',
#                                      quantity)
# ============================================================================

SEED_PREFIXES = ("rdm", "seed", "random")
STATS = ["median", "mean", "p5", "p25", "p75", "p95"]

# quantity: (benchmark, prediction prefix, kind, unit)
QUANTITIES = {
    "eform":     ("Equilibrium", "Eform_",           "abs_error", "eV/atom"),
    "fpdist":    ("Equilibrium", "d_",               "value",     "arb. units"),
    "softening": ("Softening",   "softening_scale_", "deficit",   "%"),
    "kvrh":      ("Elasticity",  "K_vrh_",           "signed_pct", "%"),
    "gvrh":      ("Elasticity",  "G_vrh_",           "signed_pct", "%"),
    "cv":        ("Phonon",      "CV_",              "signed_pct", "%"),
    "kvrh_abs":  ("Elasticity",  "K_vrh_",           "abs_pct",   "%"),
    "gvrh_abs":  ("Elasticity",  "G_vrh_",           "abs_pct",   "%"),
    "cv_abs":    ("Phonon",      "CV_",              "abs_pct",   "%"),
}
ID_COLS = ("material_id", "mp_id")


def is_replica(label: str) -> bool:
    return label.lower().startswith(SEED_PREFIXES)


def read_block(run_dir: Path, block: str) -> pd.DataFrame:
    hits = sorted(run_dir.glob(f"{block}_*_FULL.csv"))
    if len(hits) != 1:
        sys.exit(f"ERROR: expected exactly one {block}_*_FULL.csv in {run_dir}, found "
                 f"{[h.name for h in hits]}. Run merge_results.py first.")
    df = pd.read_csv(hits[0])
    idc = next((c for c in ID_COLS if c in df.columns), None)
    if idc is None or df[idc].duplicated().any():
        sys.exit(f"ERROR: {hits[0]}: missing or repeated structure ids "
                 f"(column {idc}); a shard was merged twice?")
    return df


def pred_col(df: pd.DataFrame, prefix: str, where: Path) -> str:
    cands = [c for c in df.columns if c.startswith(prefix) and not c.endswith("_DFT")]
    if len(cands) != 1:
        sys.exit(f"ERROR: {where}: expected one prediction column {prefix}<model>, "
                 f"got {cands}")
    return cands[0]


def values(df: pd.DataFrame, key: str, unphysical: str, where: Path) -> dict:
    """Valid values of one quantity in one run, with the exclusion counts."""
    _, prefix, kind, _ = QUANTITIES[key]
    pred = pd.to_numeric(df[pred_col(df, prefix, where)], errors="coerce").to_numpy(float)
    ok = np.isfinite(pred)
    n_unphysical = 0
    if kind in ("abs_error", "signed_pct", "abs_pct"):
        ref = pd.to_numeric(df[f"{prefix}DFT"], errors="coerce").to_numpy(float)
        ok &= np.isfinite(ref)
    if kind in ("signed_pct", "abs_pct"):
        ok &= ref > 0
        bad = ok & (pred <= 0)
        n_unphysical = int(bad.sum())
        if unphysical == "drop":
            ok &= ~bad
    n_failed = int((~np.isfinite(pred)).sum())
    p = pred[ok]
    if kind == "abs_error":
        v = np.abs(p - ref[ok])
    elif kind == "value":
        v = p
    elif kind == "deficit":
        v = 100.0 * (1.0 - p)
    else:
        v = 100.0 * (p - ref[ok]) / ref[ok]
        if kind == "abs_pct":
            v = np.abs(v)
    return {"values": v, "n_rows": len(df), "n_failed": n_failed,
            "n_unphysical": n_unphysical}


def stats(v: np.ndarray) -> dict:
    if v.size == 0:
        return {s: np.nan for s in STATS}
    q = np.percentile(v, [5, 25, 75, 95])
    return {"median": np.median(v), "mean": v.mean(),
            "p5": q[0], "p25": q[1], "p75": q[2], "p95": q[3]}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--groups", nargs="+")
    ap.add_argument("--unphysical", choices=["drop", "keep"], default="drop")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.labels = args.labels or [p.resolve().parent.name for p in args.runs]
    if len(args.labels) != len(args.runs):
        ap.error("--labels must have one entry per --runs entry")
    if args.groups is None:
        args.groups = ["random" if is_replica(l) else l for l in args.labels]
    if len(args.groups) != len(args.runs):
        ap.error("--groups must have one entry per --runs entry")
    pairs = list(zip(args.groups, args.labels))
    if len(set(pairs)) != len(pairs):
        ap.error("each (group, label) pair must be unique")
    return args


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    per = {}                                       # (group, label, key) -> values()
    n_rows = {}                                    # benchmark -> set of row counts
    for run_dir, lab, grp in zip(args.runs, args.labels, args.groups):
        blocks = {b: read_block(run_dir, b) for b in {q[0] for q in QUANTITIES.values()}}
        for b, df in blocks.items():
            n_rows.setdefault(b, set()).add(len(df))
        for key, (block, *_rest) in QUANTITIES.items():
            per[(grp, lab, key)] = values(blocks[block], key, args.unphysical, run_dir)
    for b, ns in n_rows.items():
        if len(ns) > 1:
            print(f"  WARNING: the runs have different numbers of {b} rows: {sorted(ns)}")

    rows = []
    groups = list(dict.fromkeys(args.groups))
    for grp in groups:
        labs = [l for g, l in zip(args.groups, args.labels) if g == grp]
        for key in QUANTITIES:
            scopes = [(l, [per[(grp, l, key)]]) for l in labs]
            if len(labs) > 1:
                scopes.insert(0, ("pooled", [per[(grp, l, key)] for l in labs]))
            for scope, parts in scopes:
                v = np.concatenate([p["values"] for p in parts])
                n_tot = sum(p["n_rows"] for p in parts)
                rows.append({
                    "group": grp, "scope": scope if len(labs) > 1 else "pooled",
                    "quantity": key, "unit": QUANTITIES[key][3],
                    "n_rows": n_tot, "n": v.size,
                    "n_failed": sum(p["n_failed"] for p in parts),
                    "n_unphysical": sum(p["n_unphysical"] for p in parts),
                    "unphysical_mode": args.unphysical,
                    "excluded_pct": 100.0 * (n_tot - v.size) / n_tot, **stats(v)})
    out = pd.DataFrame(rows)
    out.to_csv(args.out_dir / "matcalc_summary.csv", index=False)

    # ---- printout -----------------------------------------------------------
    pooled = out[out.scope == "pooled"]
    print(f"\nMatCalc-Bench (unphysical = {args.unphysical}); random seeds pooled")
    for key in QUANTITIES:
        q = pooled[pooled.quantity == key]
        print(f"\n--- {key} ({QUANTITIES[key][3]}); rows per run "
              f"{sorted(n_rows[QUANTITIES[key][0]])} ---")
        print(f"{'model':<16}{'n':>7}{'excl%':>7}" + "".join(f"{s:>10}" for s in STATS))
        for _, r in q.iterrows():
            print(f"{r.group[:15]:<16}{r.n:>7}{r.excluded_pct:>7.1f}"
                  + "".join(f"{r[s]:>10.4g}" for s in STATS))
        multi = out[(out.quantity == key) & (out.scope != "pooled")]
        multi = multi[multi.group.map(multi.group.value_counts()) > 1]
        for grp, m in multi.groupby("group", sort=False):
            print(f"  {grp} per-seed median {m['median'].min():.4g} .. {m['median'].max():.4g}"
                  f" (range {m['median'].max() - m['median'].min():.3g}), "
                  f"per-seed mean {m['mean'].min():.4g} .. {m['mean'].max():.4g}")
    print(f"\nwrote {args.out_dir / 'matcalc_summary.csv'}")


if __name__ == "__main__":
    main()
