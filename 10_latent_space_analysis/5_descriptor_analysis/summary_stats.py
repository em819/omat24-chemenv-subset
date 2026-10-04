#!/usr/bin/env python3
"""Stage 10.5a -- summary statistics of the EnvTok descriptors in the MACE latent space.

Reads the outputs of stage 10.4 (chemenv_feature_metrics.py) and prints the
numbers quoted in the manuscript, for the descriptors with at least --min-atoms
atoms (default 30):

  main text, "Mapping the EnvTok descriptor to MLFF latent space", and SI S5.1:
    * number of analysed descriptors                         (18,445)
    * share forming a single connected component             (98.42 %)
    * number / share splitting into 2 or 3 components        (292, 1.6 %)
    * share compact, i.e. tighter than random same-element
      atom sets (compactness_z_element > 0)                   (96.9 %)
    * kNN label purity min / max / mean                       (0.003, 1.0, 0.4)
    * PCA dimensions and retained variance                    (50, > 94 %)
  SI S5.3:
    * share diffuse (compactness_z_element < 0)               (3.1 %)

Inputs (either or both):
  --report-card  the per-token parquet stage 10.4 writes (--out).
  --metrics-log  the console log stage 10.4 writes (submit_metrics.sh redirects
                 it to chemenv_metrics.log). Its '=== summary' table lists every
                 token with >= --min-size (20) atoms; its header carries the PCA
                 line. The table prints only some columns, so n_components_massive
                 (an alternative "number of modes" column) is reported only when
                 the parquet is given.
When both are given, every statistic is computed from each source and the two
are checked against each other (the log rounds floats to 6 decimals).

Environment: envB-mace (numpy, pandas, pyarrow). Seconds on a laptop.
"""

import argparse
import re
import sys

import numpy as np
import pandas as pd

# ============================================================================
# >>> USER CONFIGURATION -- pass these on the command line <<<
#
#   --report-card  stage 10.4's report-card parquet (submit_metrics.sh:
#                  REPORT_OUT, default report_card.parquet).
#   --metrics-log  stage 10.4's log (submit_metrics.sh writes chemenv_metrics.log
#                  in the directory it was submitted from).
#   --min-atoms    descriptors with at least this many atoms are analysed. The
#                  manuscript uses 30 (= k of the purity kNN). It must be >= the
#                  --min-size stage 10.4 ran with (20), because the expensive
#                  metrics exist only for tokens of that size.
# ============================================================================

# Columns of the '=== summary' table, in the order chemenv_feature_metrics.py
# prints them (mean_rel_force_error only when ref_fmag was present).
LOG_COLS_WITH_REL = ["chemenv_id", "n_atoms", "knn_purity", "compactness_z",
                     "compactness_z_element", "n_components", "largest_comp_frac",
                     "mean_force_error", "mean_rel_force_error", "dist_err_spearman"]
LOG_COLS_NO_REL = [c for c in LOG_COLS_WITH_REL if c != "mean_rel_force_error"]


def read_metrics_log(path):
    """Parse stage 10.4's log: the PCA header line and the '=== summary' table."""
    with open(path) as fh:
        lines = fh.read().splitlines()

    header = {}
    for ln in lines:
        m = re.search(r"PCA: (\d+) -> (\d+) dims \(([\d.]+)% var\)", ln)
        if m:
            header["pca_in"], header["pca_dims"] = int(m.group(1)), int(m.group(2))
            header["pca_var_pct"] = float(m.group(3))
        m = re.search(r"loaded ([\d,]+) atoms", ln)
        if m:
            header["atoms_loaded"] = int(m.group(1).replace(",", ""))
        m = re.search(r"dropping ([\d,]+) INVALID", ln)
        if m:
            header["atoms_invalid"] = int(m.group(1).replace(",", ""))
        if ln.startswith("=== summary"):
            break

    try:
        i0 = next(i for i, ln in enumerate(lines) if ln.startswith("=== summary"))
    except StopIteration:
        raise SystemExit(f"{path}: no '=== summary' table -- not a stage 10.4 log?")
    # "(N of M ids shown; ...)": the table is complete only if N == M, i.e. the run
    # used --summary-rows 0 (the published setting).
    m = re.match(r"\((\d+) of (\d+) ids shown", lines[i0 + 1])
    if not m or m.group(1) != m.group(2):
        raise SystemExit(f"{path}: summary table is truncated ({lines[i0 + 1]!r}); "
                         "rerun stage 10.4 with --summary-rows 0 or use --report-card")
    n_expected = int(m.group(2))
    cols = lines[i0 + 2].split()
    if cols not in (LOG_COLS_WITH_REL, LOG_COLS_NO_REL):
        raise SystemExit(f"{path}: unexpected summary columns {cols}")
    rows = []
    for ln in lines[i0 + 3:]:
        if not ln.strip():                      # the table ends at the first blank line
            break
        p = ln.split()                          # tokens contain no whitespace
        if len(p) != len(cols):
            raise SystemExit(f"{path}: malformed table row {ln!r}")
        rows.append(p)
    if len(rows) != n_expected:
        raise SystemExit(f"{path}: parsed {len(rows)} rows, header says {n_expected}")
    df = pd.DataFrame(rows, columns=cols)
    for c in cols[1:]:
        df[c] = pd.to_numeric(df[c])            # 'NaN' -> nan
    return df, header


def summarise(df, min_atoms):
    """All manuscript statistics for the tokens with n_atoms >= min_atoms."""
    if df["n_atoms"].min() > min_atoms:
        raise SystemExit(f"smallest token has {df['n_atoms'].min()} atoms: was stage 10.4 "
                         f"run with --min-size > {min_atoms}?")
    d = df[df["n_atoms"] >= min_atoms]
    for c in ("n_components", "compactness_z_element", "knn_purity"):
        if d[c].isna().any():
            raise SystemExit(f"{d[c].isna().sum()} analysed tokens lack {c}: was stage "
                             f"10.4 run with --min-size > {min_atoms}?")
    ncomp = d["n_components"].round().astype(int)
    n = len(d)
    s = dict(
        n_descriptors=n,
        n_atoms=int(d["n_atoms"].sum()),
        ncomp_counts={int(k): int(v) for k, v in ncomp.value_counts().sort_index().items()},
        single=int((ncomp == 1).sum()),
        split=int((ncomp >= 2).sum()),
        # compact = tighter than equally sized random sets of SAME-ELEMENT atoms
        # (the per-element null of stage 10.4, as SI S5.1 describes); a z of exactly
        # 0 would be neither, and is counted separately.
        compact=int((d["compactness_z_element"] > 0).sum()),
        diffuse=int((d["compactness_z_element"] < 0).sum()),
        z_elem_zero=int((d["compactness_z_element"] == 0).sum()),
        # the global null (random atoms of ANY element), for contrast only
        compact_global_null=int((d["compactness_z"] > 0).sum()),
        purity_min=float(d["knn_purity"].min()),
        purity_max=float(d["knn_purity"].max()),
        purity_mean=float(d["knn_purity"].mean()),
        purity_median=float(d["knn_purity"].median()),
        ids=set(d["chemenv_id"]),
    )
    if "n_components_massive" in d.columns:
        # Components holding >= 10 % of the token's atoms (stage 10.4 --massive-frac).
        # Not what the manuscript counts: see the printout.
        nm = d["n_components_massive"].round().astype(int)
        s["massive_counts"] = {int(k): int(v) for k, v in nm.value_counts().sort_index().items()}
        s["massive_single"] = int((nm == 1).sum())
        s["ncomp_vs_massive"] = pd.crosstab(ncomp, nm)
    return s


def report(s, label, header, min_atoms):
    n = s["n_descriptors"]
    pct = lambda x: 100.0 * x / n
    print(f"\n=== {label} ===")
    print(f"  analysed descriptors (n_atoms >= {min_atoms}): {n:,} "
          f"({s['n_atoms']:,} atoms)                                   [paper: 18,445]")
    print(f"  connected components (n_components, symmetric kNN graph, k_graph=15):")
    for k, v in s["ncomp_counts"].items():
        print(f"     {k} component(s): {v:6,d}  ({pct(v):6.3f} %)")
    print(f"  single connected component: {s['single']:,} = {pct(s['single']):.2f} %"
          f"                              [paper: 98.42 % (SI 98.4 %)]")
    print(f"  split into >= 2 components: {s['split']:,} = {pct(s['split']):.2f} %; "
          f"max components = {max(s['ncomp_counts'])}            [paper: 292 (1.6 %), 2 or 3]")
    if "massive_counts" in s:
        print(f"  (n_components_massive, components with >= 10 % of the atoms: "
              f"single {s['massive_single']:,} = {pct(s['massive_single']):.2f} %, "
              f"split {n - s['massive_single']:,} -- NOT the manuscript's count)")
    print(f"  compact  (compactness_z_element > 0): {s['compact']:,} = "
          f"{pct(s['compact']):.2f} %                     [paper: 96.9 %]")
    print(f"  diffuse  (compactness_z_element < 0): {s['diffuse']:,} = "
          f"{pct(s['diffuse']):.2f} %                         [paper SI S5.3: 3.1 %]")
    if s["z_elem_zero"]:
        print(f"  compactness_z_element == 0 exactly: {s['z_elem_zero']}")
    print(f"  (global null, compactness_z > 0: {pct(s['compact_global_null']):.2f} % -- "
          f"random atoms of ANY element; not the manuscript's null)")
    print(f"  kNN purity (k=30): min {s['purity_min']:.4f}, max {s['purity_max']:.4f}, "
          f"mean {s['purity_mean']:.4f}, median {s['purity_median']:.4f}"
          f"    [paper: 0.003, 1.0, 0.4]")
    if header:
        if "pca_dims" in header:
            print(f"  PCA (stage 10.4 log header): {header['pca_in']} -> "
                  f"{header['pca_dims']} dims, {header['pca_var_pct']:.1f} % variance"
                  f"          [paper: 50, > 94 %]")
        if "atoms_loaded" in header:
            print(f"  atoms loaded by stage 10.4: {header['atoms_loaded']:,}; INVALID "
                  f"dropped: {header.get('atoms_invalid', 0):,}   [paper: 12 million]")


def main():

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report-card", help="stage 10.4 report-card parquet")
    ap.add_argument("--metrics-log", help="stage 10.4 log with the '=== summary' table")
    ap.add_argument("--min-atoms", type=int, default=30)
    args = ap.parse_args()
    if not (args.report_card or args.metrics_log):
        ap.error("give --report-card and/or --metrics-log")

    results = {}
    header = {}
    if args.metrics_log:
        df_log, header = read_metrics_log(args.metrics_log)
        results["metrics log"] = summarise(df_log, args.min_atoms)
        report(results["metrics log"], f"from {args.metrics_log}", header, args.min_atoms)
    if args.report_card:
        rc = pd.read_parquet(args.report_card)
        results["report card"] = summarise(rc, args.min_atoms)
        report(results["report card"], f"from {args.report_card}", header, args.min_atoms)
        if "ncomp_vs_massive" in results["report card"]:
            print("\n  n_components (rows) vs n_components_massive (columns):")
            print(results["report card"]["ncomp_vs_massive"].to_string())
        if not args.metrics_log:
            print("  (PCA dims/variance: pass --metrics-log; the parquet does not record them)")

    if len(results) == 2:
        # Both sources must describe the same tokens and give the same statistics.
        # Integer counts must agree exactly; purity only to the log's 6 decimals.
        a, b = results["metrics log"], results["report card"]
        ok = True
        for key in ("n_descriptors", "n_atoms", "ncomp_counts", "single", "split",
                    "compact", "diffuse", "z_elem_zero", "compact_global_null", "ids"):
            if a[key] != b[key]:
                ok = False
                print(f"  MISMATCH {key}")
        for key in ("purity_min", "purity_max", "purity_mean", "purity_median"):
            if abs(a[key] - b[key]) > 1e-6:
                ok = False
                print(f"  MISMATCH {key}: {a[key]} vs {b[key]}")
        print(f"\nmetrics log vs report card: {'IDENTICAL' if ok else 'DIFFERENT'} "
              f"(counts exact, purity to 1e-6)")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
