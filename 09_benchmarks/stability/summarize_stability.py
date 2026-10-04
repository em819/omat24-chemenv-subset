#!/usr/bin/env python3
"""
Stage 09 / MLIP-Arena stability -- outcome of every NPT compression run, per
model, from the files run_stability_bench.py writes.

Environment: envD-mliparena (numpy, pandas, pyarrow). CPU only, no model.

Input, per run: the --outdir of run_stability_bench.py (submit_stability.sh:
$OUTDIR, default <submit dir>/results), after its --aggregate step. Read from it:
  <model_name>-compression.parquet   one row per trajectory FRAME, with the
                                     per-trajectory `source`, `natoms`,
                                     `normalized_final_step` and the per-frame
                                     `temperatures` (written by --aggregate)
  .failed_<traj stem>.txt            traceback of the last failed MD attempt
  .oom_<traj stem>.txt               the last attempt ended in CUDA OOM
Nothing else is needed (no SLURM logs, no trajectories).

Per NPT run (the protocol's --n-npt structures, 80 in the paper), one outcome,
in this order of precedence:
  survived    normalized_final_step >= 0.999 (a finished trajectory records
              ~0.9996, never exactly 1)
  oom         CUDA out-of-memory traceback in a marker file
  diverged    "physically diverged": a singular NPT cell matrix (cell collapse)
              in a marker file, or, with no marker, a temperature that is
              non-finite or above 1e5 K in any recorded frame
  other       any other traceback
  truncated   none of the above: still integrating when the 3 x 600 s
              wall-clock budget ran out
  missing     no trajectory in the parquet; counts as completion 0
A finished run is 'survived' even if a temperature above 1e5 K was recorded on
the way (the arena metric is the step reached); such runs are counted in
n_survived_with_temp_blowup.
This is the classification of the manuscript's Fig. 3b (the original analysis
script plot_stability_npt_single.py). The text defines physical divergence as
including memory exhaustion, but the published 39 % / 27 % / 8 % / 1-of-80
exclude OOM; both variants are written (phys_div_pct, phys_div_incl_oom_pct).

Times: a run's end time is normalized_final_step x the trajectory length; the
temperature and pressure at that time follow the protocol's linear ramps
(300 -> 3000 K, 0 -> 500 GPa over 10 ps; mlip_arena.tasks.md interpolates the
schedule linearly in the step index).

Random-seed replicas (labels starting with rdm / seed / random) are pooled per
--groups entry (default group 'random'): pooled counts are over all their
runs, and every rate is also given per seed and as seed mean +- sample SD.

Backs (main text, 'Molecular dynamics stability'; Fig. 3; SI S4.3, Fig. S5):
  "MACE-OMAT-0 sustains all runs until 3.1 ps (1147 K, 157 GPa), but failures
   then accumulate rapidly: only one-third of the runs finish, with a mean
   trajectory completion of 75%. The random models start failing earlier
   (1.1--2.3 ps), yet complete 81% of the trajectory on average, with half of
   the runs finishing. EnvTok and DIRECT achieve mean completion rates of 92%
   and 96%, respectively. [...] Truncation rates are similar across models,
   whereas physical divergence affects 39% of MACE-OMAT-0 runs, 27% for random
   selection on average, 8% for EnvTok, and only 1 of 80 runs for DIRECT."

    python summarize_stability.py \\
        --runs omat/results v14/results direct/results \\
               rdm1/results rdm2/results rdm3/results rdm4/results \\
        --labels MACE-OMAT-0 EnvTok DIRECT rdm1 rdm2 rdm3 rdm4 \\
        --out-dir stability_summary
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================
# >>> USER CONFIGURATION -- pass these on the command line <<<
#
#   --runs       REQUIRED. One directory per benchmark run: the --outdir of
#                run_stability_bench.py, holding <model_name>-compression.parquet
#                (from its --aggregate step) and the .failed_/.oom_ markers.
#
#   --labels     One label per --runs entry, same order. A label starting with
#                rdm / seed / random marks a random-seed replica; replicas are
#                pooled into one model. Default: the parent directory names.
#
#   --groups     Optional. One model name per --runs entry, same order, for
#                pooling several runs into one model explicitly (e.g. two sets
#                of random replicas). Default: every replica label goes to the
#                group 'random', every other run is its own group (its label).
#
#   --n-npt      NPT runs the protocol asked for per benchmark run
#                (run_stability_bench.py --n-npt). Default 80. Runs with no
#                trajectory count as 'missing' with completion 0.
#
#   --total-time-ps / --t-start / --t-end / --p-end-gpa
#                The NPT protocol (run_stability_bench.py --total-time 1e4 fs,
#                temperature [300, 3000] K, pressure [0, 500] GPa). Used only to
#                convert the end of a run into time, temperature and pressure.
#                Defaults 10, 300, 3000, 500.
#
#   --out-dir    Where the three CSVs are written (created if missing):
#                stability_npt_runs.csv      one row per NPT run
#                stability_npt_per_seed.csv  one row per benchmark run
#                stability_npt_summary.csv   one row per model (seeds pooled)
# ============================================================================

SEED_PREFIXES = ("rdm", "seed", "random")   # as eval_with_report_efs.py --aggregate
SURVIVED = 0.999            # normalized_final_step of a finished run (~0.9996)
T_BLOWUP = 1e5              # K; the run targets <= 3000 K
OUTCOMES = ["survived", "diverged", "oom", "other", "truncated", "missing"]
_IDX_RE = re.compile(r"_(\d+)_npt$")


def is_replica(label: str) -> bool:
    return label.lower().startswith(SEED_PREFIXES)


def marker_mode(text: str) -> str:
    """Cause of failure written by run_stability_bench.run_one."""
    low = text.lower()
    if "out of memory" in low or "outofmemoryerror" in low or "cuda-oom" in low:
        return "oom"
    if "singular matrix" in low:
        return "singular_matrix"
    return "other"


def load_run(run_dir: Path, n_npt: int) -> pd.DataFrame:
    """One row per protocol NPT run (index 0..n_npt-1), classified."""
    hits = sorted(run_dir.glob("*-compression.parquet"))
    if len(hits) != 1:
        sys.exit(f"ERROR: expected exactly one *-compression.parquet in {run_dir}, "
                 f"found {[h.name for h in hits]}. Run run_stability_bench.py "
                 f"--aggregate first.")
    frames = pd.read_parquet(hits[0], columns=["source", "natoms",
                                               "normalized_final_step", "temperatures"])
    temps = pd.to_numeric(frames["temperatures"], errors="coerce")
    frames = frames.assign(_bad_t=~np.isfinite(temps) | (temps.abs() > T_BLOWUP))
    g = frames.groupby("source", sort=True)
    per = pd.DataFrame({
        "natoms": g["natoms"].first(),
        "nfs": pd.to_numeric(g["normalized_final_step"].first(), errors="coerce"),
        "n_nfs_values": g["normalized_final_step"].nunique(dropna=False),
        "temp_blowup": g["_bad_t"].any(),
    }).reset_index()
    if (per["n_nfs_values"] > 1).any():
        sys.exit(f"ERROR: {hits[0]}: normalized_final_step is not constant within "
                 f"a trajectory for {list(per.source[per.n_nfs_values > 1])}.")
    per["idx"] = per["source"].str.extract(_IDX_RE)[0].astype(float)
    if per["idx"].isna().any():
        sys.exit(f"ERROR: {hits[0]}: cannot read the structure index from "
                 f"{list(per.source[per.idx.isna()])}.")
    per["idx"] = per["idx"].astype(int)
    if per["idx"].duplicated().any():
        sys.exit(f"ERROR: {hits[0]}: structure index repeated: "
                 f"{sorted(per.idx[per.idx.duplicated()])}.")
    extra = per[(per["idx"] < 0) | (per["idx"] >= n_npt)]
    if len(extra):
        sys.exit(f"ERROR: {hits[0]}: indices {sorted(extra.idx)} lie outside "
                 f"0..{n_npt - 1}; check --n-npt.")

    # Marker files are named after the trajectory stem, i.e. the parquet's source.
    modes: dict[str, str] = {}
    for prefix in (".failed_", ".oom_"):      # .oom_ last: it names the final attempt
        for p in sorted(run_dir.glob(f"{prefix}*_npt.txt")):
            stem = p.name[len(prefix):-len(".txt")]
            modes[stem] = marker_mode(p.read_text(errors="replace"))
    orphan = sorted(set(modes) - set(per["source"]))
    if orphan:
        print(f"  WARNING {run_dir}: {len(orphan)} NPT marker(s) without a trajectory "
              f"in the parquet (counted under 'missing'): {orphan}")
    per["marker"] = per["source"].map(modes)

    survived = per["nfs"] >= SURVIVED
    out = np.full(len(per), "truncated", dtype=object)
    out[(per["marker"] == "other").to_numpy()] = "other"
    out[((per["marker"] == "singular_matrix")
         | (per["marker"].isna() & per["temp_blowup"])).to_numpy()] = "diverged"
    out[(per["marker"] == "oom").to_numpy()] = "oom"
    out[survived.to_numpy()] = "survived"
    # A trajectory whose parquet row has no usable step cannot be scored.
    out[per["nfs"].isna().to_numpy()] = "missing"
    per["outcome"] = out

    present = set(per["idx"])
    absent = [i for i in range(n_npt) if i not in present]
    if absent:
        print(f"  WARNING {run_dir}: {len(absent)} of {n_npt} NPT run(s) have no "
              f"trajectory -> 'missing', completion 0: indices {absent}")
        per = pd.concat([per, pd.DataFrame({
            "source": "", "natoms": np.nan, "nfs": np.nan, "n_nfs_values": 0,
            "temp_blowup": False, "idx": absent, "marker": None,
            "outcome": "missing"})], ignore_index=True)
    per["completion"] = per["nfs"].fillna(0.0).clip(lower=0.0)
    return per.sort_values("idx").reset_index(drop=True).drop(columns="n_nfs_values")


def at_time(frac, args) -> dict:
    """Time, temperature and pressure of the protocol ramps at fraction `frac`
    of the trajectory (scalar or array; NaN stays NaN)."""
    return {"ps": frac * args.total_time_ps,
            "K": args.t_start + frac * (args.t_end - args.t_start),
            "GPa": frac * args.p_end_gpa}


def rates(per: pd.DataFrame, args) -> dict:
    """Counts and rates for one set of runs (one seed, or several pooled)."""
    n = len(per)
    c = {f"n_{o}": int((per["outcome"] == o).sum()) for o in OUTCOMES}
    r = {"n_runs": n, **c}
    for o in OUTCOMES:
        r[f"{o}_pct"] = 100.0 * c[f"n_{o}"] / n
    r["phys_div_pct"] = r["diverged_pct"]
    r["phys_div_incl_oom_pct"] = 100.0 * (c["n_diverged"] + c["n_oom"]) / n
    r["finished_pct"] = r["survived_pct"]
    # Finished runs that nevertheless recorded a temperature blow-up on the way;
    # the arena metric (and every rate above) counts them as survived.
    r["n_survived_with_temp_blowup"] = int(((per["outcome"] == "survived")
                                            & per["temp_blowup"].astype(bool)).sum())
    comp = per["completion"].to_numpy(float)
    r["mean_completion_pct"] = 100.0 * comp.mean()
    # Same quantity with finished runs set to exactly 1 (the Fig. 3a legend).
    r["mean_completion_snapped_pct"] = 100.0 * np.where(comp >= SURVIVED, 1.0, comp).mean()
    ended = per["outcome"].isin(["diverged", "oom", "other", "truncated"])
    for tag, mask in (("first_end", ended),
                      ("first_div", per["outcome"] == "diverged"),
                      ("first_div_or_oom", per["outcome"].isin(["diverged", "oom"]))):
        frac = float(per.loc[mask, "nfs"].min()) if mask.any() else np.nan
        for k, v in at_time(frac, args).items():
            r[f"{tag}_{k}"] = v
    return r


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--groups", nargs="+")
    ap.add_argument("--n-npt", type=int, default=80)
    ap.add_argument("--total-time-ps", type=float, default=10.0)
    ap.add_argument("--t-start", type=float, default=300.0)
    ap.add_argument("--t-end", type=float, default=3000.0)
    ap.add_argument("--p-end-gpa", type=float, default=500.0)
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

    runs = []
    for run_dir, label, group in zip(args.runs, args.labels, args.groups):
        per = load_run(run_dir, args.n_npt)
        per.insert(0, "label", label)
        per.insert(0, "group", group)
        runs.append(per)
    allruns = pd.concat(runs, ignore_index=True)
    # End of the run on the protocol ramps; empty for runs that finished.
    frac = allruns["nfs"].where(allruns["outcome"] != "survived").to_numpy(float)
    for k, v in at_time(frac, args).items():
        allruns[f"end_{k}"] = v
    allruns.to_csv(args.out_dir / "stability_npt_runs.csv", index=False)

    per_seed = pd.DataFrame([{"group": g, "label": l, "run_dir": str(d), **rates(p, args)}
                             for (g, l, d, p) in zip(args.groups, args.labels, args.runs, runs)])
    per_seed.to_csv(args.out_dir / "stability_npt_per_seed.csv", index=False)

    seed_cols = ["phys_div_pct", "phys_div_incl_oom_pct", "oom_pct", "truncated_pct",
                 "finished_pct", "mean_completion_pct", "mean_completion_snapped_pct",
                 "first_end_ps"]
    rows = []
    for group in dict.fromkeys(args.groups):
        sub = allruns[allruns["group"] == group]
        r = {"group": group, "n_seeds": int(sub["label"].nunique()), **rates(sub, args)}
        s = per_seed[per_seed["group"] == group]
        for c in seed_cols:
            v = s[c].to_numpy(float)
            r[f"{c}_seed_mean"] = v.mean()
            r[f"{c}_seed_sd"] = v.std(ddof=1) if len(v) > 1 else np.nan
            r[f"{c}_seed_min"] = v.min()
            r[f"{c}_seed_max"] = v.max()
        rows.append(r)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "stability_npt_summary.csv", index=False)

    # ---- printout -----------------------------------------------------------
    pd.set_option("display.width", 200)
    print(f"\nMLIP-Arena NPT stability ({args.n_npt} protocol runs per benchmark run)")
    hdr = (f"{'model / run':<24}{'runs':>5}{'surv':>6}{'div':>5}{'oom':>5}{'oth':>5}"
           f"{'trunc':>6}{'miss':>5}{'compl%':>8}{'fin%':>7}{'div%':>7}{'div+oom%':>9}"
           f"   first end of a run")
    print(hdr)

    def line(name, r):
        fe = (f"{r['first_end_ps']:.2f} ps ({r['first_end_K']:.0f} K, "
              f"{r['first_end_GPa']:.0f} GPa)" if np.isfinite(r["first_end_ps"]) else "-")
        return (f"{name[:23]:<24}{r['n_runs']:>5}{r['n_survived']:>6}{r['n_diverged']:>5}"
                f"{r['n_oom']:>5}{r['n_other']:>5}{r['n_truncated']:>6}{r['n_missing']:>5}"
                f"{r['mean_completion_pct']:>8.1f}{r['finished_pct']:>7.1f}"
                f"{r['phys_div_pct']:>7.1f}{r['phys_div_incl_oom_pct']:>9.1f}   {fe}")

    for _, r in summary.iterrows():
        print(line(r["group"] + (" (pooled)" if r["n_seeds"] > 1 else ""), r))
        if r["n_seeds"] > 1:
            for _, s in per_seed[per_seed["group"] == r["group"]].iterrows():
                print(line("  " + s["label"], s))
            print(f"{'  seed mean +- SD':<24}" + "  ".join(
                f"{c.replace('_pct', '')}={r[c + '_seed_mean']:.1f}+-{r[c + '_seed_sd']:.1f}"
                for c in ("mean_completion_pct", "finished_pct", "phys_div_pct",
                          "phys_div_incl_oom_pct", "truncated_pct")))
            print(f"{'  first end, seed range':<24}{r['first_end_ps_seed_min']:.2f}"
                  f"-{r['first_end_ps_seed_max']:.2f} ps")
    print("\ncompl% = mean completed fraction of the trajectory over all protocol runs "
          "(missing = 0); fin% = survived in full;\ndiv% = physically diverged "
          "(cell collapse or temperature blow-up), OOM excluded; div+oom% includes "
          "CUDA OOM.")
    print("finished runs with a recorded temperature above 1e5 K (counted as survived): "
          + ", ".join(f"{r['group']} {r['n_survived_with_temp_blowup']}"
                      for _, r in summary.iterrows()))
    print(f"wrote {args.out_dir}/stability_npt_{{runs,per_seed,summary}}.csv")


if __name__ == "__main__":
    main()
