#!/usr/bin/env python3
"""
Stage 09 / nebDFT2k -- Table II of the manuscript (migration-barrier errors split
by Hubbard-U chemistry) and the numbers quoted around it, from the files
run_litraj_nebdft2k.py writes.

Environment: envD-mliparena (numpy, pandas, pyarrow). CPU only, no model.

Input, per run: the --outdir of run_litraj_nebdft2k.py (submit_nebdft2k.sh:
$OUTDIR, default <submit dir>/results_litraj_nebdft2k), after its --aggregate
step. Read from it:
  <model_name>-litraj-nebdft2k.parquet   one row per hop that returned a result
                                         (em_pred, em_dft, converged,
                                         has_specific_TM, has_unsupported,
                                         oom_terminated, edge_id)
  <model_name>_<edge_id>.done / .json    a hop with a .done marker but no .json
                                         was attempted and returned nothing
                                         (both attempts hit --per-run-timeout,
                                         or crashed); it is absent from the
                                         parquet and is counted as 'no result'

Scoring, as in the manuscript: a hop is kept if its band converged under the
published gate (max |F| < 0.1 eV/A over band and end-points), it was not
OOM-terminated, it has no out-of-support element, and both barriers are
finite. Every model is scored on its own kept hops. Per Hubbard-U split
(has_specific_TM: host contains Co, Cr, Fe, Mn, Mo, Ni, V or W):
  MAE, RMSE, median |err|        Table II columns (eV)
  median / mean signed error     err = E_m(pred) - E_m(DFT)
  n kept, n not converged, n with no result, excluded %
Random-seed replicas (labels starting with rdm / seed / random, pooled per
--groups entry, default 'random') are reported per seed and as mean +- sample
SD over seeds (the Table II row); the pooled-hop values are given as well.
Bold rule of Table II: the smallest value in each column per split, the
replica group entering with its seed mean; compared at the printed precision
(3 decimals), ties all bold.

Backs (main text, Table II and 'Li+ migration'; SI S4.4, Fig. S6):
  Table II; "621 hops" / "1035 hops"; "EnvTok achieves the lowest MAE and RMSE,
  respectively 15% and 49% below MACE-OMAT-0"; "MACE-OMAT-0 retains the lowest
  median absolute error of 38 meV"; "seed-to-seed standard deviations
  exceeding 25% and 90% of their respective means"; "every model exhibits a
  systematic shift in median signed error of 59--76 meV" (main text and
  Fig. S6 caption); "MACE-OMAT-0, EnvTok, and DIRECT differ by at most 8 meV in
  MAE and 4 meV in median signed error" (the 4 meV is the spread of the median
  ABSOLUTE error; both spreads are printed); "two hops that exceeded the
  wall-clock limit of 2 h (one each for the EnvTok model and one of the random
  models) are omitted".

    python table_nebdft2k.py \\
        --runs omat/results_litraj_nebdft2k v14/results_litraj_nebdft2k \\
               direct/results_litraj_nebdft2k rdm1/results_litraj_nebdft2k \\
               rdm2/results_litraj_nebdft2k rdm3/results_litraj_nebdft2k \\
               rdm4/results_litraj_nebdft2k \\
        --labels MACE-OMAT-0 EnvTok DIRECT rdm1 rdm2 rdm3 rdm4 \\
        --reference MACE-OMAT-0 --out-dir nebdft2k_table
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
#                run_litraj_nebdft2k.py, holding
#                <model_name>-litraj-nebdft2k.parquet (from --aggregate) and the
#                per-hop .done / .json files.
#
#   --labels     One label per --runs entry, same order. A label starting with
#                rdm / seed / random marks a random-seed replica. Default: the
#                parent directory names.
#
#   --groups     Optional. One model name per --runs entry, same order, for
#                pooling runs into one model explicitly. Default: replica
#                labels go to the group 'random', every other run is its own
#                group (its label).
#
#   --reference  Optional. Group name of the full-data reference model; the
#                printout then gives every model's MAE / RMSE relative to it.
#
#   --out-dir    Where the outputs are written (created if missing):
#                nebdft2k_per_run.csv     one row per run and split
#                nebdft2k_table.csv       one row per model and split (Table II)
#                nebdft2k_shift.csv       median signed error per split and the
#                                         shift between the splits
#                nebdft2k_table.tex       Table II body (tabular)
# ============================================================================

SEED_PREFIXES = ("rdm", "seed", "random")
SPLITS = [(False, "w/o Hubbard-U"), (True, "with Hubbard-U")]
COLS = ["mae", "rmse", "med_abs"]          # Table II columns, eV
PREC = 3                                    # printed decimals (bold rule)


def is_replica(label: str) -> bool:
    return label.lower().startswith(SEED_PREFIXES)


def load_run(run_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """The parquet's rows plus the edge_ids attempted without a result."""
    hits = sorted(run_dir.glob("*-litraj-nebdft2k.parquet"))
    if len(hits) != 1:
        sys.exit(f"ERROR: expected exactly one *-litraj-nebdft2k.parquet in {run_dir}, "
                 f"found {[h.name for h in hits]}. Run run_litraj_nebdft2k.py "
                 f"--aggregate first.")
    model_name = hits[0].name[:-len("-litraj-nebdft2k.parquet")]
    df = pd.read_parquet(hits[0]).drop(columns=["profile_pred"], errors="ignore")
    if df["edge_id"].duplicated().any():
        sys.exit(f"ERROR: {hits[0]}: duplicated edge_id "
                 f"{sorted(df.edge_id[df.edge_id.duplicated()])[:5]}")
    pre = f"{model_name}_"
    done = {p.stem[len(pre):] for p in run_dir.glob(f"{pre}*.done")}
    json = {p.stem[len(pre):] for p in run_dir.glob(f"{pre}*.json")}
    if json and json != set(df["edge_id"]):
        print(f"  WARNING {run_dir}: the .json results ({len(json)}) and the parquet "
              f"rows ({len(df)}) differ; re-run --aggregate.")
    no_result = sorted(done - set(df["edge_id"]))
    oom = (df["oom_terminated"].astype(bool) if "oom_terminated" in df
           else pd.Series(False, index=df.index))
    df["kept"] = (df["converged"].astype(bool) & ~oom
                  & ~df["has_unsupported"].astype(bool)
                  & np.isfinite(df["em_pred"]) & np.isfinite(df["em_dft"]))
    df["err"] = df["em_pred"] - df["em_dft"]
    return df, no_result


def metrics(err: np.ndarray) -> dict:
    if err.size == 0:                        # e.g. a split dropped with --tm-policy drop
        return {"mae": np.nan, "rmse": np.nan, "med_abs": np.nan,
                "median_signed": np.nan, "mean_signed": np.nan, "n_kept": 0}
    a = np.abs(err)
    return {"mae": a.mean(), "rmse": np.sqrt((err ** 2).mean()), "med_abs": np.median(a),
            "median_signed": np.median(err), "mean_signed": err.mean(), "n_kept": err.size}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--groups", nargs="+")
    ap.add_argument("--reference")
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
    if args.reference is not None and args.reference not in args.groups:
        ap.error(f"--reference {args.reference!r} is not one of the groups {args.groups}")
    return args


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    loaded = [load_run(d) for d in args.runs]

    # Split of every hop, from all parquets (a hop with no result in one run is
    # looked up in the others).
    split_of = {}
    for df, _ in loaded:
        split_of.update(dict(zip(df["edge_id"], df["has_specific_TM"].astype(bool))))
    for (df, nores), lab in zip(loaded, args.labels):
        unknown = [e for e in nores if e not in split_of]
        if unknown:
            print(f"  WARNING {lab}: split unknown for hop(s) with no result: {unknown}")

    per_run, err_by = [], {}
    for (df, nores), lab, grp in zip(loaded, args.labels, args.groups):
        for with_u, sname in SPLITS:
            sub = df[df["has_specific_TM"].astype(bool) == with_u]
            n_nores = sum(1 for e in nores if e in split_of and bool(split_of[e]) == with_u)
            err = sub.loc[sub["kept"], "err"].to_numpy(float)
            err_by[(grp, lab, with_u)] = err
            per_run.append({
                "group": grp, "label": lab, "split": sname,
                "n_rows": len(sub), "n_no_result": n_nores,
                "n_not_converged": int((~sub["converged"].astype(bool)).sum()),
                "n_oom": int(sub["oom_terminated"].astype(bool).sum())
                if "oom_terminated" in sub else 0,
                **metrics(err),
                "excluded_pct": (100.0 * (1 - err.size / len(sub))
                                 if len(sub) else np.nan),
                "excluded_incl_no_result_pct":
                    (100.0 * (1 - err.size / (len(sub) + n_nores))
                     if len(sub) + n_nores else np.nan),
                "no_result_edges": " ".join(e for e in nores if e in split_of
                                            and bool(split_of[e]) == with_u),
            })
    per_run = pd.DataFrame(per_run)
    per_run.to_csv(args.out_dir / "nebdft2k_per_run.csv", index=False)

    # One Table II row per model and split.
    table = []
    for grp in dict.fromkeys(args.groups):
        labs = [l for g, l in zip(args.groups, args.labels) if g == grp]
        for with_u, sname in SPLITS:
            pr = per_run[(per_run.group == grp) & (per_run.split == sname)]
            row = {"group": grp, "split": sname, "n_seeds": len(labs),
                   "n_rows": int(pr.n_rows.sum()), "n_no_result": int(pr.n_no_result.sum()),
                   "n_kept": int(pr.n_kept.sum())}
            for c in COLS + ["median_signed", "mean_signed"]:
                v = pr[c].to_numpy(float)
                row[c] = v.mean()                       # = the run's value if one seed
                row[f"{c}_sd"] = v.std(ddof=1) if len(v) > 1 else np.nan
            pooled = metrics(np.concatenate([err_by[(grp, l, with_u)] for l in labs]))
            for c in COLS + ["median_signed", "mean_signed"]:
                row[f"{c}_pooled"] = pooled[c]
            table.append(row)
    table = pd.DataFrame(table)
    for _, sname in SPLITS:
        m = table.split == sname
        for c in COLS:
            r = table.loc[m, c].round(PREC)
            table.loc[m, f"{c}_bold"] = r == r.min()
    table.to_csv(args.out_dir / "nebdft2k_table.csv", index=False)

    # Shift of the median signed error between the splits.
    shift = []
    for grp in dict.fromkeys(args.groups):
        labs = [l for g, l in zip(args.groups, args.labels) if g == grp]
        per_seed = [np.median(err_by[(grp, l, True)]) - np.median(err_by[(grp, l, False)])
                    for l in labs]
        t = table[table.group == grp].set_index("split")
        shift.append({
            "group": grp,
            "median_signed_wo_U": t.loc["w/o Hubbard-U", "median_signed"],
            "median_signed_with_U": t.loc["with Hubbard-U", "median_signed"],
            "shift_seed_mean": float(np.mean(per_seed)),
            "shift_seed_sd": float(np.std(per_seed, ddof=1)) if len(labs) > 1 else np.nan,
            "shift_seed_min": float(np.min(per_seed)), "shift_seed_max": float(np.max(per_seed)),
            "median_signed_wo_U_pooled": t.loc["w/o Hubbard-U", "median_signed_pooled"],
            "median_signed_with_U_pooled": t.loc["with Hubbard-U", "median_signed_pooled"],
            "shift_pooled": (t.loc["with Hubbard-U", "median_signed_pooled"]
                             - t.loc["w/o Hubbard-U", "median_signed_pooled"]),
        })
    shift = pd.DataFrame(shift)
    shift.to_csv(args.out_dir / "nebdft2k_shift.csv", index=False)

    # Hops per split: every hop that occurs in any run.
    n_hops = {w: sum(1 for v in split_of.values() if bool(v) == w) for w, _ in SPLITS}

    # ---- LaTeX body of Table II ---------------------------------------------
    def cell(row, c):
        s = f"{row[c]:.{PREC}f}"
        if row["n_seeds"] > 1:
            s += rf"\,$\pm$\,{row[c + '_sd']:.{PREC}f}"
        return rf"\textbf{{{s}}}" if row[f"{c}_bold"] else s

    tex = [r"\begin{tabular}{lccc}", r"\toprule",
           r"Model & MAE & RMSE & med.\ $|$err$|$ \\", r"\midrule"]
    for i, (with_u, sname) in enumerate(SPLITS):
        if i:
            tex.append(r"\midrule")
        title = "w/o" if not with_u else "with"
        tex.append(rf"\multicolumn{{4}}{{l}}{{\textit{{{title} Hubbard-$U$ "
                   rf"({n_hops[with_u]} hops)}} }}\\")
        for _, row in table[table.split == sname].iterrows():
            tex.append(f"{row['group']} & " + " & ".join(cell(row, c) for c in COLS) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (args.out_dir / "nebdft2k_table.tex").write_text("\n".join(tex) + "\n")

    # ---- printout -----------------------------------------------------------
    print("\nnebDFT2k migration-barrier errors (eV), each model on its own kept hops")
    for with_u, sname in SPLITS:
        print(f"\n--- {sname}: {n_hops[with_u]} hops ---")
        print(f"{'model / run':<22}{'rows':>6}{'kept':>6}{'n.c.':>6}{'no res':>7}"
              f"{'excl%':>7}{'MAE':>8}{'RMSE':>8}{'med|e|':>8}{'med sgn':>9}{'mean sgn':>9}")
        for _, t in table[table.split == sname].iterrows():
            pr = per_run[(per_run.group == t.group) & (per_run.split == sname)]
            def fmt(r, name):
                return (f"{name[:21]:<22}{r['n_rows']:>6}{r['n_kept']:>6}"
                        f"{r.get('n_not_converged', pr.n_not_converged.sum()):>6}"
                        f"{r['n_no_result']:>7}{100 * (1 - r['n_kept'] / r['n_rows']):>7.1f}"
                        f"{r['mae']:>8.3f}{r['rmse']:>8.3f}{r['med_abs']:>8.3f}"
                        f"{r['median_signed']:>+9.3f}{r['mean_signed']:>+9.3f}")
            if t.n_seeds == 1:
                print(fmt(pr.iloc[0], t.group))
                continue
            for _, r in pr.iterrows():
                print(fmt(r, "  " + r["label"]))
            print(f"{t.group + ' mean+-SD':<22}{'':>32}" + "".join(
                f"{t[c]:>8.3f}" for c in COLS) + f"{t['median_signed']:>+9.3f}"
                  f"{t['mean_signed']:>+9.3f}")
            print(f"{'':<22}{'SD':>32}" + "".join(f"{t[c + '_sd']:>8.3f}" for c in COLS)
                  + "   SD/mean: " + ", ".join(f"{c} {100 * t[c + '_sd'] / t[c]:.0f}%"
                                               for c in COLS))
            print(f"{t.group + ' pooled hops':<22}{'':>32}" + "".join(
                f"{t[c + '_pooled']:>8.3f}" for c in COLS)
                  + f"{t['median_signed_pooled']:>+9.3f}{t['mean_signed_pooled']:>+9.3f}")
        single = table[(table.split == sname) & (table.n_seeds == 1)]
        if len(single) > 1:
            print(f"spread over single-run models: MAE {1000 * np.ptp(single.mae):.1f} meV, "
                  f"median |err| {1000 * np.ptp(single.med_abs):.1f} meV, "
                  f"median signed {1000 * np.ptp(single.median_signed):.1f} meV")
        if args.reference:
            ref = table[(table.split == sname) & (table.group == args.reference)].iloc[0]
            print(f"relative to {args.reference}: " + "; ".join(
                f"{t.group} MAE {100 * (t.mae / ref.mae - 1):+.1f}%, "
                f"RMSE {100 * (t.rmse / ref.rmse - 1):+.1f}%"
                for _, t in table[table.split == sname].iterrows()
                if t.group != args.reference))

    print("\nShift of the median signed error, with-U minus w/o-U (meV):")
    for _, s in shift.iterrows():
        extra = (f"  [seeds {1000 * s.shift_seed_min:.1f}..{1000 * s.shift_seed_max:.1f}; "
                 f"pooled hops {1000 * s.shift_pooled:.1f}]"
                 if s.group in table[table.n_seeds > 1].group.values else "")
        print(f"  {s.group:<20}{1000 * s.shift_seed_mean:7.1f}{extra}")
    nores = per_run[per_run.n_no_result > 0]
    print(f"\nHops attempted without a result (.done but no .json): "
          f"{int(per_run.n_no_result.sum())}"
          + "".join(f"\n  {r.label} ({r.split}): {r.no_result_edges}" for _, r in nores.iterrows()))
    print(f"wrote {args.out_dir}/nebdft2k_{{per_run,table,shift}}.csv and nebdft2k_table.tex")


if __name__ == "__main__":
    main()
