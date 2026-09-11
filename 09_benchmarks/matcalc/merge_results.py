#!/usr/bin/env python3
"""
Concatenate the per-rank CSVs written by run_matcalc_bench.py into one CSV per
benchmark, and print summary error metrics for every predicted column vs its
*_DFT ground-truth sibling. Run on the login node after the job finishes.

    python merge_results.py --outdir results --model-name mace-omat-custom

Metrics reported per property (in addition to the original MAE):
  * MedAE  -- median absolute error, robust to the outlier tail (e.g. the boron
    reference artifact that inflates the Eform MAE ~20x over its median);
  * RMSE   -- root-mean-square error.
For the three near-equilibrium properties the MatPES paper (Fig. 4) reports as
PERCENTAGE errors -- bulk modulus K_VRH, shear modulus G_VRH, heat capacity CV --
we additionally print the percentage-error distribution so results are directly
comparable to the paper:
  * MedAPE = median |pred-DFT|/|DFT| (the paper's robust headline number),
  * MAPE   = mean of the same (dominated by the tail; report but don't lead with it),
  * signed-% median and [p25,p75] box + p90, characterising the Fig. 4 violin, and
  * a count of UNPHYSICAL negative predictions (negative modulus/CV).
The softening / off-equilibrium-force |Fi| benchmark (Fig. 4d) has no *_DFT column
(the softening scale IS the metric); it gets a dedicated summary: median/mean scale
and the implied median force underestimation (1-scale)*100%.
NOTE the paper's convention: quote the MEDIAN percentage error, not the mean/MAPE.

Two safeguards mirror the run script:
  * failed structures (every numeric prediction is NaN, e.g. a diverged
    relaxation) are counted, reported, and EXCLUDED from the metrics rather than
    dragging a whole shard down; and
  * any row containing an element outside ALLOWED_Z is dropped before scoring as
    a backstop -- this is the only place the element cut reaches Softening, which
    is indexed by material_id and cannot be filtered up front in the run script.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

BENCHES = ["Equilibrium", "Elasticity", "Phonon", "Softening"]

# Must match ALLOWED_Z in run_matcalc_bench.py: 1-56 (H..Ba) and 72-83 (Hf..Bi).
# Excludes lanthanides (57-71) and everything from Po (84) up.
ALLOWED_Z = frozenset(range(1, 57)) | frozenset(range(72, 84))

# Property BASE names (post-rename column stem, before the _DFT / _<model> suffix)
# that the MatPES paper Fig. 4 reports as PERCENTAGE errors. Raw pre-rename stems
# are included too so salvaged shards (which keep the raw keys) are still matched.
PCT_PROPS = frozenset({
    "K_vrh", "G_vrh", "CV",                                  # renamed
    "bulk_modulus_vrh", "shear_modulus_vrh", "heat_capacity",  # raw (salvage path)
})
# Physically positive-definite quantities: a negative prediction is unphysical
POSITIVE_PROPS = PCT_PROPS
# Ground-truth values with |DFT| below this are excluded from percentage metrics
_PCT_EPS = 1e-9


def _fmt_metrics(err: pd.Series) -> str:
    """Absolute-error metrics shared by every property."""
    return (f"MAE={err.mean():.4g}  MedAE={err.median():.4g}  "
            f"RMSE={np.sqrt((err ** 2).mean()):.4g}")


def _pct_line(gt: pd.Series, pred: pd.Series) -> str | None:
    """Percentage-error summary matching the paper's Fig. 4 convention, or None
    if there is nothing to score. Sign of the error is kept (via pred-gt); the
    magnitude uses |DFT| so it is well defined even for a stray negative DFT."""
    ok = (gt.abs() > _PCT_EPS)
    if not ok.any():
        return None
    g, p = gt[ok], pred[ok]
    signed = 100.0 * (p - g) / g.abs()
    a = signed.abs()
    return (f"%err: MedAPE={a.median():.3g}%  MAPE={a.mean():.3g}%  "
            f"signed-median={signed.median():+.3g}%  "
            f"[p25,p75]=[{signed.quantile(.25):+.3g},{signed.quantile(.75):+.3g}]%  "
            f"p90|%|={a.quantile(.90):.3g}%  n={len(g)}")


def _disallowed_mask(df: pd.DataFrame) -> pd.Series | None:
    """Boolean mask of rows whose `formula` contains a disallowed element.
    Returns None if there is no formula column to inspect (cannot filter)."""
    col = next((c for c in ("formula", "composition") if c in df.columns), None)
    if col is None:
        return None
    from pymatgen.core import Composition

    def bad(formula) -> bool:
        try:
            return any(el.Z not in ALLOWED_Z for el in Composition(str(formula)).elements)
        except Exception:
            return False  # unparseable -> keep, don't silently drop on a parse error

    return df[col].apply(bad)


def _neg(v: float) -> bool:
    return v < 0


def _nonpos(v: float) -> bool:
    return v <= 0


# Physical-validity rules keyed by predicted-property BASE name (post-rename stem,
# raw stems included for salvaged shards). A prediction that violates its rule is
# UNPHYSICAL, not merely inaccurate: a mechanically stable crystal cannot have a
# negative bulk/shear modulus, heat capacity is non-negative, and a softening scale
# <= 0 means the predicted forces anti-correlate with DFT. Eform (legitimately
# negative) and the fingerprint distance have no such bound and are never flagged.
PATHOLOGY_RULES = {
    "K_vrh": _neg, "bulk_modulus_vrh": _neg,
    "G_vrh": _neg, "shear_modulus_vrh": _neg,
    "CV": _neg, "heat_capacity": _neg,
    "softening_scale": _nonpos,
}


def failed_mask(df: pd.DataFrame, pred_cols: list[str]) -> pd.Series:
    """Rows where every numeric prediction is NaN (a failed equilibration/eval)."""
    if not pred_cols:
        return pd.Series(False, index=df.index)
    num = df[pred_cols].apply(pd.to_numeric, errors="coerce")
    return num.isna().all(axis=1)


def unphysical_mask(df: pd.DataFrame, model_name: str) -> tuple[pd.Series, dict]:
    """Mask of rows with >=1 unphysical prediction, plus {row_idx: [reasons]}."""
    mask = pd.Series(False, index=df.index)
    reasons: dict = {}
    for base, rule in PATHOLOGY_RULES.items():
        col = f"{base}_{model_name}"
        if col not in df.columns:
            continue
        v = pd.to_numeric(df[col], errors="coerce")
        bad = v.notna() & v.map(rule)
        for i in df.index[bad]:
            reasons.setdefault(i, []).append(f"{base}={v[i]:.4g}")
        mask |= bad
    return mask, reasons


def list_compositions(df: pd.DataFrame, idx, reasons: dict | None = None,
                      cap: int = 80) -> None:
    """Print the composition (+ id and any offending values) of each flagged row."""
    idcol = next((c for c in ("material_id", "mp_id", "wbm_id") if c in df.columns), None)
    rows = list(idx)
    for i in rows[:cap]:
        r = df.loc[i]
        formula = r["formula"] if "formula" in df.columns else "?"
        tag = f" ({r[idcol]})" if idcol else ""
        why = f"  [{', '.join(reasons[i])}]" if reasons and i in reasons else ""
        print(f"        {formula}{tag}{why}")
    if len(rows) > cap:
        print(f"        (+{len(rows) - cap} more)")


def summarize(df: pd.DataFrame) -> list[str]:
    """For each `X_DFT` column, find the matching prediction column and report
    absolute-error metrics (MAE/MedAE/RMSE) plus, for the properties the paper
    reports as percentages (K_VRH/G_VRH/CV), the percentage-error distribution.
    Also reports the standalone fingerprint distance and the softening scale.
    Returns the list of prediction columns it scored, so the caller can count
    failed rows consistently with what was actually summarized."""
    scored_pred_cols = []
    for gt_col in [c for c in df.columns if c.endswith("_DFT")]:
        base = gt_col[: -len("_DFT")]
        cands = [c for c in df.columns
                 if c != gt_col and (c == base or c.startswith(base + "_"))]
        for pred_col in cands:
            scored_pred_cols.append(pred_col)
            pair = df[[gt_col, pred_col]].apply(pd.to_numeric, errors="coerce").dropna()
            if pair.empty:
                continue
            gt, pred = pair[gt_col], pair[pred_col]
            err = (pred - gt).abs()
            print(f"    {pred_col:<32s} vs {gt_col:<28s} {_fmt_metrics(err)}  n={len(pair)}")
            if base in PCT_PROPS:
                line = _pct_line(gt, pred)
                if line is not None:
                    print(f"        {line}")
            if base in POSITIVE_PROPS:
                n_neg = int((pred < 0).sum())
                if n_neg:
                    print(f"        unphysical (negative) predictions: {n_neg} "
                          f"({100 * n_neg / len(pred):.1f}%)")

    # Equilibrium's structure-match metric is a standalone `d_<model>` distance
    # (no `_DFT` sibling) -- report its mean and median over structures that scored.
    for d_col in [c for c in df.columns if c.startswith("d_")]:
        vals = pd.to_numeric(df[d_col], errors="coerce").dropna()
        if not vals.empty:
            print(f"    {d_col:<32s} {'(fingerprint dist)':<31s} "
                  f"mean={vals.mean():.4g}  median={vals.median():.4g}  n={len(vals)}")

    # Softening / off-equilibrium force |Fi| (Fig. 4d): no _DFT sibling -- the
    # softening scale itself is the metric (ideal 1.0; <1 = forces underestimated).
    for s_col in [c for c in df.columns if c.startswith("softening_scale")]:
        scored_pred_cols.append(s_col)
        vals = pd.to_numeric(df[s_col], errors="coerce").dropna()
        if not vals.empty:
            med = vals.median()
            print(f"    {s_col:<32s} {'(softening scale, ideal=1)':<31s} "
                  f"median={med:.4g}  mean={vals.mean():.4g}  "
                  f"[p25,p75]=[{vals.quantile(.25):.4g},{vals.quantile(.75):.4g}]  n={len(vals)}")
            print(f"        |Fi| median underestimation = {100 * (1 - med):+.3g}% "
                  f"(paper Fig. 4d convention)")
    return scored_pred_cols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--model-name", default="mace-omat-custom")
    args = ap.parse_args()
    outdir = Path(args.outdir)

    for name in BENCHES:
        parts = sorted(outdir.glob(f"{name}_{args.model_name}_rank*of*.csv"))
        if not parts:
            continue
        # Sanity: confirm all shards of the same world size are present.
        worlds = {int(re.search(r"of(\d+)\.csv$", p.name).group(1)) for p in parts}
        exp = worlds.pop() if len(worlds) == 1 else "?"
        merged = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        n_raw = len(merged)

        # Backstop element cut (the only one that reaches Softening).
        mask = _disallowed_mask(merged)
        if mask is None:
            elem_note = "no formula column -> element cut NOT applied here"
        else:
            n_elem = int(mask.sum())
            merged = merged[~mask].reset_index(drop=True)
            elem_note = f"{n_elem} rows dropped for disallowed elements"

        final = outdir / f"{name}_{args.model_name}_FULL.csv"
        merged.to_csv(final, index=False)

        print(f"\n=== {name}: {len(parts)} shards (world={exp}) -> {n_raw} rows -> "
              f"{final.name} ===")
        scored_pred_cols = summarize(merged)

        # Pathological cases: failed evaluations (all-NaN) and unphysical but finite
        # predictions (negative modulus/CV, softening <= 0). Kept disjoint so counts
        # add up (a failed all-NaN row can never also be flagged unphysical).
        failed = failed_mask(merged, scored_pred_cols)
        unphys, reasons = unphysical_mask(merged, args.model_name)
        unphys = unphys & ~failed
        n_fail, n_unphys = int(failed.sum()), int(unphys.sum())

        print(f"    [{elem_note}; {n_fail} failed (NaN, excluded from metrics above); "
              f"{n_unphys} unphysical (kept in metrics above); "
              f"{len(merged) - n_fail} of {len(merged)} structures scored]")

        if n_fail:
            print(f"    FAILED equilibrations/evaluations ({n_fail}) -- compositions:")
            list_compositions(merged, merged.index[failed])
        if n_unphys:
            print(f"    UNPHYSICAL predictions ({n_unphys}) -- compositions:")
            list_compositions(merged, merged.index[unphys], reasons)

        # Failed (all-NaN) rows are already excluded from every metric above, so
        # removing them changes nothing -- only the unphysical FINITE predictions
        # only when there is something finite to exclude.
        if n_unphys:
            clean = merged[~(failed | unphys)].reset_index(drop=True)
            print(f"    --- metrics EXCLUDING pathological/unphysical cases "
                  f"(removed {n_unphys} unphysical + {n_fail} failed, n={len(clean)}) ---")
            summarize(clean)
        elif n_fail:
            print("    (only failed/NaN cases were pathological; they are already "
                  "excluded above, so clean metrics are identical)")
        else:
            print("    (no pathological/unphysical cases -- metrics above are already clean)")


if __name__ == "__main__":
    main()
