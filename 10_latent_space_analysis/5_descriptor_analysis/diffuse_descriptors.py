#!/usr/bin/env python3
"""Stage 10.5 -- which descriptors are diffuse in latent space (SI S5.3, part 1).

Backs the descriptor-level statements of SI S5.3 ("Diffuse descriptors"):
  [A] "3.1 % of the descriptors are diffuse"   (compactness_z_element < 0)
  [B] "descriptors with O or F in the coordination shell constitute 11.9 % of
      the total but account for 67 % of all diffuse ones"
  [C] "The central cation's own electronegativity is not predictive on its own"
      -> univariate ROC AUC of the central element's Pauling electronegativity
  [D] "a typical d-block element is not more often diffuse than other elements"
      -> per-element diffuse rates, d-block vs the rest, Mann-Whitney U
  [E] "Sc, Y, Zr and Hf as well as Ru, Ir and Pt are among the most compact
      environments observed"
      -> per-element median compactness z and diffuse counts
  [F] "After controlling for ligand chemistry, coordination number and
      descriptor abundance, the odds of being diffuse are two to seven times
      higher for V, Cr, Mn, Fe, Cu, Mo and Ag and for ... Tl, Pb, Sn, In and Sb
      than for the other elements, whereas alkali and alkaline-earth centres are
      diffuse an order of magnitude less often"
      -> logistic regression, one central element (or group) against the rest,
         Benjamini-Hochberg across elements
The within-descriptor PBE vs PBE+U test of S5.3 is diffuse_xc_secondshell.py.

Input: --report-card, stage 10.4's report-card parquet. Nothing else.
Element constants (Pauling electronegativity, d-block membership) come from
pymatgen, hence env A (pymatgen, numpy, scipy, pandas, pyarrow). Seconds.
"""

import argparse

import numpy as np
import pandas as pd
from pymatgen.core import Element
from scipy.stats import fisher_exact

import descriptor_common as C

# ============================================================================
# >>> USER CONFIGURATION -- pass these on the command line <<<
#
#   --report-card  stage 10.4 output (submit_metrics.sh REPORT_OUT).
#   --min-atoms    analysis floor; the manuscript uses 30.
#   --out-table    optional parquet with the per-element results of [F].
# ============================================================================

MIN_EL_RATE = 40        # elements with >= this many descriptors enter [D] and [F]
MIN_EL_RANK = 60        # ... and the median-compactness ranking of [E]
LIGANDS = "O F Cl Br I S Se Te N".split()
QUOTED_HIGH = "V Cr Mn Fe Cu Mo Ag Tl Pb Sn In Sb".split()
QUOTED_COMPACT = "Sc Y Zr Hf Ru Ir Pt".split()
ALKALI_AE = "Li Na K Rb Cs Be Mg Ca Sr Ba".split()


def element_table(symbols):
    """Pauling electronegativity (NaN for the noble gases) and d-block flag."""
    rows = {}
    for s in symbols:
        e = Element(s)
        x = e.X
        rows[s] = dict(X=x if np.isfinite(x) else np.nan,
                       dblock=bool(e.is_transition_metal))
    return pd.DataFrame.from_dict(rows, orient="index")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report-card", required=True)
    ap.add_argument("--min-atoms", type=int, default=C.MIN_ATOMS)
    ap.add_argument("--out-table", default=None)
    a = ap.parse_args()
    pd.set_option("display.width", 200)

    rc = C.load_report_card(a.report_card, a.min_atoms)
    N, ND = len(rc), int(rc.diffuse.sum())
    syms = sorted(set(rc.element) | {x for L in rc.ligands for x in L})
    el = element_table(syms)
    rc["cX"] = rc.element.map(el.X)
    rc["dblock"] = rc.element.map(el.dblock).astype(bool)
    rc["lig_X"] = rc.ligands.map(lambda L: np.nanmean([el.X[x] for x in L])
                                 if np.isfinite([el.X[x] for x in L]).any() else np.nan)
    rc["y"] = rc.diffuse.astype(int)

    # ------------------------------------------------------------------ [A]
    C.hdr("[A] DIFFUSE DESCRIPTORS")
    print(f"  {ND} of {N:,} descriptors are diffuse ({100 * ND / N:.2f} %); "
          f"compact {100 * (N - ND) / N:.2f} %")

    # ------------------------------------------------------------------ [B]
    C.hdr("[B] O OR F IN THE COORDINATION SHELL")
    of = rc.has_OF
    print(f"  descriptors with O or F among the ligands: {int(of.sum()):,} "
          f"({100 * of.mean():.1f} % of all)")
    print(f"  diffuse ones among them: {int(rc.y[of].sum())} "
          f"({100 * rc.y[of].sum() / ND:.1f} % of all diffuse)")
    print(f"  diffuse rate: {100 * rc.y[of].mean():.2f} % with O/F vs "
          f"{100 * rc.y[~of].mean():.2f} % without")
    t = pd.DataFrame([dict(ligand=x, descriptors=int(h.sum()), diffuse=int(rc.y[h].sum()),
                           rate_pct=100 * rc.y[h].mean(),
                           share_of_diffuse_pct=100 * rc.y[h].sum() / ND)
                      for x in LIGANDS for h in [rc.ligands.map(lambda L: x in L)]])
    print(t.sort_values("rate_pct", ascending=False).to_string(
        index=False, float_format=lambda x: f"{x:.2f}"))

    # ------------------------------------------------------------------ [C]
    C.hdr("[C] CENTRAL-ATOM ELECTRONEGATIVITY ON ITS OWN")
    d = rc[rc.cX.notna()]
    auc, p = C.mwu_auc(d.cX[d.y == 1], d.cX[d.y == 0])
    print(f"  ROC AUC (diffuse vs compact) of the central Pauling X: {auc:.3f} "
          f"(p = {p:.2g}; {len(d):,} descriptors with a defined X)")
    d2 = d[d.perturb != 999]
    auc2, _ = C.mwu_auc(d2.cX[d2.y == 1], d2.cX[d2.y == 0])
    print(f"  same without the {int((d.perturb == 999).sum())} UNKNOWN-polyhedron "
          f"descriptors (none diffuse): AUC {auc2:.3f}")
    for lab, s in (("O/F-coordinated", d[d.has_OF]), ("other", d[~d.has_OF])):
        print(f"     within {lab:16s}: AUC {C.mwu_auc(s.cX[s.y == 1], s.cX[s.y == 0])[0]:.3f}")
    lx = rc[rc.lig_X.notna()]
    print(f"  for comparison, mean ligand X: AUC "
          f"{C.mwu_auc(lx.lig_X[lx.y == 1], lx.lig_X[lx.y == 0])[0]:.3f}")

    # ------------------------------------------------------------------ [D]
    C.hdr("[D] d-BLOCK CENTRAL ELEMENTS")
    g = rc.groupby("element").agg(n=("y", "size"), diffuse=("y", "sum"),
                                  dblock=("dblock", "first"))
    g["rate"] = g.diffuse / g.n
    gg = g[g.n >= MIN_EL_RATE]
    rd, rn = gg.rate[gg.dblock], gg.rate[~gg.dblock]
    auc, p = C.mwu_auc(rd, rn)
    print(f"  one observation per element (>= {MIN_EL_RATE} descriptors): "
          f"{len(rd)} d-block, {len(rn)} other")
    print(f"  median diffuse rate: d-block {100 * rd.median():.2f} %, "
          f"other {100 * rn.median():.2f} %; Mann-Whitney p = {p:.2f} (AUC {auc:.3f})")
    fe = fisher_exact(pd.crosstab(rc.dblock, rc.y).to_numpy())
    print(f"  for contrast, pooled over descriptors: d-block {100 * rc.y[rc.dblock].mean():.2f} %"
          f" vs {100 * rc.y[~rc.dblock].mean():.2f} % (Fisher OR {fe.statistic:.2f}, "
          f"p = {fe.pvalue:.2g}) -- carried by a few elements:")
    top = g[g.dblock].sort_values("diffuse", ascending=False)
    print("     " + ", ".join(f"{e} {int(r.diffuse)}" for e, r in top.head(5).iterrows())
          + f"  of {int(top.diffuse.sum())} d-block diffuse descriptors")

    # ------------------------------------------------------------------ [E]
    C.hdr("[E] MOST COMPACT CENTRAL ELEMENTS")
    m = rc.groupby("element").agg(n=("y", "size"), diffuse=("y", "sum"),
                                  median_z=("compactness_z_element", "median"))
    m["rate_pct"] = 100 * m.diffuse / m.n
    mr = m[m.n >= MIN_EL_RANK].sort_values("median_z", ascending=False)
    mr["rank"] = np.arange(1, len(mr) + 1)
    print(f"  median compactness z per central element (>= {MIN_EL_RANK} descriptors; "
          f"all-descriptor median {rc.compactness_z_element.median():.2f}), top 10 of {len(mr)}:")
    print(mr.head(10).to_string(float_format=lambda x: f"{x:.2f}"))
    zero = sorted(m[(m.n >= MIN_EL_RATE) & (m.diffuse == 0)].index)
    print(f"\n  elements with >= {MIN_EL_RATE} descriptors and no diffuse one: {', '.join(zero)}")
    print("  the elements named in the manuscript:")
    for e in QUOTED_COMPACT:
        r = m.loc[e]
        rk = int(mr.loc[e, "rank"]) if e in mr.index else None
        print(f"     {e:3s} median z {r.median_z:5.2f}  rank {rk}/{len(mr)}  "
              f"diffuse {int(r.diffuse)}/{int(r.n)} ({r.rate_pct:.2f} %)")

    # ------------------------------------------------------------------ [F]
    C.hdr("[F] CENTRAL ELEMENT AFTER CONTROLS (logistic regression)")
    D = rc.dropna(subset=["lig_X", "cX"]).copy()
    D["has_OF_i"] = D.has_OF.astype(float)
    D["log_n"] = np.log(D.n_atoms)
    ctrl = ["lig_X", "has_OF_i", "cn", "n_nb_species", "log_n"]
    cnt = D.element.value_counts()
    keep = [e for e in cnt[cnt >= MIN_EL_RATE].index if D.y[D.element == e].sum() > 0]
    Dk = D[D.element.isin(keep)]
    print(f"  controls: mean ligand X, O/F in shell, CN, # ligand species, log n_atoms")
    print(f"  elements with >= {MIN_EL_RATE} descriptors and >= 1 diffuse: {len(keep)} "
          f"({len(Dk):,} descriptors); each tested against all the others")
    base = C.logit(Dk.y, Dk[ctrl], ctrl)
    full = C.logit(Dk.y, np.column_stack([Dk[ctrl].to_numpy(float),
                                          pd.get_dummies(Dk.element, drop_first=True,
                                                         dtype=float).to_numpy()]),
                   ctrl + [f"el{i}" for i in range(len(keep) - 1)])
    lr, df, p = C.lr_test(full, base)
    print(f"  central element as a block: LR chi2 = {lr:.1f}, df = {df}, p = {p:.2g}; "
          f"pseudo-R2 {base['prsquared']:.3f} -> {full['prsquared']:.3f}")
    rows = []
    for e in keep:
        is_el = (Dk.element == e).to_numpy(float)
        f = C.logit(Dk.y, np.column_stack([Dk[ctrl].to_numpy(float), is_el]), ctrl + ["is_el"])
        r = f["table"].loc["is_el"]
        n, k = int(is_el.sum()), int(Dk.y[is_el == 1].sum())
        lo, hi = C.wilson(k, n)
        rows.append(dict(element=e, descriptors=n, diffuse=k, crude_pct=100 * k / n,
                         wilson_lo=100 * lo, wilson_hi=100 * hi,
                         adj_OR=r.odds_ratio, OR_lo=np.exp(r.coef - 1.96 * r.se),
                         OR_hi=np.exp(r.coef + 1.96 * r.se), p=r.p))
    E = pd.DataFrame(rows)
    E["q_BH"] = C.bh_fdr(E.p)
    E = E.sort_values("adj_OR", ascending=False)
    sig = E[E.q_BH < 0.05]
    print(f"\n  significant at BH-FDR 5 %: {len(sig)} of {len(E)}")
    print(sig.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    up = sig[sig.adj_OR > 1]
    print(f"\n  elevated odds (q < 0.05, OR > 1): {', '.join(up.element)}")
    print(f"     adjusted OR range {up.adj_OR.min():.2f}-{up.adj_OR.max():.2f}")
    q = E.set_index("element").reindex(QUOTED_HIGH)
    print("  the elements named in the manuscript:")
    for e, r in q.iterrows():
        print(f"     {e:3s} adj OR {r.adj_OR:5.2f}  [{r.OR_lo:.2f}, {r.OR_hi:.2f}]  "
              f"q {r.q_BH:.2g}")
    extra = [e for e in up.element if e not in QUOTED_HIGH]
    print(f"  elevated but not named: {', '.join(extra)}")

    gi = Dk.element.isin(ALKALI_AE).to_numpy(float)
    f = C.logit(Dk.y, np.column_stack([Dk[ctrl].to_numpy(float), gi]), ctrl + ["grp"])
    r = f["table"].loc["grp"]
    print(f"\n  alkali + alkaline-earth centres ({', '.join(ALKALI_AE)}) as one group: "
          f"{int(Dk.y[gi == 1].sum())}/{int(gi.sum())} diffuse "
          f"({100 * Dk.y[gi == 1].mean():.2f} %), adjusted OR {r.odds_ratio:.2f} "
          f"[{np.exp(r.coef - 1.96 * r.se):.2f}, {np.exp(r.coef + 1.96 * r.se):.2f}], "
          f"p = {r.p:.2g}")
    if a.out_table:
        E.to_parquet(a.out_table, index=False)
        C.log(f"wrote {a.out_table}")


if __name__ == "__main__":
    main()
