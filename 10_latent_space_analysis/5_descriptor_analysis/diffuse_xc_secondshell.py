#!/usr/bin/env python3
"""Stage 10.5 -- is the latent spread of O/F environments due to PBE+U? (SI S5.3, part 2)

Backs, in SI S5.3 ("Diffuse descriptors"):
  "atoms of the same descriptor occupy nearly the same region of latent space
   whether their host structure was computed with pure PBE or with PBE+U, and
   the functional accounts for only a small fraction of a descriptor's spread
   compared with the chemistry of the host structure."

Design. OMat24 follows the Materials Project rule: PBE+U is used iff the cell
contains O or F AND at least one of Co, Cr, Fe, Mn, Mo, Ni, V, W. For an
O/F-coordinated descriptor whose central atom and ligands are none of those
eight, whether +U was applied is decided by the REST of the cell, while the
descriptor (element, spin, polyhedron, distortion and bond bins) is fixed.
Within each such descriptor we therefore compare
  (A) atoms in PBE+U cells with atoms in pure-PBE cells, and
  (B) atoms grouped by the host chemistry outside the first shell (cell
      element set minus the central element and the ligand elements),
by eta^2 = between-group / total sum of squares in the PCA-50 latent space,
each against a label-permutation null; (C) compares the two on the
descriptors that admit both tests. Groups need >= --min-group atoms.

Caveat for (C). The +U flag is a function of the cell's element set, hence of
the outer-chemistry label: the +U partition is (up to the dropped small groups)
a coarsening of the outer-chemistry partition, and a finer partition cannot
explain less variance. A raw "eta^2(outer) > eta^2(+U)" is therefore close to
true by construction. The primary comparisons are the ones that remove this:
  * outer chemistry measured among the pure-PBE atoms only, where it cannot
    carry the +U flag, against eta^2(+U);
  * both eta^2 minus their own permutation-null means.
The raw head-to-head is still printed, labelled, because it is the number the
manuscript draft was compared against.

Inputs (repo stage outputs)
  --report-card  stage 10.4 report-card parquet
  --feat-dir     stage 10.3 shards; the per-atom 'element' column of the meta
                 shards gives every cell's composition, so no extxyz is read
  --pca-cache    optional; the same file split_descriptors.py writes

Environment: env B (numpy, scipy, pandas, pyarrow). Memory is dominated by the
per-atom metadata of all shards.
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import descriptor_common as C

# ============================================================================
# >>> USER CONFIGURATION -- pass these on the command line <<<
#
#   --report-card  stage 10.4 output (submit_metrics.sh REPORT_OUT).
#   --feat-dir     stage 10.3 output directory (submit_metrics.sh FEAT_DIR).
#   --pca-cache    optional .npz for the PCA basis (written if absent).
#   --min-atoms    analysis floor; the manuscript uses 30.
#   --out-table    optional parquet, one row per O/F-coordinated descriptor.
# ============================================================================

U_ELEMENTS = frozenset("Co Cr Fe Mn Mo Ni V W".split())   # MP / OMat24 +U elements


def centroid_separation(X, flag):
    """Distance between the two group centroids / pooled RMS radius."""
    a, b = X[~flag], X[flag]
    rad = np.sqrt(0.5 * (((a - a.mean(0)) ** 2).sum(1).mean()
                         + ((b - b.mean(0)) ** 2).sum(1).mean()))
    return np.linalg.norm(b.mean(0) - a.mean(0)) / rad


def grouped_eta2(X, labels, min_group, rng, nperm):
    """eta^2 test on the groups that hold >= min_group atoms (>= 2 such groups)."""
    vc = pd.Series(labels).value_counts()
    keep = np.isin(labels, vc[vc >= min_group].index)
    if (vc >= min_group).sum() < 2 or keep.sum() < 3 * min_group:
        return None
    return C.eta2_perm(X[keep], labels[keep], rng, nperm) + (int((vc >= min_group).sum()),)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report-card", required=True)
    ap.add_argument("--feat-dir", required=True)
    ap.add_argument("--pca-cache", default=None)
    ap.add_argument("--min-atoms", type=int, default=C.MIN_ATOMS)
    ap.add_argument("--min-group", type=int, default=10)
    ap.add_argument("--nperm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-table", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    rc = C.load_report_card(a.report_card, a.min_atoms)
    T = rc[rc.has_OF].set_index("chemenv_id")
    C.log(f"{len(T):,} O/F-coordinated descriptors ({int(T.diffuse.sum())} diffuse)")

    meta = C.read_meta(a.feat_dir)
    cells = C.cell_elements(meta)
    sm, X = C.gather_latent(a.feat_dir, meta, T.index, C.pca_basis(a.feat_dir, a.pca_cache))
    del meta
    els = sm.sid.map(cells).str.split("-")
    sm["plus_U"] = els.map(lambda E: bool(U_ELEMENTS & set(E)) and ("O" in E or "F" in E))
    shell = sm.chemenv_id.map(lambda c: set(T.at[c, "nb_species"].split("-"))
                              | {T.at[c, "element"]})
    sm["outer"] = ["-".join(sorted(set(E) - s)) for E, s in zip(els, shell)]
    C.log(f"{len(sm):,} atoms; {100 * sm.plus_U.mean():.1f} % in PBE+U cells")

    rows = []
    for cid, g in sm.groupby("chemenv_id", sort=True):
        ix = g.index.to_numpy()
        Xi, u = X[ix], g.plus_U.to_numpy()
        rec = dict(chemenv_id=cid, n_atoms=len(ix), diffuse=bool(T.at[cid, "diffuse"]),
                   n_plusU=int(u.sum()), n_pbe=int((~u).sum()))
        if rec["n_plusU"] >= a.min_group and rec["n_pbe"] >= a.min_group:
            rec.update(zip(["U_eta2", "U_null", "U_z", "U_p"],
                           C.eta2_perm(Xi, u.astype(int), rng, a.nperm)))
            rec["U_sep"] = centroid_separation(Xi, u)
            # outer chemistry restricted to the pure-PBE atoms: there it cannot
            # carry the +U flag, so it isolates host chemistry from the functional
            r = grouped_eta2(Xi[~u], g.outer.to_numpy()[~u], a.min_group, rng, a.nperm)
            if r:
                rec["S_pbe_eta2"] = r[0]
        r = grouped_eta2(Xi, g.outer.to_numpy(), a.min_group, rng, a.nperm)
        if r:
            rec.update(zip(["S_eta2", "S_null", "S_z", "S_p", "S_levels"], r))
        rows.append(rec)
    R = pd.DataFrame(rows)

    C.hdr("(A) PBE vs PBE+U WITHIN ONE DESCRIPTOR")
    A = R.dropna(subset=["U_eta2"]).copy()
    A["U_q"] = C.bh_fdr(A.U_p)
    centre_u = A.chemenv_id.map(T.element).isin(U_ELEMENTS).any()
    print(f"  descriptors with >= {a.min_group} atoms in each class: {len(A)} "
          f"({int(A.diffuse.sum())} diffuse); any with a +U central element: {centre_u}")
    print(f"  median eta^2 of the +U flag            {A.U_eta2.median():.4f}"
          f"   (permutation null {A.U_null.median():.4f})")
    print(f"  significant at BH-FDR 5 %              {100 * (A.U_q < 0.05).mean():.1f} %")
    print(f"  median centroid distance / RMS radius  {A.U_sep.median():.3f}")

    C.hdr("(B) HOST CHEMISTRY BEYOND THE FIRST SHELL")
    B = R.dropna(subset=["S_eta2"]).copy()
    B["S_q"] = C.bh_fdr(B.S_p)
    print(f"  descriptors with >= 2 outer chemistries of >= {a.min_group} atoms: {len(B)} "
          f"({int(B.diffuse.sum())} diffuse)")
    print(f"  median eta^2 of outer chemistry        {B.S_eta2.median():.4f}"
          f"   (permutation null {B.S_null.median():.4f})")
    print(f"  significant at BH-FDR 5 %              {100 * (B.S_q < 0.05).mean():.1f} %")

    C.hdr("(C) HEAD TO HEAD ON THE SAME DESCRIPTORS")
    AB = R.dropna(subset=["U_eta2", "S_eta2"])
    print(f"  {len(AB)} descriptors admit both tests")
    P = AB.dropna(subset=["S_pbe_eta2"])
    print(f"  PRIMARY 1 -- outer chemistry among pure-PBE atoms only (cannot encode +U), "
          f"{len(P)} descriptors:")
    print(f"     median eta^2 {P.S_pbe_eta2.median():.4f} vs +U {P.U_eta2.median():.4f}; "
          f"outer chemistry larger in {100 * (P.S_pbe_eta2 > P.U_eta2).mean():.1f} % "
          f"(Wilcoxon p = {wilcoxon(P.S_pbe_eta2 - P.U_eta2).pvalue:.2g})")
    exc = (AB.S_eta2 - AB.S_null) > (AB.U_eta2 - AB.U_null)
    print(f"  PRIMARY 2 -- eta^2 minus its permutation-null mean, {len(AB)} descriptors:")
    print(f"     median excess: outer {(AB.S_eta2 - AB.S_null).median():.4f}, "
          f"+U {(AB.U_eta2 - AB.U_null).median():.4f}; outer larger in "
          f"{100 * exc.mean():.1f} %")
    print(f"  secondary (paper-comparable, but +U is nested in outer chemistry, so "
          f"largely true by construction):")
    print(f"     raw median eta^2 +U {AB.U_eta2.median():.4f}, outer {AB.S_eta2.median():.4f};"
          f" outer larger in {100 * (AB.S_eta2 > AB.U_eta2).mean():.1f} %")

    if a.out_table:
        R.to_parquet(a.out_table, index=False)
        C.log(f"wrote {a.out_table}")


if __name__ == "__main__":
    main()
