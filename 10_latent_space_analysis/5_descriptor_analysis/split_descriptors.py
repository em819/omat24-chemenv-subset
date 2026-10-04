#!/usr/bin/env python3
"""Stage 10.5 -- descriptors that split into several latent-space regions (SI S5.2).

Backs the statements of SI S5.2 ("Descriptors with multiple disjoint regions"):
  [A] "The 1.6 % (292) of descriptors that split into two or three regions"
  [B] "... differ weakly at most from those forming a single region for every
      EnvTok descriptor field including the element of the central atom"
      -> chi^2 / Cramer's V of split vs single-region, field by field
  [C] "For most of them the splitting is material-based"
      -> regions recomputed in the stage-10.4 latent space; share of atoms
         whose material puts them in its majority region (material purity),
         with a frame-level permutation null
  [D] "many single-region descriptors occur in at least as many chemically
      distinct materials as those that split"
      -> number of distinct host chemical systems per descriptor
  [E] "compared with single-region descriptors of the same size, split
      descriptors appear about as often in the training set, but in fewer
      different materials. Descriptors that never appear in the training set
      are the least likely to split."
      -> size-matched Mann-Whitney comparison and Fisher's exact test on the
         training coverage written by extract_train_coverage.py

Inputs (all written by the repo's own stages)
  --report-card     stage 10.4 report-card parquet
  --feat-dir        stage 10.3 directory with features_shard*.npy / meta_shard*.parquet
  --train-coverage  extract_train_coverage.py output parquet
Definitions: analysed = >= --min-atoms atoms (30); split = n_components >= 2.

Environment: env B (numpy, scipy, pandas, pyarrow). Memory is dominated by the
per-atom metadata of all shards; run time by one streaming pass over the feature
shards for the PCA basis (cache it with --pca-cache; diffuse_xc_secondshell.py
can reuse the file).
"""

import argparse
import sys

import numpy as np
import pandas as pd

import descriptor_common as C

# ============================================================================
# >>> USER CONFIGURATION -- pass these on the command line <<<
#
#   --report-card     stage 10.4 output (submit_metrics.sh REPORT_OUT).
#   --feat-dir        stage 10.3 output directory (submit_metrics.sh FEAT_DIR).
#   --train-coverage  extract_train_coverage.py --out.
#   --pca-cache       optional .npz for the PCA basis (written if absent).
#   --min-atoms       analysis floor; the manuscript uses 30.
#   --out-table       optional parquet with one row per split descriptor.
# ============================================================================

FIELDS = [  # token fields first, then two shell summaries derived from field 3
    ("element", "central element"), ("spin", "spin"),
    ("coord_elements", "ligand list"), ("ce", "polyhedron symbol"),
    ("cn", "coordination number"), ("perturb", "distortion (CSM) bin"),
    ("dmin", "bond-compression bin"), ("A", "asymmetry bin"),
    ("nb_species", "ligand element set"), ("n_nb_species", "# ligand species"),
]
MIN_LEVEL = 100          # levels with fewer descriptors are not tested (as in the
                         # exploratory analysis); keeps expected cell counts >~ 1.6
N_CONTROLS, TOL = 3, 0.15
SIZE_BANDS = [30, 45, 70, 120, 250, 10**9]
COV_COLS = ["n_frames", "n_materials", "n_formulas", "n_spg", "n_provenance",
            "n_atom_occurrences"]


def size_matched_controls(rc, rng=None):
    """N_CONTROLS single-region controls per split descriptor.

    Nearest n_atoms within +/-TOL, drawn without replacement, split descriptors
    processed in ascending n_atoms. Ties are broken by report-card row order,
    or at random if rng is given (sensitivity check).
    """
    split = rc[rc.split].sort_values("n_atoms", kind="stable")
    pool = rc.loc[~rc.split, ["chemenv_id", "n_atoms"]]
    if rng is not None:
        pool = pool.iloc[rng.permutation(len(pool))]
        split = split.iloc[rng.permutation(len(split))].sort_values("n_atoms", kind="stable")
    pool = pool.assign(order=np.arange(len(pool)))
    taken, rows = set(), []
    for r in split.itertuples():
        c = pool[(pool.n_atoms >= r.n_atoms * (1 - TOL)) & (pool.n_atoms <= r.n_atoms * (1 + TOL))
                 & ~pool.chemenv_id.isin(taken)]
        c = c.assign(d=(c.n_atoms - r.n_atoms).abs()).sort_values(["d", "order"]).head(N_CONTROLS)
        taken.update(c.chemenv_id)
        rows += [(x, r.chemenv_id) for x in c.chemenv_id]
    return split.chemenv_id.tolist(), pd.DataFrame(rows, columns=["chemenv_id", "matched_to"])


def purity_null(frame, material, comp, rng, nperm):
    """Material purity = share of atoms lying in the majority region of their
    material; returned with its permutation null mean and p-value. Materials are reassigned to
    whole frames (each keeps its number of frames), so atoms of one frame stay
    together and only the frame->material link is broken."""
    fcode, fr = pd.factorize(frame)
    ccode = pd.factorize(comp)[0]
    F = np.zeros((len(fr), ccode.max() + 1))
    np.add.at(F, (fcode, ccode), 1)
    fm = pd.Series(np.asarray(material)).groupby(fcode).first().to_numpy()
    mcode = pd.factorize(fm)[0]
    n = F.sum()

    def pur(mc):
        M = np.zeros((mc.max() + 1, F.shape[1]))
        np.add.at(M, mc, F)
        return M.max(1).sum() / n

    obs = pur(mcode)
    null = np.array([pur(rng.permutation(mcode)) for _ in range(nperm)])
    return obs, float(null.mean()), (1 + int((null >= obs - 1e-12).sum())) / (nperm + 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report-card", required=True)
    ap.add_argument("--feat-dir", required=True)
    ap.add_argument("--train-coverage", required=True)
    ap.add_argument("--pca-cache", default=None)
    ap.add_argument("--min-atoms", type=int, default=C.MIN_ATOMS)
    ap.add_argument("--nperm", type=int, default=1000)
    ap.add_argument("--n-rematch", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-table", default=None)
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    rng = np.random.default_rng(a.seed)
    pd.set_option("display.width", 200)

    rc = C.load_report_card(a.report_card, a.min_atoms)
    N, S = len(rc), int(rc.split.sum())

    # ------------------------------------------------------------------ [A]
    C.hdr("[A] HOW MANY DESCRIPTORS SPLIT")
    print(f"  descriptors with >= {a.min_atoms} atoms: {N:,}")
    print(f"  single region: {N - S:,} ({100 * (N - S) / N:.2f} %)   "
          f"split (n_components >= 2): {S} ({100 * S / N:.2f} %)")
    print("  components among the split ones:",
          rc.loc[rc.split, "n_components"].astype(int).value_counts().sort_index().to_dict())

    # ------------------------------------------------------------------ [B]
    C.hdr("[B] DO SPLIT DESCRIPTORS DIFFER IN ANY DESCRIPTOR FIELD?")
    print(f"  chi^2 of split vs single-region over the levels of each field that hold\n"
          f"  >= {MIN_LEVEL} descriptors; V = Cramer's V, V_bc = bias-corrected V.")
    rows = []
    for col, lab in FIELDS:
        vc = rc[col].value_counts()
        if (vc >= MIN_LEVEL).sum() < 2:          # e.g. spin: (almost) all descriptors spin 0
            ct = pd.crosstab(rc[col], rc.split)
            print(f"  {lab}: fewer than 2 levels with >= {MIN_LEVEL} descriptors, not tested;"
                  f" (single, split) per level: "
                  f"{ {k: tuple(int(x) for x in v) for k, v in ct.iterrows()} }")
            continue
        sub = rc[rc[col].isin(vc[vc >= MIN_LEVEL].index)]
        chi2, dof, p, v, vbc = C.cramers_v(sub[col].astype(str), sub.split)
        rows.append(dict(field=lab, levels=int((vc >= MIN_LEVEL).sum()),
                         descriptors=len(sub), chi2=chi2, df=dof, p=p, V=v, V_bc=vbc))
    T = pd.DataFrame(rows).sort_values("V", ascending=False)
    print(T.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"\n  largest Cramer's V over all fields: {T.V.max():.3f} "
          f"({T.iloc[0].field}); conventionally < 0.1 is negligible, 0.1-0.3 weak")
    el = T[T.field == "central element"].iloc[0]
    print(f"  central element: V = {el.V:.3f}, p = {el.p:.2f}")

    # ------------------------------------------------------------------ [C]
    C.hdr("[C] IS THE SPLITTING MATERIAL-BASED?")
    meta = C.read_meta(a.feat_dir)
    basis = C.pca_basis(a.feat_dir, a.pca_cache)
    # Regions are recomputed for EVERY analysed descriptor: the basis here agrees
    # with stage 10.4's only to floating-point precision, so a few kNN graphs
    # (near-tied neighbours) differ; those descriptors are listed, not hidden.
    sm, X = C.gather_latent(a.feat_dir, meta, rc.chemenv_id, basis)
    ncomp_new = {cid: C.knn_components(X[g.index.to_numpy()])
                 for cid, g in sm.groupby("chemenv_id", sort=False)}
    rc["n_comp_recomputed"] = rc.chemenv_id.map({k: v[0] for k, v in ncomp_new.items()})
    dis = rc[rc.n_comp_recomputed != rc.n_components]
    print(f"  components recomputed for {len(rc):,} descriptors; "
          f"{len(dis)} disagree with the report card:")
    for r in dis.itertuples():
        print(f"     {r.chemenv_id:45s} report card {int(r.n_components)} -> "
              f"recomputed {int(r.n_comp_recomputed)}")
    sp = rc[rc.split]
    sm = sm[sm.chemenv_id.isin(set(sp.chemenv_id))]
    sm = sm.assign(material=C.material_of(sm.sid).to_numpy())
    recs = []
    for cid, g in sm.groupby("chemenv_id", sort=False):
        ncomp, lab = ncomp_new[cid]
        obs, null_mean, p = purity_null(g.sid.to_numpy(), g.material.to_numpy(), lab,
                                        rng, a.nperm)
        recs.append(dict(chemenv_id=cid, n_atoms=len(g), n_comp_recomputed=ncomp,
                         largest_comp_frac=np.bincount(lab).max() / len(g),
                         n_materials=g.material.nunique(), material_purity=obs,
                         purity_null_mean=null_mean, purity_p=p))
    R = pd.DataFrame(recs).merge(sp[["chemenv_id", "n_components"]], on="chemenv_id")
    R["purity_q"] = C.bh_fdr(R.purity_p)
    hi = R.material_purity >= 0.9
    multi = R.n_comp_recomputed >= 2
    print(f"  split descriptors (report card): {len(R)}; >= 2 recomputed regions: "
          f"{int(multi.sum())}")
    print(f"  material purity >= 0.90, all report-card splits (paper-comparable; a "
          f"descriptor recomputed as 1 region scores a trivial 1.0): "
          f"{int(hi.sum())} / {len(R)} ({100 * hi.mean():.1f} %)")
    print(f"  material purity >= 0.90, only descriptors with >= 2 recomputed regions: "
          f"{int((hi & multi).sum())} / {int(multi.sum())} "
          f"({100 * (hi & multi).sum() / multi.sum():.1f} %)")
    print(f"     of which hosted by >= 2 materials: {int((hi & (R.n_materials >= 2)).sum())}")
    print(f"  median purity {R.material_purity.median():.3f}  vs frame-level permutation "
          f"null {R.purity_null_mean.median():.3f}")
    sig = (R.purity_q < 0.05) & (R.n_materials >= 2)
    print(f"  purity above its null (BH q < 0.05, >= 2 materials): {int(sig.sum())} "
          f"({100 * sig.mean():.1f} %); purity >= 0.9 AND above null: "
          f"{int((sig & hi).sum())} ({100 * (sig & hi).mean():.1f} %)")
    print(f"  split inside a single material: {int((R.n_materials == 1).sum())}")
    print(f"  split with largest region >= 90 % of the atoms: "
          f"{int((R.largest_comp_frac >= 0.9).sum())}")

    # ------------------------------------------------------------------ [D]
    C.hdr("[D] CHEMICAL DIVERSITY OF THE HOST MATERIALS")
    cells = C.cell_elements(meta)
    am = meta[meta.chemenv_id.isin(set(rc.chemenv_id))]
    nsys = (am.assign(chemsys=am.sid.map(cells).to_numpy())
            .groupby("chemenv_id").chemsys.nunique().rename("n_chemsys"))
    rc = rc.merge(nsys, left_on="chemenv_id", right_index=True, how="left")
    s_med = rc.loc[rc.split, "n_chemsys"].median()
    one = rc[~rc.split]
    print(f"  distinct host chemical systems per descriptor (median): split {s_med:.0f}, "
          f"single-region {one.n_chemsys.median():.0f}")
    k = int((one.n_chemsys >= s_med).sum())
    print(f"  single-region descriptors with >= {s_med:.0f} chemical systems "
          f"(the split median): {k:,} ({100 * k / len(one):.1f} %) -- "
          f"{k / S:.0f}x the number of split descriptors")
    rc["band"] = pd.cut(rc.n_atoms, SIZE_BANDS, right=False)
    t = rc.groupby(["band", "split"], observed=True).n_chemsys.median().unstack()
    t.columns = ["single median", "split median"]
    t["single >= split median, %"] = [
        100 * (rc[(rc.band == b) & ~rc.split].n_chemsys >= t.loc[b, "split median"]).mean()
        for b in t.index]
    print("\n  by descriptor size (n_atoms band):")
    print(t.to_string(float_format=lambda x: f"{x:.1f}"))

    # ------------------------------------------------------------------ [E]
    C.hdr("[E] TRAINING-SET COVERAGE")
    # files from the original extraction call the formula count n_prototypes
    cov = pd.read_parquet(a.train_coverage).rename(columns={"n_prototypes": "n_formulas"})
    if cov.n_unparsed_mat_id.sum():
        raise SystemExit("training coverage contains unparsed mat_ids")
    D = rc.merge(cov, left_on="chemenv_id", right_on="env_id", how="left")
    D["in_train"] = D.n_frames.notna()
    a_s, a_n = int((~D.in_train & D.split).sum()), int((~D.in_train).sum())
    p_s, p_n = int((D.in_train & D.split).sum()), int(D.in_train.sum())
    from scipy.stats import fisher_exact
    fe = fisher_exact([[a_s, a_n - a_s], [p_s, p_n - p_s]])
    print(f"  absent from training: {a_n:,} descriptors, {a_s} split ({100 * a_s / a_n:.2f} %)")
    print(f"  present in training : {p_n:,} descriptors, {p_s} split ({100 * p_s / p_n:.2f} %)")
    print(f"  Fisher exact: odds ratio {fe.statistic:.2f}, p = {fe.pvalue:.2g}")
    # Size adjustment: absent descriptors are smaller, and small descriptors
    # rarely split, so the same contrast is repeated within n_atoms strata.
    absent = (~D.in_train).to_numpy(float)
    y = D.split.to_numpy(int)
    band = pd.cut(D.n_atoms, SIZE_BANDS, right=False).cat.codes.to_numpy()
    num = den = cmh_a = cmh_e = cmh_v = 0.0
    for b in np.unique(band):
        m = band == b
        a11 = ((absent == 1) & (y == 1) & m).sum(); a12 = ((absent == 1) & (y == 0) & m).sum()
        a21 = ((absent == 0) & (y == 1) & m).sum(); a22 = ((absent == 0) & (y == 0) & m).sum()
        n = a11 + a12 + a21 + a22
        if n < 2:
            continue
        num += a11 * a22 / n
        den += a12 * a21 / n
        r1, c1 = a11 + a12, a11 + a21
        cmh_a += a11
        cmh_e += r1 * c1 / n
        cmh_v += r1 * (n - r1) * c1 * (n - c1) / (n * n * (n - 1))
    from scipy.stats import chi2 as chi2_dist
    cmh = (cmh_a - cmh_e) ** 2 / cmh_v
    print(f"  Cochran-Mantel-Haenszel over the n_atoms bands {SIZE_BANDS[:-1]}: "
          f"OR_MH {num / den:.2f}, chi2 {cmh:.2f}, p = {chi2_dist.sf(cmh, 1):.2g}")
    f = C.logit(y, np.column_stack([absent, np.log(D.n_atoms)]), ["absent", "log_n"])
    r = f["table"].loc["absent"]
    print(f"  logistic regression split ~ absent + log n_atoms: OR {r.odds_ratio:.2f} "
          f"[{np.exp(r.coef - 1.96 * r.se):.2f}, {np.exp(r.coef + 1.96 * r.se):.2f}], "
          f"p = {r.p:.2g}")

    def compare(split_ids, ctrl_ids):
        A = D[D.chemenv_id.isin(split_ids) & D.in_train]
        B = D[D.chemenv_id.isin(ctrl_ids) & D.in_train]
        out = []
        for c in COV_COLS:
            auc, p = C.mwu_auc(A[c], B[c])
            out.append(dict(quantity=c, split_median=A[c].median(),
                            single_median=B[c].median(), AUC=auc, p=p,
                            n_split=len(A), n_single=len(B)))
        return pd.DataFrame(out), A, B

    split_ids, ctrl = size_matched_controls(rc)
    R2, A, B = compare(split_ids, ctrl.chemenv_id)
    print(f"\n  size-matched: {len(split_ids)} split descriptors, {len(ctrl)} single-region "
          f"controls ({N_CONTROLS} each, n_atoms within +/-{100 * TOL:.0f} %); present in "
          f"training: {len(A)} vs {len(B)}; median n_atoms {A.n_atoms.median():.0f} vs "
          f"{B.n_atoms.median():.0f}")
    print("  AUC = P(split > single); 0.5 = no difference; two-sided Mann-Whitney p")
    print(R2.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    sens = []
    for _ in range(a.n_rematch):
        s_ids, c = size_matched_controls(rc, rng)
        r, _, _ = compare(s_ids, c.chemenv_id)
        sens.append(r.set_index("quantity")[["AUC", "p"]].stack())
    Sx = pd.concat(sens, axis=1).T
    print(f"\n  robustness: {a.n_rematch} re-matchings with random tie-breaking")
    for c in ("n_frames", "n_materials", "n_atom_occurrences"):
        auc, p = Sx[(c, "AUC")], Sx[(c, "p")]
        print(f"     {c:20s} AUC median {auc.median():.3f} (range {auc.min():.3f}-"
              f"{auc.max():.3f}); p < 0.05 in {100 * (p < 0.05).mean():.0f} % of matchings")

    if a.out_table:
        R.to_parquet(a.out_table, index=False)
        C.log(f"wrote {a.out_table}")


if __name__ == "__main__":
    main()
