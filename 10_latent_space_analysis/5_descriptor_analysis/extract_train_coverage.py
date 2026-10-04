#!/usr/bin/env python3
"""Stage 10.5 -- training-set coverage of the analysed EnvTok descriptors.

Backs, in SI S5.2 ("Descriptors with multiple disjoint regions"):
  * "compared with single-region descriptors of the same size, split
    descriptors appear about as often in the training set, but in fewer
    different materials"
  * "Descriptors that never appear in the training set are the least likely
    to split."
The statistics themselves are computed by split_descriptors.py; this script
only counts, for every analysed descriptor (>= --min-atoms atoms in the
validation set), how it occurs in the TRAINING set.

Input
  --db           stage 03's material_vectors.duckdb. Its table env_counts has
                 one row per (mat_id, env_id) with the per-frame count; mat_id
                 is the OMat24 structure id ('agm<digits>_<prototype>_..._spg<n>_
                 ..._<sub-dataset>_...'), env_id the token string. Opened
                 READ-ONLY (attached to an in-memory catalog), so this script
                 cannot modify it.
  --report-card  stage 10.4's report-card parquet; it defines the descriptors
                 (tokens with the kNN-graph metrics and >= --min-atoms atoms).

Output (--out, parquet): one row per descriptor PRESENT in the training set
  env_id, n_frames (frames containing it), n_atom_occurrences (summed counts),
  n_materials (distinct agm ids), n_formulas (distinct anonymous formulas,
  e.g. 'ABC2', the 2nd field of the id), n_spg, n_provenance (distinct
  sub-datasets, rattled-* / aimd-*), n_unparsed_mat_id (must be 0). An id
  field that does not match its pattern is not counted as a value.
Descriptors absent from training get no row -- that absence is the finding.
A JSON next to it records the table sizes and the database fingerprint.

Environment: env A (duckdb, pandas, pyarrow). The query is a full scan of
env_counts (one row per frame and token; there is no index on env_id), so run
it on the machine that holds the database.
"""

import argparse
import json
import os
import sys
import time

import duckdb
import pandas as pd

# ============================================================================
# >>> USER CONFIGURATION -- pass these on the command line <<<
#
#   --db           stage 03 output material_vectors.duckdb (TOKEN_VEC_DB there).
#   --report-card  stage 10.4 output report_card.parquet (REPORT_OUT there).
#   --out          output parquet; read by split_descriptors.py --train-coverage.
#   --min-atoms    must equal split_descriptors.py --min-atoms (manuscript: 30).
#   --threads / --memory   DuckDB resources.
# ============================================================================

# Fields of the structure id, as regular expressions on mat_id.
AGM = r"'^(agm[0-9]+)'"                  # material
FORMULA = r"'^agm[0-9]+_([^_]+)'"        # anonymous formula, e.g. ABC2 (2nd field)
SPG = r"'_spg([0-9]+)'"                  # space-group number
PROV = r"'(rattled[^_]*|aimd[^_]*)'"     # sub-dataset


def fingerprint(path):
    st = os.stat(path)
    return dict(size_bytes=st.st_size, mtime=st.st_mtime, inode=st.st_ino)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", required=True)
    ap.add_argument("--report-card", required=True)
    ap.add_argument("--out", default="train_coverage.parquet")
    ap.add_argument("--min-atoms", type=int, default=30)
    ap.add_argument("--threads", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--memory", default="64GB")
    a = ap.parse_args()

    db = os.path.abspath(a.db)
    if not os.path.isfile(db):
        sys.exit(f"ERROR: no such database: {db}")
    if os.path.exists(db + ".wal"):
        # a read-write open would replay the WAL and rewrite the file
        sys.exit(f"ERROR: {db}.wal exists; refusing to open the database")
    before = fingerprint(db)

    # Target descriptors: exactly the population split_descriptors.py analyses.
    rc = pd.read_parquet(a.report_card, columns=["chemenv_id", "n_atoms", "n_components"])
    tgt = rc.loc[rc.n_components.notna() & (rc.n_atoms >= a.min_atoms), ["chemenv_id"]]
    tgt = tgt.rename(columns={"chemenv_id": "env_id"}).drop_duplicates()
    print(f"{len(tgt):,} target descriptors (n_atoms >= {a.min_atoms})", flush=True)

    t0 = time.time()
    con = duckdb.connect(":memory:", config={"threads": a.threads,
                                             "memory_limit": a.memory})
    con.execute("ATTACH '{}' AS src (READ_ONLY)".format(db.replace("'", "''")))
    con.register("tgt_df", tgt)
    con.execute("CREATE TABLE tgt AS SELECT DISTINCT env_id FROM tgt_df")
    meta = dict(source=db, source_before=before, n_targets=len(tgt),
                min_atoms=a.min_atoms, duckdb=duckdb.__version__)
    for t in ("env_counts", "material_vectors", "environment_mapping"):
        meta[f"rows_{t}"] = con.execute(f"SELECT count(*) FROM src.{t}").fetchone()[0]
        print(f"src.{t}: {meta[f'rows_{t}']:,} rows", flush=True)

    cov = con.execute(f"""
        SELECT ec.env_id,
               count(*)                                              AS n_frames,
               sum(ec."count")                                       AS n_atom_occurrences,
               count(DISTINCT NULLIF(regexp_extract(ec.mat_id, {AGM},     1), ''))
                                                                     AS n_materials,
               count(DISTINCT NULLIF(regexp_extract(ec.mat_id, {FORMULA}, 1), ''))
                                                                     AS n_formulas,
               count(DISTINCT NULLIF(regexp_extract(ec.mat_id, {SPG},     1), ''))
                                                                     AS n_spg,
               count(DISTINCT NULLIF(regexp_extract(ec.mat_id, {PROV},    1), ''))
                                                                     AS n_provenance,
               sum(CASE WHEN regexp_extract(ec.mat_id, {AGM}, 1) = '' THEN 1 ELSE 0 END)
                                                                     AS n_unparsed_mat_id
        FROM src.env_counts ec
        JOIN tgt t ON t.env_id = ec.env_id
        GROUP BY ec.env_id
        ORDER BY ec.env_id""").df()
    con.close()

    cov.to_parquet(a.out, index=False)
    meta.update(rows_coverage=len(cov), targets_absent_from_training=len(tgt) - len(cov),
                unparsed_mat_ids=int(cov.n_unparsed_mat_id.sum()),
                source_after=fingerprint(db), seconds=round(time.time() - t0, 1))
    meta["source_unchanged"] = meta["source_before"] == meta["source_after"]
    with open(os.path.splitext(a.out)[0] + "_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {a.out}: {len(cov):,} descriptors present in training, "
          f"{meta['targets_absent_from_training']:,} absent; "
          f"source unchanged: {meta['source_unchanged']}", flush=True)
    if meta["unparsed_mat_ids"]:
        sys.exit("ERROR: some mat_id did not match the agm<digits> pattern")


if __name__ == "__main__":
    main()
