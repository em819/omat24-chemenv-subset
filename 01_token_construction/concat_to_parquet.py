
"""Concatenate the per-rank chemenv HDF5 files of one OMat24 subset into a single
parquet, with constant memory, completeness checking, and a failure-token report.

Usage:
    python concat_to_parquet.py <subset_dir> [expected_ranks]

    <subset_dir>     directory holding the per-rank files (chemenv_*_rank*.hdf5),
                     e.g. rattled-1000-subsampled
    [expected_ranks] optional int: the number of MPI ranks the chemenv job used.
                     If given, the script flags missing/extra rank files.

Improvements over the original concat_to_parquet.py:
  * Streams HDF -> parquet one file at a time via a pyarrow ParquetWriter, so peak
    memory is one rank file, not the whole subset (the old pd.concat held every
    rank in RAM plus a full second copy -> OOM on the large aimd subsets).
  * Creates the output directory if missing (old to_parquet crashed if it wasn't).
  * Reports per-file row counts, files that fail to read (truncated/corrupt ranks),
    and gaps in the rank numbering -> a silently-missing rank is no longer invisible.
  * Scans every row for the failure tokens emitted by build_coord_descriptor and
    prints a summary at the end:
        - coord_symbol == 'INVALID'   : site fully unassigned (chemenv AND CrystalNN failed)
        - coord_symbol  ~ 'UNKNOWN:n' : neighbours found but no ideal polyhedron
                                        (chemenv 'UNKNOWN' or CrystalNN rescue; perturb_scale=999)
      and a cross-check that the env_id perturb_scale=999 count matches INVALID+UNKNOWN.

Reading the token
-----------------
The field count is NOT fixed at 7, because 11 chemenv symbols contain an
underscore themselves:
    {central}_{spin}_{coords}_{symbol}_{pert}_{rmin_bin}_{a_bin}
perturb_scale is not the last field, so it cannot be tested with
`env_id.endswith("_999")`.
It is now the ANTEPENULTIMATE field, and the last field is {rmax_bin} -- which
carries its own, DIFFERENT 999 sentinel (DIST_BIN_UNDEFINED, meaning "no neighbour
set to measure distances against"). The two do not coincide: a CrystalNN-rescued
'UNKNOWN:n' site has perturb_scale=999 but perfectly well-defined distance bins.
PERTURB_999_RE below pins the 999 to the perturb_scale position so the count only
picks up sites that really have no environment.

The two new float columns (dist_ratio_min, dist_ratio_max) need no handling: whole
frames are streamed through, so they are carried into the parquet automatically.

_and_A variant
--------------
Reads the token of analyze_db_parallel_fixed_spin_twocutoffs_dists_and_A.py:
    {central}_{spin}_{coords}_{symbol}_{pert}_{rmin_bin}_{a_bin}

NOTHING IN THIS FILE NEEDED A CODE CHANGE, and that is worth stating explicitly
rather than leaving it to be rediscovered:

  * PERTURB_999_RE still pins 999 to the perturb_scale slot. The regex does not
    care what the last two fields MEAN, only that exactly two integer fields follow
    the 999 -- and the _and_A token still has exactly two ({rmin_bin} and {a_bin},
    where {rmax_bin} used to sit). The two cases that matter are
    '..._0_999_999', which must NOT match because perturb is 0 and it is the two
    trailing fields that are undefined, and '..._999_999_999', which must.

  * a_bin's undefined sentinel is the same 999 as the distance bins', so it cannot
    introduce a new collision either.

  * The three new columns (a_raw, a_bin, and dist_bin_max -- still WRITTEN to HDF5
    even though it is no longer part of the token) need no handling for the same
    reason the _dists float columns did not: whole frames are streamed through and
    the parquet schema is locked in from the first frame, so any column the
    tokeniser writes is carried through automatically.

This file exists as a separate _and_A copy to keep one downstream stack per token
definition. Mixing parquets built from _dists tokens with parquets built from
_and_A tokens would join SILENTLY and WRONGLY: the two token strings have the same
shape and differ only in the meaning of the last field and the numbering of the
one before it, so nothing raises.
"""
import glob
import os
import re
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HDF_KEY = "data"
OUTDIR = "parquet_file_collection"
RANK_RE = re.compile(r"rank(\d+)", re.IGNORECASE)
EXAMPLE_CAP = 10  # how many example mat_ids to keep per failure category

# perturb_scale == 999, i.e. '_999_<rmin_bin>_<rmax_bin>' at the very end of env_id.
# Requiring two further '_'-separated integer fields after the 999 is what pins it to
# the perturb_scale slot: '..._0_999_999' (perturb 0, both bins undefined) does NOT
# match, while '..._999_2_2' and '..._999_999_999' both do.
PERTURB_999_RE = re.compile(r"_999_\d+_\d+$")


def parse_rank(path):
    """Extract the integer rank from a filename like chemenv_<set>_rank7.hdf5."""
    m = RANK_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else None


def scan_failures(df):
    """Return per-file failure counts and the affected mat_ids for one rank frame."""
    cs = df["coord_symbol"].astype(str)
    invalid_mask = cs == "INVALID"
    unknown_mask = cs.str.startswith("UNKNOWN")
    suffix_999 = df["env_id"].astype(str).str.contains(PERTURB_999_RE, regex=True).sum()
    return {
        "rows": len(df),
        "invalid": int(invalid_mask.sum()),
        "unknown": int(unknown_mask.sum()),
        "suffix_999": int(suffix_999),
        "invalid_mat_ids": set(df.loc[invalid_mask, "mat_id"].astype(str)),
        "unknown_mat_ids": set(df.loc[unknown_mask, "mat_id"].astype(str)),
    }


def concat_hdfs_to_parquet(subset_dir, expected_ranks=None):
    label = os.path.basename(os.path.normpath(subset_dir))
    os.makedirs(OUTDIR, exist_ok=True)
    outfile = os.path.join(OUTDIR, f"total_chemenv_{label}.parquet")

    hdf_files = sorted(glob.glob(os.path.join(subset_dir, "chemenv*hdf5")))
    n_files = len(hdf_files)
    print(f"Found {n_files} files to concatenate in '{subset_dir}'")
    if n_files == 0:
        print("ERROR: no chemenv*hdf5 files matched — nothing to do. "
              "Check the subset directory argument.")
        return 1

    # --- completeness: rank-number gaps -------------------------------------
    ranks = {parse_rank(f) for f in hdf_files}
    ranks.discard(None)
    missing_ranks = []
    if ranks:
        hi = max(ranks)
        missing_ranks = [r for r in range(hi + 1) if r not in ranks]
    if expected_ranks is not None:
        missing_ranks = sorted(set(missing_ranks) |
                               {r for r in range(expected_ranks) if r not in ranks})

    # --- stream each file: write to parquet + accumulate failure counts ------
    writer = None
    schema = None
    total_rows = 0
    tot_invalid = tot_unknown = tot_999 = 0
    invalid_mat_ids = set()
    unknown_mat_ids = set()
    unreadable = []      # (filename, error) for files that could not be read
    per_file = []        # (filename, rows, invalid, unknown) for the per-file log

    try:
        for i, f in enumerate(hdf_files):
            try:
                df = pd.read_hdf(f, key=HDF_KEY)
            except Exception as e:
                unreadable.append((f, repr(e)))
                print(f"  [{i + 1}/{n_files}] !! UNREADABLE {os.path.basename(f)}: {e}",
                      flush=True)
                continue

            stats = scan_failures(df)
            total_rows += stats["rows"]
            tot_invalid += stats["invalid"]
            tot_unknown += stats["unknown"]
            tot_999 += stats["suffix_999"]
            invalid_mat_ids |= stats["invalid_mat_ids"]
            unknown_mat_ids |= stats["unknown_mat_ids"]
            per_file.append((os.path.basename(f), stats["rows"],
                             stats["invalid"], stats["unknown"]))

            # Write this frame to parquet under a single, locked-in schema.
            if schema is None:
                table = pa.Table.from_pandas(df, preserve_index=False)
                schema = table.schema
                writer = pq.ParquetWriter(outfile, schema, compression="snappy")
            else:
                table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
            writer.write_table(table)

            if i % 10 == 0 or stats["invalid"] or stats["unknown"]:
                print(f"  [{i + 1}/{n_files}] {os.path.basename(f)}: "
                      f"{stats['rows']} rows  (INVALID={stats['invalid']}, "
                      f"UNKNOWN={stats['unknown']})", flush=True)
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        print("ERROR: every file failed to read — no parquet written.")
        _print_summary(label, outfile, n_files, ranks, missing_ranks, expected_ranks,
                       total_rows, tot_invalid, tot_unknown, tot_999,
                       invalid_mat_ids, unknown_mat_ids, unreadable, written=False)
        return 1

    print(f"\nWrote {total_rows} rows to {outfile}")
    _print_summary(label, outfile, n_files, ranks, missing_ranks, expected_ranks,
                   total_rows, tot_invalid, tot_unknown, tot_999,
                   invalid_mat_ids, unknown_mat_ids, unreadable, written=True)
    # non-zero exit if anything looks incomplete, so a calling shell loop notices
    return 1 if (unreadable or missing_ranks) else 0


def _print_summary(label, outfile, n_files, ranks, missing_ranks, expected_ranks,
                   total_rows, tot_invalid, tot_unknown, tot_999,
                   invalid_mat_ids, unknown_mat_ids, unreadable, written):
    def pct(x):
        return f"{100.0 * x / total_rows:.4f}%" if total_rows else "n/a"

    print("\n" + "=" * 70)
    print(f"SUMMARY — {label}")
    print("=" * 70)

    # ---- completeness ------------------------------------------------------
    print("Completeness:")
    print(f"  files matched         : {n_files}")
    print(f"  files read OK         : {n_files - len(unreadable)}")
    print(f"  files unreadable      : {len(unreadable)}")
    for f, err in unreadable:
        print(f"     - {os.path.basename(f)}: {err}")
    if ranks:
        print(f"  rank range present    : 0..{max(ranks)} ({len(ranks)} distinct)")
    if expected_ranks is not None:
        print(f"  expected ranks        : {expected_ranks}")
    if missing_ranks:
        print(f"  !! MISSING RANK FILES : {missing_ranks}")
    else:
        print("  missing rank files    : none")

    # ---- chemenv failure tokens -------------------------------------------
    n_fail = tot_invalid + tot_unknown
    print("\nChemenv assignment failures (in the data, per atom):")
    print(f"  total atoms (rows)    : {total_rows}")
    print(f"  INVALID  (no env at all): {tot_invalid:>10}  ({pct(tot_invalid)})  "
          f"affecting {len(invalid_mat_ids)} structures")
    print(f"  UNKNOWN:n (no geometry) : {tot_unknown:>10}  ({pct(tot_unknown)})  "
          f"affecting {len(unknown_mat_ids)} structures")
    print(f"  failed atoms total      : {n_fail:>10}  ({pct(n_fail)})")
    # cross-check: every INVALID and UNKNOWN row carries perturb_scale 999.
    # Matched on the perturb_scale FIELD (see PERTURB_999_RE), not on a trailing
    # '_999', which under the _dists token is the rmax distance bin instead.
    if tot_999 != n_fail:
        print(f"  !! consistency warning  : env_id perturb_scale=999 count = {tot_999} "
              f"but INVALID+UNKNOWN = {n_fail} (expected equal)")
    else:
        print(f"  env_id perturb_scale=999 cross-check: OK ({tot_999} == INVALID+UNKNOWN)")

    if invalid_mat_ids:
        print(f"  example INVALID mat_ids : "
              f"{sorted(invalid_mat_ids)[:EXAMPLE_CAP]}")
    if unknown_mat_ids:
        print(f"  example UNKNOWN mat_ids : "
              f"{sorted(unknown_mat_ids)[:EXAMPLE_CAP]}")

    # ---- verdict -----------------------------------------------------------
    ok = written and not unreadable and not missing_ranks
    print("\nVerdict:", "OK — parquet complete." if ok else
          "ATTENTION — see flagged items above (data may be incomplete).")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python concat_to_parquet.py <subset_dir> [expected_ranks]")
        sys.exit(2)
    subset = sys.argv[1]
    exp = int(sys.argv[2]) if len(sys.argv) > 2 else None
    sys.exit(concat_hdfs_to_parquet(subset, exp))
