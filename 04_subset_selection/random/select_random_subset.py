#!/usr/bin/env python3
# As-run filename: generate_random_train_val_split.py
"""
Random baseline subset of OMat24.

Draws N materials uniformly at random from the materials composed only of the
allowed elements, with every allowed element present at least once:

    pool      = materials whose elements are all in ALLOWED_ATOMIC_NUMBERS
    reserved  = one material per allowed element not yet covered (35 in the paper)
    fill      = N - len(reserved) materials drawn from pool \\ reserved
                (DuckDB reservoir sample, REPEATABLE(seed))
    selection = reserved + fill

Composition is read from the `material_elements` table of stage 02's
freq_chemenv.duckdb (the distinct central elements of a material's tokens;
every site is a central atom, so this is its full element set).

Output, in --output-dir:
    random_subset_mat_ids.txt   the selected material ids, one per line
    train.xyz                   the selected frames, copied verbatim from the
                                OMat24 extxyz files (stage 06's train_file)
"""

import argparse
import glob
import os
import time

import duckdb
from ase.data import atomic_numbers as Z_OF_SYMBOL
from ase.data import chemical_symbols as SYMBOL_OF_Z

# The model's z-table: {1..56} U {72..83}, i.e. no f-block.
ALLOWED_ATOMIC_NUMBERS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
    39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
]


def select_mat_ids(con, allowed_symbols, n_target, seed):
    """Return n_target mat_ids (or the whole pool if smaller), drawn at random
    from the all-allowed materials, covering every allowed element in the pool."""
    con.execute("CREATE OR REPLACE TEMP TABLE _allowed(sym VARCHAR)")
    con.executemany("INSERT INTO _allowed VALUES (?)", [(s,) for s in allowed_symbols])

    print("Building candidate pool (materials with only allowed elements)...", flush=True)
    t0 = time.time()
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE _pool AS
        SELECT mat_id, elements
        FROM material_elements me
        WHERE len(me.elements) > 0
          AND NOT EXISTS (
              SELECT 1 FROM unnest(me.elements) AS t(e)
              WHERE e NOT IN (SELECT sym FROM _allowed)
          )
        """
    )
    pool_size = con.execute("SELECT COUNT(*) FROM _pool").fetchone()[0]
    print(f"  Pool size: {pool_size:,} materials  ({(time.time()-t0)/60:.2f} min)", flush=True)

    present = set(
        con.execute("SELECT DISTINCT e FROM _pool, unnest(elements) AS t(e)")
        .fetchnumpy()["e"].tolist()
    )
    missing_from_pool = sorted(set(allowed_symbols) - present)
    if missing_from_pool:
        print(f"  WARNING: {len(missing_from_pool)} allowed element(s) never occur in any "
              f"all-allowed material and cannot be covered: {missing_from_pool}", flush=True)

    if pool_size <= n_target:
        print(f"  Pool ({pool_size:,}) <= target ({n_target:,}); taking the entire pool.", flush=True)
        return con.execute("SELECT mat_id FROM _pool").fetchnumpy()["mat_id"].tolist()

    # Reserve one material per not-yet-covered element (guarantees coverage).
    print("Reserving materials to guarantee element coverage...", flush=True)
    reserved = {}
    covered = set()
    for sym in sorted(present):
        if sym in covered:
            continue
        row = con.execute(
            "SELECT mat_id, elements FROM _pool WHERE list_contains(elements, ?) LIMIT 1",
            [sym],
        ).fetchone()
        if row is None:
            continue
        mat_id, elems = row
        reserved[mat_id] = elems
        covered.update(elems)
    print(f"  Reserved {len(reserved):,} materials covering {len(covered):,} elements.", flush=True)

    # Fill the remainder with a seeded uniform random sample.
    n_fill = n_target - len(reserved)
    con.execute("CREATE OR REPLACE TEMP TABLE _reserved(mat_id VARCHAR)")
    if reserved:
        con.executemany("INSERT INTO _reserved VALUES (?)", [(m,) for m in reserved])

    print(f"Reservoir-sampling {n_fill:,} more materials (seed={seed})...", flush=True)
    t0 = time.time()
    fill = con.execute(
        f"""
        SELECT mat_id
        FROM _pool
        WHERE mat_id NOT IN (SELECT mat_id FROM _reserved)
        USING SAMPLE reservoir({n_fill} ROWS) REPEATABLE ({seed})
        """
    ).fetchnumpy()["mat_id"].tolist()
    print(f"  Sampled {len(fill):,} materials  ({(time.time()-t0)/60:.2f} min)", flush=True)

    selected = list(reserved.keys()) + fill
    assert len(set(selected)) == len(selected), "duplicate mat_ids in selection"
    return selected


def write_train_xyz(selected_mat_ids, xyz_dir, train_path):
    """Stream the OMat24 extxyz files and copy every selected frame verbatim."""
    wanted = set(selected_mat_ids)
    xyz_files = sorted(glob.glob(f"{xyz_dir}/*/*xyz"))
    print(f"\nScanning {len(xyz_files)} xyz file(s) under {xyz_dir} ...", flush=True)

    n_scanned = n_written = 0
    seen = set()
    with open(train_path, "w") as train_f:
        for xyz_path in xyz_files:
            print(f"  Reading {xyz_path}...", flush=True)
            with open(xyz_path, "r") as f:
                while True:
                    line1 = f.readline()
                    if not line1:
                        break
                    n_atoms = int(line1.strip())
                    comment = f.readline()
                    atom_lines = [f.readline() for _ in range(n_atoms)]
                    n_scanned += 1

                    sid = None
                    for token in comment.split():
                        if token.startswith("sid="):
                            sid = token[4:]
                            break
                    if sid is None:
                        raise ValueError(f"No 'sid=' in comment of frame {n_scanned} in {xyz_path}.\n"
                                         f"Comment: {comment}")
                    if sid not in wanted:
                        continue

                    train_f.write(line1 + comment + "".join(atom_lines))
                    seen.add(sid)
                    n_written += 1
            print(f"    {n_scanned:,} frames scanned, {n_written:,} selected so far.", flush=True)

    print(f"\nWrote {n_written:,} structures -> {train_path}")
    not_found = wanted - seen
    if not_found:
        print(f"WARNING: {len(not_found):,} selected mat_ids were not found in any xyz file "
              f"(e.g. {sorted(not_found)[:5]}).", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freq-db", required=True, help="freq_chemenv.duckdb from stage 02.")
    ap.add_argument("--xyz-dir", required=True,
                    help="OMat24 training split as extxyz, laid out as <dir>/<subset>/*xyz.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-target", type=int, default=356249,
                    help="number of materials (default: the paper's 356,249, ~6.5 M atoms).")
    ap.add_argument("--seed", type=int, required=True, help="seed of the reservoir sample.")
    ap.add_argument("--threads", type=int, default=128, help="DuckDB threads.")
    ap.add_argument("--memory-limit", default="500GB", help="DuckDB memory limit.")
    args = ap.parse_args()

    allowed_symbols = [SYMBOL_OF_Z[z] for z in ALLOWED_ATOMIC_NUMBERS]
    assert all(Z_OF_SYMBOL[s] == z for s, z in zip(allowed_symbols, ALLOWED_ATOMIC_NUMBERS))
    print(f"Allowed elements ({len(allowed_symbols)}): {allowed_symbols}", flush=True)

    t_connect = time.time()
    con = duckdb.connect(args.freq_db, read_only=True,
                         config={"threads": args.threads, "memory_limit": args.memory_limit})
    print(f"Connected to DB ({(time.time()-t_connect)/60:.2f} min).", flush=True)

    selected = select_mat_ids(con, allowed_symbols, args.n_target, args.seed)
    print(f"\nSelected {len(selected):,} materials total.", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    ids_path = os.path.join(args.output_dir, "random_subset_mat_ids.txt")
    with open(ids_path, "w") as f:
        f.write("\n".join(selected))
    print(f"Wrote selected mat_ids -> {ids_path}", flush=True)
    con.close()

    write_train_xyz(selected, args.xyz_dir, os.path.join(args.output_dir, "train.xyz"))


if __name__ == "__main__":
    main()
