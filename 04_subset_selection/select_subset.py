#!/usr/bin/env python3
# ===========================================================================
# Select a minimal subset of OMat24 that covers the space of chemical-environment
# tokens.
#
# The idea in one paragraph: every site in the corpus carries a token describing
# its local environment. Rare tokens cannot each be covered on their own, so
# tokens are grouped into "demand classes" by walking a ladder of progressively
# coarser descriptions until a group has at least T site observations behind it.
# A lazy-greedy (CELF) set cover then picks materials until every demand class is
# covered.
#
# TOKEN GRAMMAR
#
#     <element>_<spin>_<neighbours>_<polyhedron>_<perturb>_<dmin bin>_<a bin>
#
# The field COUNT IS NOT FIXED: 11 chemenv polyhedron symbols contain an
# underscore themselves. Always index the trailing fields from the END, or anchor
# on the ':' inside the symbol. Parsing a token with the wrong field count does
# not raise -- it silently reads the wrong quantity and builds a wrong demand
# space -- so the parse check below rejects tokens whose shape is unexpected.
#
# THE LADDER
#
# A rung is a SQL expression that maps a token to a coarser signature. The rungs
# are generated as a product of identity levels (how much chemistry is kept) and
# a modifier chain (whether the dmin and a_bin fields are kept), with the
# MODIFIER LOOP OUTSIDE the identity loop:
#
#     all identities, keeping dmin and a_bin
#     all identities, keeping a_bin only
#     all identities, keeping neither
#
# So a token gives up its chemistry before it gives up its length scale. That
# ordering is the point: a demand class that still names dmin can only be
# discharged by a material that actually has an atom at that separation, whereas
# a class that has dropped dmin can be discharged by any member, and most of the
# corpus sits at comfortable distances.
#
# COST MODEL
#
#     score = (help + uniformity_weight * uniformity) / site_count ** cost_alpha
#
# site_count(m) is the row sum of the CSR matrix: the number of site observations
# material m contributes inside the demand universe. Under the production
# settings (design 'a', a strict partition where every token maps to exactly one
# class) that equals the material's atom count.
#
# FLOORS
#
# Four floors guarantee that certain coarse combinations are present at all:
# element+polyhedron, element+coordination number, and those two crossed with
# dmin and with a_bin. They are enforced after the greedy, by topping the subset
# up with whatever is missing.
#
# RESTARTS
#
# restart.npz records the configuration it was written with, and a resume is
# refused if any of it differs -- half a subset picked under one rule and half
# under another is not a greedy solution to anything.
#
# RUN THIS IN ITS OWN WORKING DIRECTORY. restart.npz and
# env_ids_found_distribution.txt are fixed filenames, so a leftover restart file
# makes a fresh run silently resume the previous one.
# ===========================================================================

import pandas as pd
import duckdb
import numpy as np
import time
import sys
import json
import hashlib
from collections import defaultdict, Counter
import heapq
import gc
from numba import jit, prange
import numba
print(f"Numba threads configured: {numba.config.NUMBA_NUM_THREADS}")
from scipy import sparse
import pickle
from pathlib import Path
import os 

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# ============================================================================
# >>> USER CONFIGURATION -- EDIT THIS BLOCK (or set the environment variables) <<<
#
# INPUTS -- both are DuckDB files produced earlier in this pipeline. The
# defaults are bare filenames, i.e. "in the current working directory", which is
# the layout the submit script assumes (stages 02, 03 and 04 all run in
# work/selection/). Give absolute paths if yours differ.
#
#   V10_TOKEN_VEC_DB  material_vectors.duckdb   <- written by stage 03
#   V10_ENV_FREQ_DB   freq_chemenv.duckdb       <- written by stage 02
#
# RESOURCES
#
#   V10_THREADS       DuckDB threads (default 128).
#   V10_MEMORY        What DuckDB is ALLOWED to take (default 400GB). NOTE this
#                     is not the real memory requirement: the Python side alone
#                     peaks near 165 GiB holding the CSR matrix and the
#                     material-vector dict, and that allocation has no knob. On
#                     a smaller machine you can only run a smaller corpus.
#
# SELECTION KNOBS -- the published subset used the values below, which are also
# the in-file defaults, so leave them unset to reproduce the paper:
#
#   V10_T=100  V10_DESIGN=a  V10_SUPPORT=materials  V10_NCOV=1  V11_COST_ALPHA=1
#
# Further boolean knobs (V10_FLOORS, V10_TOPUP, V10_DIST_FLOOR, V13A_ASYM_FLOOR,
# V13A_PRUNE, V13A_FULL_TOKEN) are documented beside their defaults in main().
#
# OUTPUTS, written to the CURRENT WORKING DIRECTORY under fixed names:
#
#   best_subset_*.txt                 one selected material id per line
#                                     -> this is stage 05's --subset-mat-ids
#   env_ids_found_distribution.txt    per-token atom counts of the subset
#   restart.npz                       periodic checkpoint
#
# >>> RUN THIS IN ITS OWN, EMPTY WORKING DIRECTORY. <<< restart.npz is a fixed
# filename, so a leftover one from a previous run makes a fresh run silently
# RESUME that run instead of starting over.
# ============================================================================
TOKEN_VEC_DB_DEFAULT = os.environ.get('V10_TOKEN_VEC_DB', 'material_vectors.duckdb')
ENV_FREQ_DB_DEFAULT  = os.environ.get('V10_ENV_FREQ_DB',  'freq_chemenv.duckdb')
DUCK_THREADS_DEFAULT = int(os.environ.get('V10_THREADS', '128'))
DUCK_MEMORY_DEFAULT  = os.environ.get('V10_MEMORY', '400GB')
# ============================================================================

# ---------------------------------------------------------------------------
# BOOLEAN ENVIRONMENT OVERRIDES. Defined HERE, at the top, rather than beside
# main()'s knobs, because ENABLE_FULL_TOKEN_RUNG is a MODULE-LEVEL constant read
# at import time and therefore cannot call a helper defined 200 lines below it.
# That ordering accident is exactly how V13A_FULL_TOKEN came to be parsed by a
# hand-rolled `in ('1','true','yes','on')` test instead of this function -- see
# the docstring for why that silently disabled the rung on any other spelling.
# Anything in this file that reads a boolean from the environment must use
# _env_flag; there is no second parsing rule.
# ---------------------------------------------------------------------------
_ENV_TRUE  = {'1', 'true', 'yes', 'on'}
_ENV_FALSE = {'0', 'false', 'no', 'off'}

def _env_flag(name, default):
    """Parse a boolean environment override, or RAISE.

    The previous form was `os.environ.get(NAME, ...) == '1'`, which silently
    means False for every spelling except '1' -- so V10_DIST_FLOOR=True quietly
    disabled the very floor it was meant to enable, and V10_TOPUP=true quietly
    disabled the top-up. A size knob that fails silently in the OFF direction is
    the worst possible default, because the run still succeeds and the subset is
    merely smaller than intended. Unset keeps `default`; anything unrecognised is
    a configuration error and stops the run before hours of selection.
    """
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    v = raw.strip().lower()
    if v in _ENV_TRUE:
        return True
    if v in _ENV_FALSE:
        return False
    raise ValueError(
        f"{name}={raw!r} is not a boolean. Use one of "
        f"{sorted(_ENV_TRUE)} or {sorted(_ENV_FALSE)}, or leave it unset "
        f"(default {bool(default)}).")


def save_restart_file_pickle(restart_path, iteration, selected, current_coverage):
    start_time = time.time()
    with open(restart_path, 'wb') as f:
        pickle.dump({'iteration': iteration, 'selected': selected, 'current_coverage': current_coverage}, f)
    elapsed = time.time() - start_time
    print(f"[RESTART] Saved iteration {iteration} in {elapsed:.3f}s", flush=True)

def load_restart_file_pickle(restart_path):
    with open(restart_path, 'rb') as f:
        return pickle.load(f)

def save_restart_file(restart_path, iteration, selected, current_coverage,
                      config=None):
    """
    Save restart state using NumPy compressed format.
    Efficient for large arrays and Counters.

    v10: `config` records the coarsening settings. Coverage is keyed by
    demand_index, and demand_index is only meaningful for one particular
    (T, design, support_metric, ladder) combination -- resuming with different
    settings would silently attribute coverage to the wrong classes.
    """
    start_time = time.time()
    
    # Convert Counter to parallel arrays for efficient storage
    if current_coverage:
        env_ids = np.array(list(current_coverage.keys()), dtype=np.int64)
        counts = np.array(list(current_coverage.values()), dtype=np.int64)
    else:
        env_ids = np.array([], dtype=np.int64)
        counts = np.array([], dtype=np.int64)
    
    # Convert selected to array
    selected_arr = np.array(selected, dtype=np.int64)
    
    # Save using compressed NumPy format
    np.savez_compressed(
        restart_path,
        iteration=np.array([iteration], dtype=np.int32),
        selected=selected_arr,
        coverage_env_ids=env_ids,
        coverage_counts=counts,
        config=np.array([json.dumps(config or {}, sort_keys=True)])
    )
    
    elapsed = time.time() - start_time
    file_size_mb = Path(restart_path).stat().st_size / (1024 * 1024)
    print(f"[RESTART] Saved at iteration {iteration} in {elapsed:.3f}s ({file_size_mb:.2f} MB)", flush=True)
    
    return elapsed

def load_restart_file(restart_path):
    """
    Load restart state from NumPy compressed format.
    """
    start_time = time.time()

    data = np.load(restart_path)

    # Reconstruct Counter from parallel arrays
    current_coverage = Counter(
        dict(zip(data['coverage_env_ids'].tolist(),
                data['coverage_counts'].tolist()))
    )

    # 'config' is absent in restart files written by v9 / early v10.
    try:
        cfg = json.loads(str(data['config'][0]))
    except (KeyError, ValueError):
        cfg = None

    state = {
        'iteration': int(data['iteration'][0]),
        'selected': data['selected'].tolist(),
        'current_coverage': current_coverage,
        'config': cfg
    }

    elapsed = time.time() - start_time
    print(f"[RESTART] Loaded iteration {state['iteration']} in {elapsed:.3f}s", flush=True)

    return state



class MatVecPool:
    """The material pool in flat CSR-shaped buffers instead of 42.8M dict entries.

    WHY THIS EXISTS. The load query already produces a CSR: one row per material,
    a list of demand_index and a matching list of counts. `mat_vecs_from_arrow`
    took that apart into 42.8M numpy slices behind 42.8M dict entries, and
    `build_sparse_matrix` then spent fourteen statements putting it back
    together -- 42.8M-element `sorted()`, a 42.8M-element list comprehension, a
    `np.fromiter` over a generator of `len()` calls, two `np.concatenate`s over
    42.8M array objects, and a 675.7M-element `np.searchsorted`. Measured at
    At production shapes that round trip is orders of magnitude slower than
    building the CSR straight from these buffers.

    LAYOUT (exactly scipy's CSR triple, before column sorting):
        mat_index : int64[n_mat]      material ids, STRICTLY ASCENDING
        off       : int64[n_mat + 1]  row start/end into the two flat arrays
        idx_flat  : int64[nnz]        demand_index per stored entry
        cnt_flat  : float64[nnz]      count per stored entry

    INVARIANT -- mat_index is strictly ascending. The loader reads
    `... ORDER BY mat_index`, so row k IS the k-th smallest material id. That is
    what makes `mat_list` equal to the old `sorted(mat_vecs_dict.keys())` without
    sorting anything, and what makes first-k-per-group dedup equal to the old
    `sorted(ids)[:n_coverage]`. Both consumers re-check it rather than trust it.

    This object is duck-compatible with the old dict only where the callers
    actually touch it: `len()`, `.keys()`, and `.items()`. Everything else takes
    the buffer path explicitly via isinstance().
    """

    __slots__ = ('mat_index', 'off', 'idx_flat', 'cnt_flat', '_id_to_row')

    def __init__(self, mat_index, off, idx_flat, cnt_flat):
        self.mat_index = mat_index
        self.off = off
        self.idx_flat = idx_flat
        self.cnt_flat = cnt_flat
        self._id_to_row = None          # built lazily, only by __getitem__

    def __len__(self):
        return len(self.mat_index)

    def __contains__(self, mat_id):
        return int(mat_id) in self._index()

    def __getitem__(self, mat_id):
        """(env_ids, counts) by material id, as the old dict returned.

        ONLY the dict-based fallbacks use this -- collapse_identical_materials
        and build_sparse_matrix both subscript their argument when they cannot
        take the buffer path (a pool whose ids are not ascending). It
        materialises the id->row map that the pool exists to avoid, so it is
        built lazily and never touched on the production path.
        """
        return self.row(self._index()[int(mat_id)])

    def _index(self):
        if self._id_to_row is None:
            self._id_to_row = {int(m): k for k, m in enumerate(self.mat_index)}
        return self._id_to_row

    @property
    def nnz(self):
        return int(self.cnt_flat.size)

    @property
    def nbytes(self):
        return int(self.mat_index.nbytes + self.off.nbytes
                   + self.idx_flat.nbytes + self.cnt_flat.nbytes)

    def is_ascending(self):
        """The INVARIANT above, checked in O(n) with no Python-level iteration."""
        return self.mat_index.size < 2 or bool(np.all(np.diff(self.mat_index) > 0))

    def keys(self):
        """Python ints, matching the old dict's key type exactly (see build_sparse_matrix
        INVARIANT 3: np.int64 keys would leak into the restart file and the SQL
        parameter binding)."""
        return (int(m) for m in self.mat_index)

    def row(self, k):
        """(env_ids, counts) for row k -- views, never copies."""
        a, b = self.off[k], self.off[k + 1]
        return self.idx_flat[a:b], self.cnt_flat[a:b]

    def items(self):
        """(mat_id, (env_ids, counts)) in row order, i.e. ascending mat_id.

        Reproduces what iterating the old dict produced, so the dict-based
        implementations remain usable on a pool for equivalence testing.
        """
        for k in range(len(self.mat_index)):
            yield int(self.mat_index[k]), self.row(k)

    def to_dict(self):
        """The old representation. Only for tests -- this is the thing being avoided."""
        return {int(self.mat_index[k]): self.row(k)
                for k in range(len(self.mat_index))}

    def select_rows(self, keep_rows):
        """A new pool holding only `keep_rows` (ascending row indices).

        Fully vectorised: the surviving entries are gathered with one boolean
        mask over the nnz axis, so no per-material Python work and no
        concatenation over 42.8M array objects.
        """
        lengths = np.diff(self.off)
        row_mask = np.zeros(len(self.mat_index), dtype=bool)
        row_mask[keep_rows] = True
        nnz_mask = np.repeat(row_mask, lengths)
        new_len = lengths[row_mask]
        new_off = np.empty(new_len.size + 1, dtype=np.int64)
        new_off[0] = 0
        np.cumsum(new_len, out=new_off[1:])
        return MatVecPool(self.mat_index[row_mask], new_off,
                          self.idx_flat[nnz_mask], self.cnt_flat[nnz_mask])


def build_sparse_matrix(mat_vecs_dict, env_indices):
    """
    Build sparse matrix representation: materials × environments
    Returns CSR matrix where mat_env_matrix[i, j] = count of env j in material i

    VECTORISED. The previous implementation was a pure-Python triple-append loop

        for mat_id, (env_ids, env_counts) in mat_vecs_dict.items():
            mat_idx = mat_to_idx[mat_id]
            for env_id, count in zip(env_ids, env_counts):
                if env_id in env_to_idx:
                    rows.append(mat_idx); cols.append(...); data.append(count)

    i.e. ~5.6e8 interpreted iterations creating ~5.6e8 short-lived boxed objects
    for the real problem. WHY IT HAD TO GO (measured on OMat24, T=30, mel0295):
    py-spy located a job on the innermost append after 4h55m, 27.2% done
    (mat_idx=11,652,795 of 42,831,061), projecting ~18 hours for a step that the
    earlier successful run completed in ~10 minutes. /proc showed the cause:
    over 30s of wall clock, delta-utime = 0s and delta-stime = 29.93s -- a whole
    core burned inside the KERNEL with the interpreter doing nothing, and only 6
    minor faults, no swap. That is allocator/VM churn (pymalloc arena
    mmap/munmap traffic, whose TLB shootdowns IPI every core the address space
    has touched -- all 128, since DuckDB ran 128 threads earlier in this same
    process), not slow Python. Measured throughput 8,600 nnz/s against 714,000
    nnz/s for the identical loop in a benchmark: 83x.

    The rewrite performs a handful of large C-level array operations instead, so
    there is nothing left to churn. It returns byte-identical results to the
    dict path: CSR indptr/indices/data/dtype, mat_list, env_list and their element
    types.

    THREE INVARIANTS THE REWRITE MUST PRESERVE (each is separately tested):

    1. ROW AND COLUMN ORDER. mat_list/env_list stay `sorted(...)`. Row order is
       not mere bookkeeping: the lazy-greedy heap stores (-score, site_count, row)
       and breaks score ties by SMALLEST SITE COUNT, then by LOWEST ROW, which is
       lowest mat_index; that interacts with collapse_identical_materials keeping
       sorted(ids)[0] and is why a resumed run reproduces a byte-identical subset.
       (v11 note: the site_count key inserted ahead of `row` cannot disturb that
       interaction, because identical materials have identical env vectors and
       therefore identical row sums -- within a duplicate group the middle key is
       constant and lowest-row still decides.) Deriving the order any other way
       would silently yield a different (still valid) subset.
    2. THE OUT-OF-UNIVERSE FILTER. The old code skipped entries via
       `if env_id in env_to_idx`. np.searchsorted does NOT fail on a missing key
       -- it returns an insertion position, i.e. it would silently map an unknown
       environment onto a NEIGHBOURING column. The membership test below is
       therefore explicit, not optional. (It is vacuous on the production data,
       since relevant_mat_vectors is built by joining to demand_mapping and
       env_indices is every distinct demand_index -- but that is a property of
       the query, not of this function.)
    3. mat_list STAYS A PYTHON LIST OF PYTHON int. If it became a numpy array,
       `mat_list[best_mat]` would yield np.int64, so `selected` would fill with
       numpy scalars and flow into `WHERE mat_index = ANY(?)` and the restart
       file with a different scalar type.

    Duplicate (row, col) pairs are summed by the COO->CSR constructor, exactly as
    before; there are none, because the load query does GROUP BY
    (mat_index, demand_index).
    """
    t0 = time.time()
    # FAST PATH: the pool already IS a CSR (see MatVecPool). Falls back to the
    # dict code below for any other input, and for a pool whose ascending-id
    # invariant does not hold.
    if isinstance(mat_vecs_dict, MatVecPool) and mat_vecs_dict.is_ascending():
        return _build_sparse_matrix_from_pool(mat_vecs_dict, env_indices, t0)

    # INVARIANT 1: identical ordering contract to the dict-based version.
    env_list = sorted(env_indices)
    mat_list = sorted(mat_vecs_dict.keys())
    n_mat, n_env = len(mat_list), len(env_list)

    # Row i corresponds to mat_list[i] BY CONSTRUCTION -- this replaces the
    # mat_to_idx dict without materialising 42.8M entries. Both lines are safe on
    # an empty pool (empty comprehension, np.fromiter(count=0)), so they sit above
    # the guard and the two degenerate cases share one exit.
    vals = [mat_vecs_dict[m] for m in mat_list]
    lengths = np.fromiter((len(v[0]) for v in vals), dtype=np.int64, count=n_mat)

    # np.concatenate([]) raises on an empty list, and searchsorted bounds are
    # meaningless without columns, so both degenerate cases short-circuit to the
    # empty matrix the old loop produced. Its nnz is 0, so the single log line
    # below prints identically in every branch.
    if n_env == 0 or not lengths.any():
        matrix = sparse.csr_matrix((n_mat, n_env), dtype=np.float32)
    else:
        # int64 view of the (sorted) column labels, for searchsorted. env_list
        # holds np.int64 in production (main() does set() over a numpy array);
        # the explicit dtype coerces plain ints identically.
        env_arr = np.array(env_list, dtype=np.int64)

        # np.concatenate tolerates zero-length members (only an EMPTY LIST
        # raises), so materials with no environments stay in place and contribute
        # zero rows via `lengths` -- which is what keeps np.repeat aligned.
        all_env = np.concatenate([v[0] for v in vals]).astype(np.int64, copy=False)
        all_cnt = np.concatenate([v[1] for v in vals])
        rows = np.repeat(np.arange(n_mat, dtype=np.int64), lengths)
        del vals, lengths

        # INVARIANT 2: locate each environment, then VERIFY the hit. Without the
        # `keep` mask an out-of-universe env_id would land on a neighbouring
        # column. take(mode='clip') fuses the bounds clamp into the gather, which
        # avoids a separate read+write pass over ~5.6e8 elements; `pos` may then
        # still hold n_env for a non-member, but `keep` is False exactly there
        # which therefore cannot compare equal), so such entries are dropped.
        pos = np.searchsorted(env_arr, all_env)
        keep = env_arr.take(pos, mode='clip') == all_env
        del all_env                      # ~4.5 GB, dead before the CSR is built
        if not keep.all():
            rows, pos, all_cnt = rows[keep], pos[keep], all_cnt[keep]
        del keep

        matrix = sparse.csr_matrix(
            (all_cnt, (rows, pos)),
            shape=(n_mat, n_env),
            dtype=np.float32
        )

    print(f"[v10] sparse matrix: {n_mat:,} x {n_env:,}, {matrix.nnz:,} nonzeros "
          f"({time.time()-t0:.1f}s)", flush=True)
    return matrix, mat_list, env_list


def _build_sparse_matrix_from_pool(pool, env_indices, t0):
    """build_sparse_matrix for a MatVecPool: assemble the CSR, don't rebuild it.

    Returns exactly what the dict path returns -- same matrix (same indptr,
    indices, data, and index dtypes), same mat_list (same values AND same
    element type), same env_list. Only the route differs.

    THE FOUR THINGS THAT MAKE IT EQUIVALENT
    1. mat_list. The dict path is `sorted(mat_vecs_dict.keys())`. The pool's
       mat_index is strictly ascending (checked by the caller), so row k already
       is the k-th smallest id; int() restores the dict's Python-int key type.
    2. Column mapping. The dict path does searchsorted(env_arr, all_env) and
       then VERIFIES the hit. When env_list is 0..n_env-1 -- which a fresh run
       always is, because demand_index is `ROW_NUMBER() OVER (...) - 1` -- the
       map is the identity by pigeonhole (sorted + strictly increasing + first 0
       + last n_env-1 forces env_arr == arange(n_env)), so searchsorted is a
       675.7M-element no-op and is skipped. The bounds test that `keep` performed
       is still applied. Any other env_list falls back to the real searchsorted.
       CORRECTION: a RESUME does NOT make the universe gappy. main() builds
       env_indices_filtered but never rebinds env_indices, so the selector always
       receives the full, dense 0..n_classes-1 universe and this fast path fires on
       resume too. (That is correct, not a bug: an already-covered class has its
       coverage restored from the restart file, so its capped target is met and it
       is never needy. The filtered array is used only for the resume load query.)
       The fallback is therefore currently unreachable from main().
    3. Duplicate (row, col) pairs. The dict path builds COO, whose constructor
       SUMS duplicates. The CSR triple constructor does not, so sum_duplicates()
       is called explicitly. There are none in production -- the load query does
       GROUP BY (mat_index, demand_index) -- but relying on that silently is how
       the two paths would diverge on any other input.
    4. Column order within a row. COO->CSR emits sorted indices; the triple
       constructor preserves insertion order. sum_duplicates() sorts first, so
       this is covered by 3.
    """
    env_list = sorted(env_indices)
    mat_list = [int(m) for m in pool.mat_index]
    n_mat, n_env = len(mat_list), len(env_list)

    if n_env == 0 or pool.nnz == 0:
        matrix = sparse.csr_matrix((n_mat, n_env), dtype=np.float32)
    else:
        env_arr = np.array(env_list, dtype=np.int64)
        cols = pool.idx_flat

        # Is the column map the identity? (see 2 above)
        identity = (int(env_arr[0]) == 0 and int(env_arr[-1]) == n_env - 1
                    and (env_arr.size < 2 or bool(np.all(np.diff(env_arr) > 0))))
        if identity:
            pos = cols
            keep = (cols >= 0) & (cols < n_env)
        else:
            pos = np.searchsorted(env_arr, cols)
            keep = env_arr.take(pos, mode='clip') == cols

        # scipy picks its index dtype from max(nnz, n_col); mirror that so the
        # returned matrix is byte-identical to the COO path, not merely equal.
        maxval = max(pool.nnz, n_env)
        idx_dtype = np.int32 if maxval <= np.iinfo(np.int32).max else np.int64

        if keep.all():
            indptr = pool.off.astype(idx_dtype, copy=False)
            indices = pos.astype(idx_dtype, copy=False)
            data = pool.cnt_flat.astype(np.float32, copy=False)
        else:
            # Dropping entries shortens rows, so indptr is rebuilt from the
            # per-row survivor counts rather than reused.
            lengths = np.diff(pool.off)
            row_of_nnz = np.repeat(np.arange(n_mat, dtype=np.int64), lengths)
            new_len = np.bincount(row_of_nnz[keep], minlength=n_mat)
            indptr = np.empty(n_mat + 1, dtype=idx_dtype)
            indptr[0] = 0
            np.cumsum(new_len, out=indptr[1:])
            indices = pos[keep].astype(idx_dtype, copy=False)
            data = pool.cnt_flat[keep].astype(np.float32, copy=False)

        matrix = sparse.csr_matrix((data, indices, indptr),
                                   shape=(n_mat, n_env), dtype=np.float32)
        # sort_indices() is implied by sum_duplicates(); see 3 and 4 above.
        matrix.sum_duplicates()

    print(f"[v10] sparse matrix: {n_mat:,} x {n_env:,}, {matrix.nnz:,} nonzeros "
          f"({time.time()-t0:.1f}s)", flush=True)
    return matrix, mat_list, env_list


@jit(nopython=True, fastmath=True, cache=True)
def _score_row(indptr, indices, data, row_idx, deficit, excess_ratio):
    """Raw (help, uniformity) for one material row -- the single source of truth
    for the scoring arithmetic, shared by the batched kernel and the per-pop
    kernel so the two can never drift. help = sum min(count, deficit);
    uniformity = -sum count*(1+excess_ratio) over envs already at/above target.
    Accumulates in float64; callers cast to float32 at their boundary."""
    row_start = indptr[row_idx]
    row_end = indptr[row_idx + 1]
    help_sum = 0.0
    uniformity_sum = 0.0
    for j in range(row_start, row_end):
        env_idx = indices[j]
        count = data[j]
        help_sum += min(count, deficit[env_idx])
        if excess_ratio[env_idx] >= 0:
            uniformity_sum -= count * (1.0 + excess_ratio[env_idx])
    return help_sum, uniformity_sum


@jit(nopython=True, fastmath=True, cache=True, parallel=True)
def score_materials_from_csr_data(indptr, indices, data, candidate_rows,
                                   deficit, excess_ratio):
    """
    Numba-optimized batched scoring directly on CSR sparse matrix internals.

    Args:
        indptr: CSR row pointers
        indices: CSR column indices
        data: CSR data values
        candidate_rows: which rows (materials) to score
        deficit: target_vector - coverage_vector (capped at 0)
        excess_ratio: (coverage - target_vector) / target_vector
    """
    n_candidates = len(candidate_rows)
    help_counts = np.zeros(n_candidates, dtype=np.float32)
    uniformity_scores = np.zeros(n_candidates, dtype=np.float32)

    for i in prange(n_candidates):
        help_sum, uniformity_sum = _score_row(
            indptr, indices, data, candidate_rows[i], deficit, excess_ratio)
        help_counts[i] = help_sum
        uniformity_scores[i] = uniformity_sum

    return help_counts, uniformity_scores


@jit(nopython=True, fastmath=True, cache=True)
def score_one_row(indptr, indices, data, m, deficit, excess_ratio):
    """Score a SINGLE material row (per-pop hot kernel of the lazy-greedy
    selection). Delegates the arithmetic to ``_score_row`` -- the same code the
    batch kernel runs -- and casts to float32 exactly as the batch kernel does
    when it stores into its float32 output arrays, so a re-scored material
    matches its seed score bit-for-bit when its coverage has not changed (keeps
    the heap's stored values a true upper bound on live scores).
    """
    help_sum, uniformity_sum = _score_row(indptr, indices, data, m,
                                          deficit, excess_ratio)
    return np.float32(help_sum), np.float32(uniformity_sum)




# ---------------------------------------------------------------------------
# v9 preprocessing reductions (see module notes / commit message)
# ---------------------------------------------------------------------------

def coverage_vector_to_counter(coverage_vector, env_list):
    """Rebuild the per-env coverage Counter from the dense coverage vector.

    v9 no longer maintains the Counter inside the hot loop (that per-iteration
    Python update scaled with the picked material's row length). The dense
    ``current_coverage_vector`` already holds all coverage information, so the
    Counter is reconstructed only when actually needed (restart snapshots and
    the final report). ``env_list`` maps dense index -> env_id.
    """
    nz = np.nonzero(coverage_vector)[0]
    # round(), not int(): coverage_vector is float32 and int() TRUNCATES. Counts
    # are integers and float32 is exact below 2**24, so this is a no-op today
    # bias every restart cycle downward once a class exceeds ~16.7M occurrences.
    return Counter({int(env_list[i]): int(round(float(coverage_vector[i])))
                    for i in nz})


def rebuild_exact_coverage_counter(selected, n_initial, mat_env_matrix,
                                   mat_to_idx_csr, env_list, initial_coverage):
    """Rebuild the EXACT float64 coverage Counter for the final report.

    Reproduces bit-for-bit what the original per-iteration Counter maintenance
    would have produced for the actually-selected subset: it sums the (small,
    integer) per-env counts of every material picked *this run* in float64
    (exact for integers < 2**53, unlike the float32 current_coverage_vector),
    added on top of any restart-provided initial_coverage. Materials selected
    in a previous run are already folded into initial_coverage and are absent
    from the loaded matrix, so only ``selected[n_initial:]`` are summed here --
    exactly as the original loop only added this run's picks to the loaded
    Counter.

    ``n_initial`` is passed in rather than re-derived as len(initial_selected):
    the caller is the only place that knows where the restart prefix ends, and
    inferring it from a second, caller-owned list is what produced the resume
    mis-report (the caller aliased the two lists, so the boundary silently became
    the end of the list and this run's picks were all excluded).
    """
    this_run = selected[n_initial:]
    counter = Counter()
    if this_run:
        idx = np.fromiter((mat_to_idx_csr[m] for m in this_run),
                          dtype=np.int64, count=len(this_run))
        # Accumulate in float64 directly (exact for integer counts < 2**53)
        # without upcasting the whole selected submatrix to float64 first.
        cov = np.asarray(mat_env_matrix[idx].sum(axis=0, dtype=np.float64)).ravel()
        nz = np.nonzero(cov)[0]
        counter = Counter({int(env_list[i]): cov[i] for i in nz})
    if initial_coverage:
        for k, v in initial_coverage.items():
            counter[k] += v
    return counter


def mat_vecs_from_arrow(tbl):
    """Build mat_vecs_dict from an Arrow table WITHOUT boxing a single element.

    Contract, identical to the fetchall() loop it replaces:
        {int mat_index: (int64 ndarray of demand_index, float64 ndarray of counts)}
    Same keys, same key TYPE (Python int), same values, same dtypes, same raw
    bytes -- the last one matters because collapse_identical_materials
    fingerprints materials with `arr.tobytes()`.

    WHY. The previous loader was

        for mat_index, indices, counts in conn.execute(...).fetchall():
            mat_vecs_dict[int(mat_index)] = (np.array(indices, dtype=np.int64),
                                             np.array(counts,  dtype=np.float64))

    and .fetchall() materialises the ENTIRE 64.6M-row result as Python objects
    before the loop starts: ~8.5e8 boxed integers inside ~1.3e8 Python lists.
    That is the same "hundreds of millions of short-lived small objects"
    allocation pattern that made build_sparse_matrix degenerate into kernel-side
    allocator churn (measured there: 100% system time, 0% user time, ~zero page
    faults). This phase shows the same signature: the identical work took 1,702s,
    4,813s and 9,932s across three runs, and the slow one spent 2h15m of its
    2h45m at exactly 1.00 core with RSS pinned at 123.4 GB.

    Arrow hands each LIST column over as a flat child buffer plus offsets, so the
    per-material arrays become numpy SLICES of two big buffers and the boxed
    integers never exist. ~10x fewer allocations, and ~35 GB instead of ~80-100.

    NOTE ON LIFETIME: the slices keep the two flat buffers alive, so dedup no
    longer frees the dropped materials' element data. Net memory is still far
    lower, but it is not strictly lower at every instant.

    SUPERSEDED on the uniform-numba path by mat_vec_pool_from_arrow, which stops
    one step earlier and keeps the buffers. Retained because it is the reference
    the pool is tested against, and because the dict remains the contract for
    every other search_algo.
    """
    mat_index, off_i, idx_flat, cnt_flat = _arrow_flat_buffers(tbl)
    return {int(mat_index[k]): (idx_flat[off_i[k]:off_i[k + 1]],
                                cnt_flat[off_i[k]:off_i[k + 1]])
            for k in range(len(mat_index))}


def _arrow_flat_buffers(tbl):
    """(mat_index, offsets, idx_flat, cnt_flat) from the loader's Arrow table.

    The shared front half of both loaders: null checks, offset-equality check
    and dtype normalisation. Extracted so mat_vecs_from_arrow and
    mat_vec_pool_from_arrow cannot validate the same table differently.
    """
    import pyarrow as pa

    mat_index = np.asarray(tbl['mat_index'])
    n_mat = len(mat_index)

    def _flat(name, dtype):
        """(offsets, values) of one LIST column as plain numpy, dtype-normalised."""
        col = tbl[name]
        if isinstance(col, pa.ChunkedArray):
            col = col.combine_chunks()
        if isinstance(col, pa.ChunkedArray):      # some pyarrow versions keep the wrapper
            col = col.chunk(0) if col.num_chunks else pa.array([], type=col.type)
        # A null list, or a null element inside one, would break the zero-copy
        # view and has no meaning here (every material has a list, every entry a
        # count). Fail loudly rather than produce a silently wrong matrix.
        if col.null_count or col.values.null_count:
            raise RuntimeError(
                f"[v10] NULLs in relevant_mat_vectors.{name} "
                f"({col.null_count} null lists, {col.values.null_count} null "
                f"elements). The loader cannot represent those.")
        # offsets are ABSOLUTE indices into .values, so this stays correct even
        # if the array is a slice of a larger one.
        return (np.asarray(col.offsets, dtype=np.int64),
                np.asarray(col.values).astype(dtype, copy=False))

    off_i, idx_flat = _flat('indices', np.int64)
    off_c, cnt_flat = _flat('counts', np.float64)
    # indices[k] and counts[k] MUST have the same length for every k, or the
    if not np.array_equal(off_i, off_c):
        raise RuntimeError(
            "[v10] indices and counts have different list lengths in "
            "relevant_mat_vectors; the (class, count) pairing would be wrong.")
    if n_mat and len(off_i) != n_mat + 1:
        raise RuntimeError(
            f"[v10] Arrow offset array has {len(off_i)} entries for {n_mat} "
            f"materials; expected {n_mat + 1}.")

    return mat_index, off_i, idx_flat, cnt_flat


def mat_vec_pool_from_arrow(tbl):
    """Same Arrow table as mat_vecs_from_arrow, kept as flat buffers.

    mat_vecs_from_arrow's final dict comprehension is the whole problem: it
    creates 42.8M dict entries, 42.8M tuples and 85.6M numpy slice objects, and
    build_sparse_matrix then walks all of them again. This returns the buffers
    the comprehension was about to shred. Everything above it -- the null check,
    the offset-equality check, the dtype normalisation -- is shared, so the two
    loaders cannot drift apart.

    The returned pool holds the SAME arrays the dict's slices would have viewed,
    so per-material data is byte-identical; see MatVecPool.
    """
    mat_index, off_i, idx_flat, cnt_flat = _arrow_flat_buffers(tbl)
    pool = MatVecPool(mat_index, off_i, idx_flat, cnt_flat)
    if not pool.is_ascending():
        # Every consumer's fast path depends on this. The loader query says
        raise RuntimeError(
            "[v10] relevant_mat_vectors did not arrive sorted by mat_index; "
            "the loader query must keep 'ORDER BY mat_index'.")
    return pool


def collapse_identical_materials(mat_vecs_dict, n_coverage):
    """Collapse materials with identical (env, count) fingerprints.

    Keeps at most ``min(copies, n_coverage)`` representatives per fingerprint
    and drops the rest.

    SAFETY (valid for every n_coverage, hence every min_freq_env):
      A material with per-env count c >= 1 satisfies env e by itself in
      ceil(target(e)/c) <= target(e) <= n_coverage copies. The j-th identical
      copy only helps env e while (j-1)*c < target(e); every copy beyond
      n_coverage therefore adds pure excess to *every* env it touches and can
      never appear in a sensible (non-wasteful) cover. So discarding copies
      past min(copies, n_coverage) removes only provably useless materials.
      Moreover the capped target_vector is preserved: for any env e the kept
      copies still supply >= min(n_coverage, achievable_original(e)) occurrences
      (n_coverage copies * count >= n_coverage when copies >= n_coverage, and
      all copies are kept otherwise), so min(n_coverage, achievable) is
      unchanged after collapsing.

    At n_coverage == 1 this is ordinary de-duplication (keep one). The
    min(copies, n_coverage) cap IS the safeguard that generalises it safely.
    Removed copies are interchangeable with a kept representative, so the final
    subset (built from kept representatives) still covers every environment.
    """
    if isinstance(mat_vecs_dict, MatVecPool) and mat_vecs_dict.is_ascending():
        return _collapse_identical_materials_pool(mat_vecs_dict, n_coverage)

    keep_per_group = int(n_coverage)
    groups = defaultdict(list)
    for mat_id, (env_ids, counts) in mat_vecs_dict.items():
        order = np.argsort(env_ids, kind='stable')
        # Fingerprint from raw bytes of the canonically-ordered arrays: equal
        # values (int64 env_ids, float64 counts) produce equal bytes, so this
        # groups identically to a tuple-of-ints key but skips per-element boxing.
        key = (env_ids[order].tobytes(), counts[order].tobytes())
        groups[key].append(mat_id)

    kept = {}
    n_dropped = 0
    for ids in groups.values():
        ids_sorted = sorted(ids)                       # deterministic pick
        keep = ids_sorted[:keep_per_group]             # slice already clamps to len
        for mid in keep:
            kept[mid] = mat_vecs_dict[mid]
        n_dropped += len(ids_sorted) - len(keep)
    return kept, n_dropped


def _collapse_identical_materials_pool(pool, n_coverage):
    """collapse_identical_materials for a MatVecPool. Same groups, same picks.

    The fingerprint is computed exactly as in the dict version -- stable argsort
    of the env ids, then raw bytes of both canonically-ordered arrays -- so
    grouping is identical by construction.

    THE PICK. The dict version keeps `sorted(ids)[:n_coverage]`. Here rows are
    appended to their group in ascending row order, and mat_index is strictly
    ascending (checked by the caller), so ascending row order IS ascending id
    order and the first n_coverage rows of a group ARE sorted(ids)[:n_coverage].

    Returns (new_pool, n_dropped) rather than (dict, n_dropped): the surviving
    rows are gathered by one vectorised mask (MatVecPool.select_rows) instead of
    rebuilding 42.8M dict entries.
    """
    keep_per_group = int(n_coverage)
    off, idx_flat, cnt_flat = pool.off, pool.idx_flat, pool.cnt_flat
    n = len(pool)

    groups = defaultdict(list)
    for k in range(n):
        a, b = off[k], off[k + 1]
        env_ids = idx_flat[a:b]
        order = np.argsort(env_ids, kind='stable')
        key = (env_ids[order].tobytes(), cnt_flat[a:b][order].tobytes())
        groups[key].append(k)

    keep_rows = []
    n_dropped = 0
    for rows in groups.values():
        keep_rows.extend(rows[:keep_per_group])       # slice already clamps to len
        n_dropped += len(rows) - min(len(rows), keep_per_group)

    # Restore row order. The dict version's output order was the insertion order
    # of `kept`, but its only consumer immediately does sorted(keys); building
    # the pool in ascending row order keeps mat_index ascending, which is the
    # invariant everything downstream relies on.
    keep_rows = np.sort(np.asarray(keep_rows, dtype=np.int64))
    return pool.select_rows(keep_rows), n_dropped


def forced_selection_cascade(mat_env_matrix, target_vector, coverage_vector, col_sums):
    """Iteratively select every 'forced' (mandatory) material.

    A not-yet-selected material m is FORCED when dropping it would make some
    environment infeasible, i.e. there is an env e in m with

        available(e) - count_m[e] < remaining_need(e)

    where available(e) = occurrences of e among not-yet-selected materials and
    remaining_need(e) = max(0, target(e) - coverage(e)). Equivalently, writing
    slack(e) = available(e) - remaining_need(e) >= 0, m is forced iff some env e
    in m has count_m[e] > slack(e).

    SAFETY (valid for every n_coverage / min_freq_env): forced materials belong
    to *every* feasible cover, therefore to every optimal cover. Selecting them
    up front can never enlarge the optimum and only shrinks the residual
    problem; feasibility is preserved (slack stays >= 0). Whole batches of
    currently-forced materials are mandatory simultaneously and order
    independent, so we select them round by round (vectorised) until no
    material is forced.

    ``col_sums`` are the CSR column sums (occurrences per env in the full loaded
    pool), computed once by the caller and shared here to avoid a second pass.

    Returns (selected_row_indices, updated_coverage_vector) in CSR row-index
    space. Only meaningful on a fresh start (coverage_vector all zeros); the
    caller skips it on restart.
    """
    csr = mat_env_matrix
    n_mat = csr.shape[0]
    indptr = csr.indptr
    indices = csr.indices
    data = csr.data
    # Work in float64 for exact integer arithmetic on (large) column sums.
    coverage = coverage_vector.astype(np.float64).copy()
    target = target_vector.astype(np.float64)
    # available(e) = occurrences among not-yet-selected materials; starts at the
    # full column sums (fresh start => nothing selected yet). astype() copies, so
    # the in-place -= below never mutates the caller's col_sums.
    available = col_sums.astype(np.float64)

    # Map each stored nonzero to its material row (robust to any empty rows).
    row_of_nnz = np.repeat(np.arange(n_mat, dtype=np.int64), np.diff(indptr))

    is_selected = np.zeros(n_mat, dtype=bool)
    while True:
        remaining_need = np.maximum(0.0, target - coverage)
        slack = available - remaining_need
        # A (material, env) incidence forces the material if count > slack(env).
        edge_forced = data > slack[indices]
        force_count = np.bincount(row_of_nnz[edge_forced], minlength=n_mat)
        newly_forced = (force_count > 0) & (~is_selected)
        forced_idx = np.nonzero(newly_forced)[0]
        if forced_idx.size == 0:
            break
        contrib = np.asarray(csr[forced_idx].sum(axis=0)).ravel().astype(np.float64)
        coverage += contrib
        available -= contrib
        is_selected[forced_idx] = True

    return np.nonzero(is_selected)[0], coverage.astype(np.float32)


def prune_redundant_materials(mat_env_matrix, selected_rows, target_vector,
                              mat_site_counts):
    """Drop selected materials whose every class is already covered without them.

    WHY THIS EXISTS. The greedy adds materials one at a time and never revisits a
    pick. A material taken early because it was the only route to one or two needy
    classes can later become entirely unnecessary, because materials chosen
    AFTERWARDS happen to contain those same classes as surplus. Nothing in the
    selection notices. Measured on the T=220 production subset: 119,128 of 901,452
    materials (13.22%) and 790,717 of 9,928,334 atoms (7.96%).

    THIS IS SPECIFICALLY A COMPANION TO cost_alpha > 0, NOT A GENERAL WIN. At
    cost_alpha=0 the greedy takes large materials covering many classes at once,
    and for such a material to become useless EVERY one of those classes must
    later be covered elsewhere -- measured on the two fixtures: 0.32% and 0.00% of
    atoms. At cost_alpha=1 it takes small materials bought for one or two classes,
    and one or two classes are easily subsumed: 4.54% and 7.08%. The production
    breakdown says the same thing by cell size -- 19.4% redundancy in the
    7.4-atom sub-dataset, 0.05% in the 67-atom one, monotone in between. Read this
    as recovering part of the fragmentation cost cost_alpha=1 pays, not as an
    independent improvement to the greedy.

    THE TEST. Material m is droppable iff for EVERY class e it touches, the other
    surviving materials still supply at least required(e). `target_vector` is that
    requirement, min(n_coverage, achievable) -- the same contract the selector
    enforced -- so a dropped material is one whose absence nobody can detect.

    Two properties fall out with no special-casing:
      * A material that is the ONLY supplier of some class automatically fails the
        test (cov - count < required there), so forced materials are never dropped.
      * When enable_floors=True the floor classes ARE demand classes and so are in
        `target_vector`, protected here like any other. Under the default
        enable_floor_topup route the floors are repaired after selection, and the
        caller runs this BEFORE that -- which is why this function needs no floor
        logic at all. Reversing that order would force it to re-derive the floor
        signatures, a database pass.

    CALLED FROM main(), not from the selector: see the comment at the call site for
    why (the selector stays byte-identical to v13, and a resumed run gets pruned
    exactly like a fresh one).

    ORDER. Largest material first, ties broken by CSR row (= ascending mat_index,
    the greedy's own tie-break), so the result is deterministic. Measured against
    reverse-selection, selection, smallest-first and random order on both fixtures:
    all five drop essentially the SAME NUMBER of materials (the redundancy is
    structural, not an artifact of the order), but largest-first recovers the most
    atoms -- 4.54%/7.08% against 4.44%/6.44% for smallest-first. Choosing the truly
    optimal set to remove is NP-hard; this is a greedy heuristic on the same
    objective, never worse than the alternatives tried, and free.

    Returns (keep_mask, n_dropped, atoms_dropped, sweeps); keep_mask is aligned to
    `selected_rows`.
    """
    rows = np.asarray(selected_rows, dtype=np.int64)
    n = int(rows.size)
    keep = np.ones(n, dtype=bool)
    if n == 0:
        return keep, 0, 0, 0

    # float64 throughout. current_coverage_vector is float32 and inexact above
    # 2**24; that cannot actually flip a comparison here (the test only bites
    # where cov is CLOSE to req, and req <= n_coverage is small and exact), but
    # the cast costs nothing and removes the need for the argument.
    sub = mat_env_matrix[rows].astype(np.float64)
    indptr, indices, data = sub.indptr, sub.indices, sub.data
    req = np.asarray(target_vector, dtype=np.float64)
    cov = np.asarray(sub.sum(axis=0)).ravel()
    atoms = np.asarray(mat_site_counts, dtype=np.int64)[rows]

    order = np.lexsort((rows, -atoms))          # atoms DESC, then mat_index ASC
    starts = indptr[:-1]
    nonempty = np.diff(indptr) > 0

    n_dropped, atoms_dropped, sweeps = 0, 0, 0
    while True:
        sweeps += 1
        # EXACT candidate set at the start of this sweep: m is droppable iff
        # min over its classes of (cov - req - count) >= 0. One vectorised pass
        # via reduceat over the non-empty row starts -- empty rows have start ==
        # end, so the segment of each non-empty row still ends where it should.
        # Without this the sequential walk would test every selected material
        # every sweep; with it, only the ~1-in-8 that can possibly be dropped.
        margin = (cov - req)[indices] - data
        rowmin = np.full(n, np.inf)             # an empty row touches no class
        if nonempty.any():
            rowmin[nonempty] = np.minimum.reduceat(margin, starts[nonempty])
        cand = (rowmin >= 0.0) & keep
        # Walk ONLY the candidates, in the global order. Iterating `order` and
        # skipping non-candidates would still be n Python steps per sweep; at
        # 901k materials that is the difference between ~1 s and ~0.1 s.
        cand_order = order[cand[order]]

        changed = 0
        for k in cand_order:
            if not keep[k]:
                continue
            a, b = indptr[k], indptr[k + 1]
            e, c = indices[a:b], data[a:b]
            # RE-CHECK: drops earlier in this same sweep have lowered cov, so the
            # start-of-sweep candidacy is necessary but no longer sufficient.
            if not np.all(cov[e] - c >= req[e]):
                continue
            cov[e] -= c
            keep[k] = False
            n_dropped += 1
            atoms_dropped += int(atoms[k])
            changed += 1
        # A material can only BECOME droppable through another material's drop,
        # which is caught by the next sweep's fresh candidate set. Terminates
        # because every sweep but the last removes at least one material.
        if changed == 0:
            break
    return keep, n_dropped, atoms_dropped, sweeps


def parallel_uniform_n_coverage(
        env_indices: set[int],
        mat_vecs_dict: tuple[np.ndarray, np.ndarray], 
        n_coverage=10, 
        n_proc=1, 
        explore_first=False,
        uniformity_weight=0.0,
        cost_alpha=0.0,
        batch_size=1,
        #for restart
        restart_interval=None,
        restart_path="restart.npz",
        initial_selected=None,
        initial_coverage=None,
        restart_config=None,
        #v9 preprocessing reductions (safe; skipped on restart)
        enable_dedup=True,
        enable_forced_selection=True
        ):

    print(f"Covering {len(env_indices)} env_ids with n_coverage={n_coverage}")
    print(f'Using uniformity weight of {uniformity_weight}')
    print(f"Using {n_proc} processes for parallel scoring")
    # v11: cost_alpha == 0.0 is the v10 rule exactly (cost_pow is all-ones and
    # IEEE division by 1.0 is exact, so scores are bit-identical to v10's).
    print(f"[v11] cost_alpha = {cost_alpha} "
          f"({'v10-equivalent cardinality greedy' if cost_alpha == 0.0 else 'per-atom cost-benefit greedy'})",
          flush=True)
    if cost_alpha < 0.0:
        raise ValueError(
            f"cost_alpha must be >= 0 (0 = v10 cardinality greedy, 1 = weighted "
            f"set-cover greedy); got {cost_alpha}. A negative exponent would "
            f"reward large materials, which is the bias this knob exists to remove.")

    # The lazy-greedy (hash) selection is exact only for single-material picks:
    # it takes the argmax, then updates coverage before the next pick. Batch
    if batch_size != 1:
        raise ValueError(
            f"this implementation supports only batch_size=1 "
            f"(exact lazy-greedy); got batch_size={batch_size}.")

    # Both preprocessing reductions apply only on a fresh run; on restart the
    # loaded pool/coverage must stay consistent with the saved selection.
    # `is None` and not falsiness: a restart file can legitimately carry an
    # EMPTY selection with non-empty coverage, and that is still a resume -- the
    # loaded pool was already filtered by the resume query, so re-running the
    # fresh-run reductions on it would be inconsistent.
    fresh_run = initial_selected is None

    # v9 reduction 1: collapse identical materials (keep min(copies, n_coverage)).
    # Safe for every n_coverage (see collapse_identical_materials).
    if enable_dedup and fresh_run:
        n_before = len(mat_vecs_dict)
        mat_vecs_dict, n_dropped = collapse_identical_materials(mat_vecs_dict, n_coverage)
        print(f"[DEDUP] Identical-material collapse: {n_before} -> {len(mat_vecs_dict)} "
              f"materials ({n_dropped} redundant copies removed; "
              f"keep <= min(copies, n_coverage={n_coverage}))", flush=True)

    # Initialize tracking. (The per-env coverage Counter is no longer kept here;
    # it is rebuilt on demand for restart snapshots and the final report.)
    #
    # SAME list object as the caller's initial_selected, so every append below grew
    # BOTH. rebuild_exact_coverage_counter then computed
    # start = len(initial_selected) == len(selected), making this_run empty, and
    # the '=== Selection Complete ===' block of every RESUMED run reported only the
    # restart file's coverage -- none of the materials picked after the resume.
    # A resumed run whose subset was byte-identical to the uninterrupted one
    # still reported the restart file's coverage rather than the final one. The
    # subset itself was never affected (the DuckDB verification
    # block below is independent and was always correct), but the run looked
    # broken. Copying decouples the two lists.
    selected = list(initial_selected) if initial_selected else []
    # Where the restart prefix ends, captured HERE, at the one point that knows it.
    # Passed to rebuild_exact_coverage_counter instead of letting it re-derive the
    # boundary from initial_selected -- which is what silently broke when the two
    # names aliased the same list.
    n_initial = len(selected)
    remaining_mats = set(mat_vecs_dict.keys())

    #Construct current_coverage_vector
    mat_env_matrix, mat_list, env_list = build_sparse_matrix(mat_vecs_dict, list(env_indices))
    # Convert data to float32 ONCE
    if mat_env_matrix.data.dtype != np.float32:
        print("Converting sparse matrix to float32...")
        mat_env_matrix = mat_env_matrix.astype(np.float32)
        print(f"Converted. Matrix data dtype: {mat_env_matrix.data.dtype}")

    # ---- v11 cost model (Phase 0 tie-break + Phase 1 cost_alpha) -------------
    # Row sum = site observations this material contributes inside the demand
    # universe = its atom count on the production configuration (see the module
    # header for when that identity can fail). Taken from the matrix rather than
    # from a separate natoms lookup so the cost can never drift out of sync with
    # the coverage vectors it is divided into.
    #
    # int64, not the matrix's float32: this array is the middle key of every heap
    # tuple, and tuple comparison against a Python int must be exact and total.
    # Counts are small (<= a few hundred) so the cast is lossless.
    mat_site_counts = np.asarray(
            mat_env_matrix.sum(axis=1)).ravel().round().astype(np.int64)
    # Precomputed ONCE, so both scoring kernels divide by the bit-identical
    # float32 value; recomputing the power per call would risk the two paths
    # disagreeing in the last ulp and breaking the heap's upper-bound property.
    # maximum(...,1) guards empty rows: they have help == 0 and are dropped by
    # the `keep` filter below, but 0 ** alpha would still make an inf/nan here.
    cost_pow = np.power(np.maximum(mat_site_counts, 1).astype(np.float64),
                        np.float64(cost_alpha)).astype(np.float32)

    env_list = np.array(env_list, dtype=np.int64)
    env_to_idx = {env_id: idx for idx, env_id in enumerate(env_list)}
    mat_to_idx_csr = {mat_id: idx for idx, mat_id in enumerate(mat_list)}
    current_coverage_vector = np.zeros(len(env_list), dtype=np.float32)
    if initial_coverage:
        for env_id, count in initial_coverage.items():
            if env_id in env_to_idx:  # Only for envs still needing coverage
                current_coverage_vector[env_to_idx[env_id]] = count

    # Per-env coverage target, capped at what is actually achievable.
    # achievable = occurrences available in the loaded pool (CSR column sums)
    #              + already-covered occurrences (non-zero only on restart).
    # Capping lets the loop terminate correctly when n_coverage exceeds the
    # number of times some env can ever occur (e.g. n_coverage > min_freq_env).
    # Column sums (occurrences per env in the loaded pool), computed once and
    # reused by the forced-selection cascade below.
    col_sums = np.asarray(mat_env_matrix.sum(axis=0)).ravel().astype(np.float32)
    achievable = col_sums + current_coverage_vector
    target_vector = np.minimum(np.float32(n_coverage), achievable)
    n_capped = int(np.count_nonzero(achievable < n_coverage))
    if n_capped > 0:
        print(f"{n_capped} of {len(target_vector)} env_ids have fewer than "
              f"n_coverage={n_coverage} occurrences available; their target is "
              f"capped at their total available count.", flush=True)

    # NOTE (hash version): the original env_id -> {mat_ids} inverted index is no
    # longer built. The lazy-greedy selection below never needs "which materials
    # share a newly-covered env" (it defers rescoring until a material reaches
    # the top of the heap), so building that large dict-of-sets would be pure
    # wasted time and memory at scale.

    # v9 reduction 2: forced-selection cascade. Pre-select every mandatory
    # material (one that is the only way to keep some env feasible). These
    # belong to every feasible cover, so selecting them first is safe for any
    # n_coverage and only shrinks the residual problem. Skipped on restart.
    if enable_forced_selection and fresh_run:
        t_forced = time.time()
        forced_rows, current_coverage_vector = forced_selection_cascade(
                mat_env_matrix, target_vector, current_coverage_vector, col_sums)
        if forced_rows.size:
            forced_mat_ids = [mat_list[int(r)] for r in forced_rows]
            for mid in forced_mat_ids:
                selected.append(mid)
                remaining_mats.discard(mid)
        print(f"[FORCED] Pre-selected {forced_rows.size} mandatory materials "
              f"in {time.time() - t_forced:.3f}s", flush=True)

    pool = None
    if n_proc > 1:
        print(f"Creating persistent pool with {n_proc} workers...")
        import sys
        sys.stdout.flush()
        #pool = Pool(processes=n_proc, 
        print("Pool created successfully!")
        sys.stdout.flush()

    try:
        # ==================================================================
        # Scalable lazy-greedy selection (CELF / Minoux 1978) -- hash version
        # ------------------------------------------------------------------
        # Replaces the O(N) masked-argmax scan + eager affected-material
        # rescoring of the v9 core. A material's marginal score is monotonically
        # NON-INCREASING as coverage grows (per-env deficits only shrink; the
        # uniformity penalty only grows), so re-scoring a material ONLY when it
        # reaches the top of a max-heap reproduces EXACTLY the greedy rule (take
        # the highest-scoring material each step) -- without ever scanning all N
        # materials, and without rescoring materials that never approach the top.
        #
        # The heap stores (-score, row). A stale top (its true score has since
        # dropped) is re-scored and reinserted at most once each time it
        # surfaces (reinsert-once); the heap therefore never fills with obsolete
        # duplicates and the per-selection cost does NOT grow toward the end of
        # the run -- the failure mode of a naive push-on-every-update heap.
        #
        # Per-env deficit / excess_ratio drive the score and are maintained
        # incrementally: only the picked material's envs change each step.
        deficit = np.maximum(0.0, target_vector - current_coverage_vector).astype(np.float32)
        # Denominator guarded: target_vector is >= 1 for every reachable class
        # >= 1), but that argument is indirect and a 0 would produce nan/inf
        # here and silently corrupt the uniformity term. max(target,1) is a
        # no-op whenever target >= 1.
        _tgt_safe = np.maximum(target_vector, np.float32(1.0))
        excess_ratio = ((current_coverage_vector - target_vector) / _tgt_safe).astype(np.float32)

        indptr = mat_env_matrix.indptr
        indices = mat_env_matrix.indices
        data = mat_env_matrix.data

        # Envs still below target; drives termination.
        n_needy = int((current_coverage_vector < target_vector).sum())

        # Candidate rows = every still-available material (forced-selected ones
        # were already removed from remaining_mats above).
        candidate_rows = np.fromiter(
                (mat_to_idx_csr[m] for m in remaining_mats),
                dtype=np.int64, count=len(remaining_mats))

        # One batched pass for the initial scores, then seed the heap with only
        # the materials that actually help some needy env (help > 0). Materials
        # with help == 0 can never help later (scores only decrease), so they
        # are dropped up front.
        print(f"Scoring {candidate_rows.size} candidate materials to seed the heap...", flush=True)
        t_seed = time.time()
        help0, uni0 = score_materials_from_csr_data(
                indptr, indices, data, candidate_rows, deficit, excess_ratio)
        # v11: the cost division is written the SAME way here and in the per-pop
        # path below (same operand order, same float32 operands), so a re-scored
        # material still matches its seed score bit-for-bit when its coverage has
        # not changed -- which is what keeps the heap's stored values a true upper
        # bound. At cost_alpha == 0 every divisor is exactly 1.0f and the division
        # is the identity, so these scores equal v10's bit-for-bit.
        score0 = (help0 + uniformity_weight * uni0) / cost_pow[candidate_rows]
        removed = np.zeros(mat_env_matrix.shape[0], dtype=bool)
        # Bulk-convert to Python floats/ints in C (.tolist) rather than boxing
        # each of N elements in a Python loop. Heap contents after heapify are
        # identical (pre-heapify order is irrelevant).
        keep = help0 > 0.0
        kept_rows = candidate_rows[keep]
        heap = list(zip((-score0[keep]).tolist(),
                        mat_site_counts[kept_rows].tolist(),
                        kept_rows.tolist()))
        heapq.heapify(heap)
        print(f"Heap seeded with {len(heap)} materials in {time.time() - t_seed:.3f}s "
              f"({n_needy} env_ids under target)", flush=True)

        iteration = 0
        while n_needy > 0:
            time_iter_start = time.time()
            iteration += 1

            # ---- lazy pop: return the true highest-score material that still
            # helps a needy env, re-scoring stale tops on demand ----
            best_mat = -1
            best_score = 0.0
            while heap:
                _, _, m = heapq.heappop(heap)
                if removed[m]:
                    continue
                help_m, uni_m = score_one_row(indptr, indices, data, m,
                                              deficit, excess_ratio)
                if help_m <= 0.0:
                    # No needy env left in this material -> permanently useless.
                    removed[m] = True
                    continue
                # Same expression as the batched seed above -- see the comment there.
                score_m = (help_m + uniformity_weight * uni_m) / cost_pow[m]

                # ---- v11 acceptance rule: FULL KEY, not score alone ----------
                # v10 accepted on `score_m >= -heap[0][0]`, i.e. as soon as the
                # rescored score tied the best bound left on the heap. That is
                # enough to guarantee m is *an* argmax, but not enough to
                # guarantee it is the argmax the ordering key names: an entry
                # carrying a STALE (too high) stored score is popped ahead of
                # accurately-stored ones, and on rescoring down to a tying value
                # it was accepted on the spot, never reinserted to compete on
                # site_count. Which of several equal-scoring materials won then
                # depended on the staleness landscape -- deterministic, but
                # history-dependent, so a run resumed from restart.npz could
                # legitimately return a different (equally valid) subset from an
                # uninterrupted one. The full-key tie-break removes that.
                #
                # WHY THE COMPARISON IS SOUND. Stored score >= live score for
                # every heap entry, so stored_key <= true_key entrywise (the
                # negated score dominates the tuple, and site_count/row are exact
                # rather than bounds). Accepting only when
                # therefore proves m beats the best any remaining entry could
                # possibly achieve. Conservative in the safe direction: it may
                # reject an m that would in fact have been fine, costing one
                # extra pop, but it can never accept a wrong one.
                #
                # WHY IT TERMINATES. Coverage is frozen for the duration of this
                # inner loop, so a rescored entry's key is stable for the rest of
                # the step. Each pop either accepts, marks the row removed
                # (help <= 0), or replaces a stale key with its true one. After
                # at most one rescore per entry every stored key is true, the top
                # of the heap is then the true minimum, and the test passes. The
                # loop is thus bounded by the heap size per selection.
                #
                # The `not heap` branch is the empty-heap case: with nothing left
                # to compare against, m is trivially the best remaining choice.
                key_m = (-score_m, int(mat_site_counts[m]), m)
                if not heap or key_m <= heap[0]:
                    best_mat = m
                    best_score = score_m
                    break
                # Either stale (score has dropped) or outranked on the tie-break
                # by an entry still on the heap: reinsert with the TRUE key and
                # keep popping. Reinserting the true key is what makes the loop
                # monotone and therefore finite.
                heapq.heappush(heap, key_m)

            if best_mat < 0:
                print("Warning: Cannot satisfy all env_ids with remaining materials")
                break

            # ---- commit the pick (the `removed` bool array is the loop's
            # source of truth for availability; remaining_mats was only needed
            # to seed the heap) ----
            best_mat_id = mat_list[best_mat]
            selected.append(best_mat_id)
            removed[best_mat] = True

            # ---- incremental coverage / deficit / excess_ratio update, for the
            # picked row's envs only (everything else is left stale on purpose;
            # the lazy pop above corrects staleness when it matters) ----
            r0 = indptr[best_mat]
            r1 = indptr[best_mat + 1]
            cols = indices[r0:r1]
            cnts = data[r0:r1]
            old_cov = current_coverage_vector[cols]   # advanced indexing already copies
            new_cov = old_cov + cnts
            current_coverage_vector[cols] = new_cov
            tgt = target_vector[cols]
            deficit[cols] = np.maximum(0.0, tgt - new_cov)
            excess_ratio[cols] = (new_cov - tgt) / np.maximum(tgt, np.float32(1.0))
            n_needy -= int(((old_cov < tgt) & (new_cov >= tgt)).sum())

            # v11: `sites` appended, not substituted -- the existing
            # "with score [...]" prefix stays parseable by anything that already
            # scrapes this log. It is the per-pick atom cost, which is the whole
            # quantity cost_alpha exists to control, so it belongs in the trace.
            print(f"Selected best material [{best_mat_id}] with score "
                  f"[{best_score}] sites [{int(mat_site_counts[best_mat])}]", flush=True)
            print(f"Iteration {iteration}: {len(selected)} materials selected, "
                  f"{n_needy} env_ids still under target")

            if restart_interval and iteration % restart_interval == 0:
                save_restart_file(
                        restart_path,
                        iteration,
                        selected,
                        coverage_vector_to_counter(current_coverage_vector, env_list),
                        config=restart_config
                        )

            print(f'Total iteration time : {time.time() - time_iter_start}', flush=True)

        if n_needy == 0:
            print("All env_ids satisfied!")

    finally:
        # Clean up pool when done
        if pool is not None:
            print("Closing pool...")
            pool.close()
            pool.join()
            print("Pool closed")

    # Report final statistics
    print(f"\n=== Selection Complete ===")
    print(f"Selected {len(selected)} materials")

    # v9: rebuild the EXACT float64 coverage Counter for the reporting below
    # bit-for-bit what in-loop maintenance would have produced for the selected
    # subset, so the printed report is unaffected by dropping the in-loop update.
    current_coverage = rebuild_exact_coverage_counter(
            selected, n_initial, mat_env_matrix,
            mat_to_idx_csr, env_list, initial_coverage)

    covered = sum(1 for env in env_indices if current_coverage[env] >= n_coverage)
    print(f"Coverage: {covered}/{len(env_indices)} env_ids have >= {n_coverage} occurrences")
    target_met = sum(1 for env in env_indices
                     if current_coverage[env] >= target_vector[env_to_idx[env]])
    print(f"Target met (capped at availability) for {target_met}/{len(env_indices)} env_ids")

    if current_coverage:
        counts = [current_coverage[env] for env in env_indices]
        print(f"Coverage distribution:")
        print(f"  Min: {min(counts)}, Max: {max(counts)}, Mean: {sum(counts)/len(counts):.1f}")
        print(f"  Std: {np.std(counts):.1f}")

    return selected#, current_coverage

# =====================================================================
# ITERATIVE COARSENING (per-token backoff)
# =====================================================================
#
# WHY
# ---
# A token that occurs at exactly one site occurs in exactly one material, so that
# material has to be selected: there is no choice to make. Singletons are common
# enough that covering tokens directly would force a large fraction of the corpus
# into the subset.
#
# The fix is not to pick rare tokens more cleverly, but to make sure no coverage
# target is ever a singleton. Each token is mapped to the first rung of the
# coarsening ladder whose class has support >= T. Nothing is discarded; rare
# tokens are covered as members of a coarser class.
#
# Required: T >= n_coverage + 1.
#
# WHAT support_metric CHANGES
# ---------------------------
# 'sites' (cheap) counts site observations. A class can then clear T with all of
# its sites inside a single material, which is still forced -- unavoidably so,
# since that material really is the only source of the class.
#
# 'materials' counts distinct materials instead. Then support >= T >= n_coverage+1
# genuinely implies at least T independent suppliers, so no material is forced.
# It costs a pass over the material vectors per rung with an exact COUNT(DISTINCT)
# per class. Use it when the guarantee matters more than build time.
#
# INVARIANTS THE LADDER RELIES ON
# -------------------------------
# * No rung ever drops the central element, including the per-element catch-all.
# * The coordination number survives every rung except R5 and R8, which drop it
#   on purpose.
# * poly_reliable is applied as a RUNG, not as preprocessing: a polyhedron label
#   is discarded only for tokens too rare to support their own target. Frequent
#   tokens keep their geometry whatever their symmetry measure says.
#
# TOKEN GRAMMAR
# -------------
#   <elem>_<spin>_<nb1-nb2-...>_<polyhedron>_<f5>_<dmin>_<a_bin>
#
# The polyhedron MAY contain '_' (e.g. 'BS_2:10'); the neighbour list never does.
# So the field count is NOT constant, and every trailing field must be indexed
# from the END. With p = str_split(env_id, '_'):
#
#   p[1]=elem  p[2]=spin  p[3]=nb  p[-3]=f5  p[-2]=dmin  p[-1]=a_bin
#   array_to_string(p[4:-4],'_') = poly
#
# (DuckDB list slices are 1-indexed and inclusive at both ends, with -1 = last.)
#
# Legal values, which is what the format guard keys on:
#   dmin  (p[-2])  0..4  or 999      -- bin edges 0.60, 0.70, 0.85, 1.15
#   a_bin (p[-1])  0..2  or 999      -- bin edges 1.0, 3.0
#
# Degenerate tokens from the tokeniser's fallback paths parse correctly too, e.g.
# 'X_0_INVALID_INVALID_999_999_999' and 'X_0_Cl-Cl_UNKNOWN:2_999_2_2'.
#
# f5 is the binned continuous symmetry measure:
#   0: csm <= 1   1: 1 < csm <= 5   2: 5 < csm < 10   3: csm >= 10
#   999: csm undefined, a CrystalNN rescue, or INVALID
# f5 >= 2 means the nominated polyhedron is the best of several poor fits and is
# not a trustworthy label.
#
# dmin is the shortest contact of the shell divided by the sum of the two Cordero
# covalent radii, so its bins are positions on the pair dissociation curve rather
# than quantiles of this dataset:
#   0 repulsive wall | 1 compressed | 2 near equilibrium | 3 stretched | 4 ...
#
# dmin and a_bin are the only absolute length scale in the token. Everything else
# is scale-invariant, because chemenv selects neighbours with a cutoff relative to
# the nearest contact and evaluates the symmetry measure after rescaling.
#
# 999 (no usable contact, or no tabulated covalent radius) is set for both fields
# together by every exit of the tokeniser, so there is no half-undefined token and
# a rung that keeps only dmin still isolates the undefined population.


# Discard the polyhedron label when the CSM bin says it is untrustworthy.
POLY_RELIABLE = ("CASE WHEN TRY_CAST(f5 AS INTEGER) IS NULL "
                 "          OR TRY_CAST(f5 AS INTEGER) >= 2 THEN 'UNCL' "
                 "     ELSE poly END")

# Bin each neighbour element's multiplicity instead of deleting it outright.
# 'Cl-Cl-Cl-Cl-Cl-Br' -> 'Br:1-Cl:56'; 'Cl-Cl-Cl-Br-Br-Br' -> 'Br:34-Cl:34'
# so the 5:1 and 3:3 shells stay distinguishable one rung longer than at R3.
NB_BINNED = """array_to_string(list_sort(list_transform(
        list_distinct(str_split(nb, '-')),
        e -> e || ':' || CASE
                 WHEN len(list_filter(str_split(nb, '-'), x -> x = e)) = 1 THEN '1'
                 WHEN len(list_filter(str_split(nb, '-'), x -> x = e)) = 2 THEN '2'
                 WHEN len(list_filter(str_split(nb, '-'), x -> x = e)) <= 4 THEN '34'
                 WHEN len(list_filter(str_split(nb, '-'), x -> x = e)) <= 6 THEN '56'
                 ELSE '7p' END)), '-')"""

# Coordination number AS IT APPEARS IN A SIGNATURE. `cn` is computed as
# len(str_split(nb, '-')), which is the true coordination number only when `nb` is
# a real element list. It is NOT, in exactly one case:
# has three exits, and the third -- chemenv failed AND the CrystalNN rescue failed
# -- returns ('INVALID', 'INVALID', nan, 999, nan, nan, 999, 999). `nb` is then
# the literal 'INVALID',
# whose split length is 1, so a site with NO DETERMINED GEOMETRY was indistinguishable
# from a genuine 1-coordinate site at every rung that carries cn (R2b, R3, R4, R6,
# R7) and at the F_elem_cn floor. The other two exits are fine
# -- the CrystalNN rescue keeps `nb` intact and emits UNKNOWN:<cn> in the POLY field
# (307,786 such tokens in that subset, cn correct), and it guarantees len(neigh) >= 1.
#
# '' is included for the one unguarded residue of the first exit: the guard there is
# `if neighbors_info:` (the site has >= 1 neighbour SET), not that the set has >= 1
# neighbour, so an empty neighb_sites would join to the empty string. Such a token
# has a non-empty poly, so _build_parsed_token_table's parse check does NOT catch it.
# `nb IS NULL` is included because CAST(NULL AS VARCHAR) is NULL, and a NULL
# signature is silently dropped by both `JOIN ... USING (sig)` branches of the
# resolve step -- i.e. the token would leave the ladder entirely rather than fall
# through. (That hazard is narrowed here, not closed: R1/R2/R2a concatenate `nb`
# itself. It is currently unreachable, because every token shape that yields a NULL
# nb also yields an empty poly, which the parse check rejects.)
#
# A STRING sentinel, never NULL, for the reason just given.
#
# KNOWN, ACCEPTED CONSEQUENCE. Under design 'a' a rung's support is summed over the
# still-unresolved pool, so the placeholder tokens used to contribute to elem|1's
# support. Separating them removes that contribution, and any elem|cn class that was
# only supported thanks to them now falls below T -- so GENUINE 1-coordinate tokens
# in it coarsen one rung further. That is not avoidable while also separating the
# placeholders (it is the same support total), and it is the honest outcome: the
# class really does not have T sites of that coordination.
CN_SIG = ("CASE WHEN nb IS NULL OR nb IN ('INVALID', '') THEN 'NA' "
          "ELSE CAST(cn AS VARCHAR) END")

# Ordered finest-to-coarsest, but NOT a strict refinement chain. Every rung is a
# total function of the token, so a token always lies inside its own class at
# every rung. That is all the algorithm needs: a token resolves at the first rung
# whose class reaches support T, and otherwise falls through to the next.
#
# Two rungs deliberately break the nesting:
#
#  * R2 -> R2a SPLITS rather than coarsens. poly_reliable depends on f5, which R1
#    already discarded, so R2a reintroduces an f5-derived distinction. For f5 <= 1
#    the R2a class is a strict subset of the R2 class, so it can never reach
#    support where R2 did not, and the rung is simply inert for those tokens. For
#    the rest it does its job, merging across all polyhedra.
#
#  * R6 and R7 are sibling fallbacks after R5, not coarsenings of it: they
#    re-expose the geometry R5 dropped. They ARE coarsenings of R3.
#
# So support does not decrease monotonically down the ladder. Do not assume it
# does.
#
# NOTE ON CN: poly_reliable collapses to 'UNCL' for f5 >= 2, and 'UNCL' carries no
# ':N' suffix, so it does not imply a coordination number. nbset drops
# multiplicities and nb_binned only bins them. Every rung from R2b onwards
# therefore appends `cn` explicitly, so the coordination number survives
# everywhere except at the two rungs that drop it on purpose (R5, R8). Without
# that, a 4-coordinate and a 6-coordinate site with the same element and the same
# neighbour set would merge whenever both had an unreliable polyhedron.
#
# The finest rung is the full token. It is not an option; it is what the modifier
# rule emits at the finest identity level. V13A_FULL_TOKEN=0 removes it as an
# ablation.

R0_FULL_RUNG = (
    "R0_full",
    "elem||'|'||spin||'|'||nb||'|'||poly||'|'||f5||'|'||dmin||'|'||a_bin")

# ===========================================================================
# THE MODIFIER RULE -- what generates the ladder
# ===========================================================================
# The token's seven fields split into two kinds:
#
#   IDENTITY   elem, spin, nb, poly, f5      what the environment IS
#   MODIFIER   dmin, a_bin                   how DEFORMED it is
#
# The test that separates them: a field is a MODIFIER if it is defined
# independently of the polyhedron assignment. dmin and a_bin are per-atom
# geometric quantities computed from the neighbour vectors alone, and mean the
# same thing whether or not chemenv identified a geometry. f5 fails that test,
# because the symmetry measure is computed against the assigned ideal polyhedron
# and is meaningless without it -- which is also why f5 is 999 exactly when no
# geometry was assigned.
#
# THE RULE: at every identity level, attach all modifiers, then drop them one at
# a time in reverse priority order:
#
#       X|dmin|a_bin   ->   X|a_bin   ->   X
#
# applied at all eleven identity levels with no exception. At the finest level
# this emits the full token, so the full-token rung is not a special case -- it is
# what the rule produces at the top.
#
# a_bin outlives dmin because most atoms share one dmin value, so a class keyed on
# dmin often says only "normal", whereas the asymmetry divides the population more
# evenly. dmin is not thereby demoted: it is present at the top rung of every
# level and is separately guaranteed by the F_elem_cn_dmin floor.
#
# THIS IS NOT A SINGLE REFINEMENT CHAIN. It is a lexicographic product of two
# axes, identity and modifier, and which axis is the outer one decides the whole
# shape of the ladder. Here the modifier axis is outer, so every identity level is
# offered with both modifiers attached before dmin is dropped anywhere: the
# modifier state coarsens monotonically down the ladder and the identity chain is
# re-offered at each modifier state. The preference this encodes is "a more
# specific deformation over a more specific environment".

_IDENTITY_LEVELS = [
    # R0 base is the OLD (pre-_dists) full token; with both modifiers attached it
    # is the current full token, which is why rung 1 is renamed R0_full below.
    ("R0",  "",             "elem||'|'||spin||'|'||nb||'|'||poly||'|'||f5"),
    ("R1",  "drop_f5",      "elem||'|'||spin||'|'||nb||'|'||poly"),
    ("R2",  "drop_spin",    "elem||'|'||nb||'|'||poly"),
    ("R2a", "unrel_poly",   "elem||'|'||nb||'|'||poly_reliable"),
    ("R2b", "binned_mult",
     f"elem||'|'||nb_binned||'|'||poly_reliable||'|'||{CN_SIG}"),
    ("R3",  "nbset_poly",
     f"elem||'|'||nbset||'|'||poly_reliable||'|'||{CN_SIG}"),
    ("R4",  "nbset_cn",     f"elem||'|'||nbset||'|'||{CN_SIG}"),
    # R5 and R8 carry no coordination number: nbset drops multiplicities and elem
    # has none. They still get the full modifier chain, because dmin and a_bin are
    # per-atom quantities and do not need an identified polyhedron to mean
    # something.
    ("R5",  "nbset",        "elem||'|'||nbset"),
    ("R6",  "elem_poly",    f"elem||'|'||poly_reliable||'|'||{CN_SIG}"),
    # R7 with a_bin attached has the same signature as the F_elem_cn_abin floor.
    # That is not a duplication bug: the floor is a presence guarantee over the
    # whole vocabulary with no support test, while this rung is a demand class that
    # only tokens reaching it can satisfy. See FLOOR_SIGS_ASYM.
    ("R7",  "elem_cn",      f"elem||'|'||{CN_SIG}"),
    ("R8",  "elem",         "elem"),
]

# The modifier chain, finest first. Adding a third modifier means one more entry
# here and nothing else: the rule and the CN-dropping derivation both follow from
# this list.
_MODIFIER_CHAIN = [
    ("dA", "||'|'||dmin||'|'||a_bin"),   # every modifier
    ("A",  "||'|'||a_bin"),              # drop dmin (lower priority survives)
    ("",   ""),                          # identity alone
]


def _build_ladder():
    """The rungs the modifier rule produces, finest first: 33 for 11 levels.

    The modifier loop is OUTSIDE the identity loop. Every identity level is tried
    with both modifiers attached (rungs 0-10) before dmin is given up anywhere
    (rungs 11-21, a_bin only), and a_bin goes last (rungs 22-32). So a token gives
    up its chemistry before it gives up its length scale.

    Nesting the loops the other way would produce the same 33 rungs in a different
    order, and every rung-level invariant would still hold -- each rung a total
    function of the token, the central element never dropped, CN carried except at
    R5 and R8. What the order changes is WHICH rung a given token resolves at,
    because resolution is first-match.

    Worth knowing when reading a run log: the last rung of the first block is
    R8dA_elem = elem|dmin|a_bin, which has few enough combinations that nearly all
    of them clear any production T. Almost nothing reaches rungs 11-32, so the
    ladder behaves like an 11-rung one with a long terminal safety net. That net is
    not dead code: it catches an (element, dmin, a_bin) combination too rare to
    clear T, which would otherwise fall straight to the per-element catch-all.

    The trade is that a token failing its own identity level with both modifiers
    attached coarsens its chemistry towards elem|dmin|a_bin rather than dropping
    its length scale. Distance resolution is bought with identity resolution.
    """
    out = []
    for suffix, mod in _MODIFIER_CHAIN:
        for prefix, desc, ident in _IDENTITY_LEVELS:
            name = f"{prefix}{suffix}_{desc}" if desc else f"{prefix}{suffix}"
            out.append((name, ident + mod))
    return out


LADDER_BASE = _build_ladder()
# Rung 0 is the full token. Give it the name the rest of this file uses, and check
# the two definitions have not drifted apart.
#
# `raise`, not `assert`: the rebinding below runs unconditionally, so under
# `python -O`, which strips asserts, a drifted _IDENTITY_LEVELS[0] would silently
# substitute R0_FULL_RUNG's expression at rung 0 while the other rungs built from
# that same identity level still carried the drifted one -- a ladder whose level-0
# rungs disagree about what level 0 is.
if LADDER_BASE[0][1] != R0_FULL_RUNG[1]:
    raise RuntimeError(
        "the modifier rule's top rung is no longer the full token: R0_FULL_RUNG "
        "and _IDENTITY_LEVELS[0] have drifted apart.\n"
        f"  rule emits: {LADDER_BASE[0][1]}\n"
        f"  R0_FULL_RUNG: {R0_FULL_RUNG[1]}")
LADDER_BASE[0] = R0_FULL_RUNG

# The full-token rung is part of the rule, not an option: it is what "attach every
# modifier" produces at the finest identity level. The environment variable removes
# it, which deliberately breaks the uniformity of the rule and exists only so the
# rung can be ablated. It defaults to keeping the rung.
#
# THESE TWO CONSTANTS ARE NOT WHAT A RUN USES. main() owns the decision, via its
# own `enable_full_token_rung` bool, and builds `ladder_active` from LADDER_BASE;
# it passes that explicitly to build_demand_mapping and records it in
# restart_config and the output filename. LADDER exists only so that
# build_demand_mapping / ladder_fingerprint / topup_floor_coverage have a concrete
# default argument to bind at import time, and so that the tests and the audit
# script have a module-level view of the default ladder. If you edit main()'s bool
# and leave V13A_FULL_TOKEN unset, LADDER here and ladder_active in main() WILL
# disagree -- that is harmless, because nothing in a run reads LADDER, but it is
# why every diagnostic below prints ladder_active rather than LADDER.
#
# PARSED BY _env_flag, like every other boolean knob in this file. It was
# originally a hand-rolled `in ('1','true','yes','on')` test, purely because
# _env_flag was defined further down the file than this line -- and that test
# failed SILENTLY IN THE OFF DIRECTION on any other spelling: V13A_FULL_TOKEN=Y,
# =2 or =anything-unrecognised dropped rung 1 and let the run finish, the exact
# failure mode _env_flag's docstring exists to describe. _env_flag now lives at
# the top of the module so this line can use it.
ENABLE_FULL_TOKEN_RUNG = _env_flag('V13A_FULL_TOKEN', True)
LADDER = list(LADDER_BASE) if ENABLE_FULL_TOKEN_RUNG else list(LADDER_BASE[1:])
# Rung indices (into LADDER) that intentionally do NOT preserve coordination
# number. DERIVED FROM THE RUNG NAMES, NOT HARD-CODED -- this changed in _and_A and
# had to. The tuple was literally (7, 10) pre-_dists, (9, 12) in v12 and (14, 19)
# in v13, and every one of those had to be re-derived by hand whenever a rung moved.
# It cannot be a literal here at all: the full-token rung is removable at run time
# (V13A_FULL_TOKEN=0), which shifts every index below it by one, so no literal is
# correct in both cases.
#
# test_v13_dists_and_A.py section D derives the same predicate INDEPENDENTLY from
# the rung expressions and asserts it matches this list, so a rename or a new
# CN-free rung fails the suite rather than silently disabling the CN-preservation
# invariant.
#
# ONE THING TO KNOW WHEN CHECKING THESE INVARIANTS AGAINST THIS FILE: its
# fixture builder emits FIVE-field tokens, which this script's parse check rejects
# by design. It needs the seven-field grammar. (Its check 4 used to hard-code the
# floor pseudo-rungs as exactly {-100, -101}, which would have reported this
# script's third and fourth floors as CN violations that are not real; that is
# fixed -- it now derives the set from FLOOR_RUNG_BASE and the active floor count.)
# SIX names, not two: applying the modifier rule at R5 and R8 gives each of them
# three rungs, and all six carry no coordination number. A rung carries one either
# EXPLICITLY via CN_SIG or IMPLICITLY by keeping the exact multiset `nb`, from
# which cn = len(split(nb,'-')); `nbset` (multiplicities dropped) and `nb_binned`
# (multiplicities bucketed) do NOT carry one, which is exactly why the rungs built
# on those two have CN_SIG appended. test_v13_dists_and_A.py derives this
# predicate from the expressions and asserts it matches, so a rename or a new
# CN-free rung fails the suite rather than silently disabling the invariant.
_CN_DROPPING_NAMES = ("R5dA_nbset", "R5A_nbset", "R5_nbset",
                      "R8dA_elem", "R8A_elem", "R8_elem")
CN_DROPPING_RUNGS = tuple(i for i, (name, _) in enumerate(LADDER)
                          if name in _CN_DROPPING_NAMES)
# `raise`, not `assert`, for the same reason as the R0_FULL_RUNG guard above: this
# is the only thing standing between a renamed rung and a silently disabled
# CN-preservation invariant, and `python -O` would delete an assert.
if len(CN_DROPPING_RUNGS) != len(_CN_DROPPING_NAMES):
    raise RuntimeError(
        f"CN_DROPPING_RUNGS derivation found {len(CN_DROPPING_RUNGS)} of "
        f"{len(_CN_DROPPING_NAMES)} expected rungs in LADDER; a rung was renamed. "
        f"missing: {sorted(set(_CN_DROPPING_NAMES) - {n for n, _ in LADDER})}")

# ---------------------------------------------------------------------------
# FLOORS
#
# A floor is a coarse signature that must be present in the subset whatever its
# support. A rung is a demand class WITH a support test, so a rare combination
# that never reaches T sites at any rung falls through and is never demanded. A
# floor has no support test, so it guarantees presence however rare the
# combination is. No arrangement of rungs substitutes for that, and the rare
# combinations are exactly the ones a support test cannot catch.
#
# Floors are enforced after the greedy, by topping the subset up with whatever is
# missing.
FLOOR_SIGS_BASE = [
    ("F_elem_poly", "elem||'|'||poly_reliable"),
    # CN_SIG, not CAST(cn AS VARCHAR): otherwise this floor is satisfied for elem|1
    # by a site whose geometry was never determined, and a genuine 1-coordinate site
    # of that element counts as already present. See CN_SIG.
    ("F_elem_cn",   f"elem||'|'||{CN_SIG}"),
]

# The distance floor. Optional: the knob is `enable_dist_floor` at the top of
# main(), and V10_DIST_FLOOR overrides it per run. This is only the signature it
# switches on.
FLOOR_SIGS_DIST = [
    ("F_elem_cn_dmin", f"elem||'|'||{CN_SIG}||'|'||dmin"),
]

# The asymmetry floor. On by default; `enable_asym_floor` at the top of main(),
# or V13A_ASYM_FLOOR=0 to compare against a run without it.
#
# Its signature is the same as rung R7A_elem_cn. That is not a duplication bug:
# the rung is reachable only by tokens that survived every rung above it and is
# support-tested, while the floor is a presence guarantee over the whole
# vocabulary. The floor is what catches rare combinations, which are exactly the
# ones that fail a support test.
#
# Do not fold it and the distance floor into one elem|CN|dmin|a_bin signature.
# Two separate floors give the same presence guarantee on each axis and cost far
# fewer signatures than the cross-product.
FLOOR_SIGS_ASYM = [
    ("F_elem_cn_abin", f"elem||'|'||{CN_SIG}||'|'||a_bin"),
]

def active_floor_sigs(enable_dist_floor, enable_asym_floor):
    """The floors that are ENFORCED (as demand classes or as top-up).

    Both arguments are REQUIRED on purpose. Every user-tunable knob in this script
    lives at the top of main(); giving these defaults would put a second, silent
    copy of that decision at module level, and the two would drift. _and_A adds the
    second argument rather than defaulting it for the same reason -- a caller that
    has not thought about the asymmetry floor should fail, not silently omit it.
    """
    return (FLOOR_SIGS_BASE
            + (FLOOR_SIGS_DIST if enable_dist_floor else [])
            + (FLOOR_SIGS_ASYM if enable_asym_floor else []))

# BASELINE for default arguments only -- NOT the knob. Python binds default
# arguments at def time, so build_demand_mapping / topup_floor_coverage /
# ladder_fingerprint need a concrete list at import; that is all this is. It is
# fixed at the dist-free pair, which is both the conservative choice and exactly
# the pre-_dists behaviour, so any caller that forgets floor_sigs= gets the same
# floors the old script had rather than a surprise.
#
# THE ACTUAL CHOICE IS `enable_dist_floor` AT THE TOP OF main(), which resolves to
# floor_sigs_active and is passed explicitly everywhere. Rebinding FLOOR_SIGS at
# run time would NOT reach those defaults -- that is the trap that silently built
# two floors instead of three while this change was being measured.
FLOOR_SIGS = FLOOR_SIGS_BASE

# Every floor that the end-of-run check REPORTS on, enforced or not. Reporting is
# free (one COUNT DISTINCT per signature over v10_tok) and a disabled floor's gap
# is exactly the number needed to decide whether to enable it.
# _and_A: the asymmetry floor is listed here as well as being enforced by default.
# The point of this list is that a run REPORTS how many (element, CN, asymmetry)
# combinations it left uncovered whether or not it enforced them -- so if either
# optional floor is switched off (V13A_ASYM_FLOOR=0 / V10_DIST_FLOOR=0), the cost
# of that choice is visible in the same run rather than having to be asked for.
# The asymmetry gap is the direct read-out on the problem this whole change set
# exists to fix.
ALL_FLOOR_SIGS_FOR_REPORT = FLOOR_SIGS_BASE + FLOOR_SIGS_DIST + FLOOR_SIGS_ASYM

# PSEUDO-RUNG SENTINELS. demand_mapping.rung is a LADDER index for real rungs and
# one of these for the two synthetic families. They are a property of the demand
# space, so they are named once here rather than being spelled as bare literals in
# every diagnostic query -- adding a third family should not mean auditing each
# report by hand. The j-th floor family is written as FLOOR_RUNG_BASE - j, hence
# `rung <= FLOOR_RUNG_BASE` identifies any floor.
RUNG_CATCHALL = -1
FLOOR_RUNG_BASE = -100


def ladder_fingerprint(ladder=LADDER, floor_sigs=FLOOR_SIGS):
    """Hash of the SQL that DEFINES the demand space, for restart validation.

    Lives beside the definitions it hashes, and takes the same arguments
    build_demand_mapping does, so a caller passing a custom ladder gets a
    fingerprint describing the mapping that was actually built rather than the
    module-level default.

    Rung NAMES alone are not enough: editing a rung's expression while keeping its
    name repartitions the whole demand space yet leaves a name-only fingerprint
    unchanged, so a stale restart.npz would be accepted and its coverage silently
    attributed to different classes. That is not hypothetical -- the CN fix rewrote
    the R2b/R3/R6 expressions without renaming those rungs. Whitespace is
    normalised so reformatting does not invalidate a restart file.
    """
    def _norm(s):
        return ' '.join(s.split())
    # POLY_RELIABLE and NB_BINNED need their own entries because the rungs refer to
    # them by COLUMN name (poly_reliable, nb_binned), so editing them would not
    # change the rung text. CN_SIG deliberately has no entry: it is interpolated
    # into the rung/floor expressions, so its text is already hashed above. Any new
    # constant referenced by column name must be added here.
    parts = ([(name, _norm(expr)) for name, expr in ladder]
             + [(name, _norm(expr)) for name, expr in floor_sigs]
             + [('POLY_RELIABLE', _norm(POLY_RELIABLE)),
                ('NB_BINNED', _norm(NB_BINNED))])
    return hashlib.sha256(repr(parts).encode()).hexdigest()


def _build_parsed_token_table(conn):
    """Materialise the parsed vocabulary ONCE (every rung would otherwise reparse).

    Joins chemenv_frequency_table to environment_mapping so every row carries the
    env_index used by material_vectors. An INNER join is deliberate: a token with
    no env_index has no material vectors and could never be covered; the count is
    reported so a truncated environment_mapping cannot pass silently.
    """
    t0 = time.time()
    # _dists: the slice bounds moved because the token grew two trailing fields.
    # poly p[4:-2] -> p[4:-4] and f5 p[-1] -> p[-3]; dmin/dmax are new. See TOKEN
    # GRAMMAR above for the shapes this was checked against.
    # _and_A: the SLICE BOUNDS ARE UNCHANGED -- the token still has the same number
    # of trailing fields. Only the MEANING of the last one changed, from dmax to
    # the asymmetry bin, so the only edit here is the column name. That is exactly
    # why the format guard below exists: nothing about the shape distinguishes the
    # two generations, so a stale table parses perfectly and selects wrongly.
    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE v10_parsed AS
        SELECT env_index, env_id, frequency, elem, spin, nb, poly, f5, cn, nbset,
               dmin, a_bin, {POLY_RELIABLE} AS poly_reliable
        FROM (
            SELECT env_index, env_id, frequency,
                   p[1]                          AS elem,
                   p[2]                          AS spin,
                   p[3]                          AS nb,
                   array_to_string(p[4:-4], '_') AS poly,
                   p[-3]                         AS f5,
                   p[-2]                         AS dmin,
                   p[-1]                         AS a_bin,
                   len(str_split(p[3], '-'))     AS cn,
                   array_to_string(list_sort(list_distinct(str_split(p[3], '-'))), '-') AS nbset
            FROM (
                SELECT cf.env_id, em.env_index, cf.frequency,
                       str_split(cf.env_id, '_') AS p
                FROM env_freq_db.chemenv_frequency_table cf
                JOIN environment_mapping em ON cf.env_id = em.env_id
            )
        )
    """)

    # nb_binned is a PURE function of `nb` alone, and `nb` repeats heavily across
    # the vocabulary (measured on OMat24: 14,975,951 distinct neighbour multisets
    # for 124,173,161 tokens, an 8.29x reuse factor). Evaluating the O(k^2)
    # expression once per DISTINCT nb rather than once per token therefore does
    # ~8x less work, and gives byte-identical results.
    #
    # Scale note, to avoid over-selling this: measured in isolation nb_binned runs
    # at ~2.3M rows/s, i.e. of order a minute for the full 124M vocabulary, so the
    # absolute saving here is seconds, not the ~75 minutes an earlier measurement
    # suggested. That figure came from timing the WHOLE parsed-table build (parquet
    # scan + nbset + poly_reliable + nb_binned) and wrongly attributing it all to
    # nb_binned. Kept because it is free and provably result-identical, not because
    # this step was ever the bottleneck.
    #
    # LEFT JOIN, deliberately: an INNER join would silently DROP any token whose
    # `nb` were NULL (SQL NULL never matches itself), violating the
    # "nothing is discarded" invariant. LEFT JOIN yields NULL nb_binned for such a
    # token instead -- exactly what the per-row expression would have produced.
    t_nb = time.time()
    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE v10_nbmap AS
        SELECT nb, {NB_BINNED} AS nb_binned
        FROM (SELECT DISTINCT nb FROM v10_parsed)
    """)
    n_nb = conn.execute("SELECT COUNT(*) FROM v10_nbmap").fetchone()[0]
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE v10_tok AS
        SELECT p.env_index, p.env_id, p.frequency, p.elem, p.spin, p.nb, p.poly,
               p.f5, p.cn, p.nbset, p.dmin, p.a_bin, p.poly_reliable, m.nb_binned
        FROM v10_parsed p
        LEFT JOIN v10_nbmap m ON m.nb = p.nb
    """)
    conn.execute("DROP TABLE IF EXISTS v10_parsed")
    conn.execute("DROP TABLE IF EXISTS v10_nbmap")
    print(f"[v10] nb_binned computed over {n_nb:,} distinct neighbour multisets "
          f"({time.time()-t_nb:.1f}s)", flush=True)
    n_tok, n_obs = conn.execute(
        "SELECT COUNT(*), SUM(frequency) FROM v10_tok").fetchone()
    n_freq = conn.execute(
        "SELECT COUNT(*) FROM env_freq_db.chemenv_frequency_table").fetchone()[0]
    print(f"[v10] parsed vocabulary: {n_tok:,} tokens / {int(n_obs or 0):,} site "
          f"observations   ({time.time()-t0:.1f}s)", flush=True)
    if n_tok != n_freq:
        print(f"[v10] WARNING: {n_freq - n_tok:,} of {n_freq:,} env_ids in the "
              f"frequency table have NO env_index in environment_mapping and are "
              f"therefore uncoverable. They are excluded.", flush=True)
    # Parse sanity: these must all be zero or the grammar assumption is wrong.
    #
    # _dists adds four checks, and they are the ONLY thing standing between a
    # stale 5-field database and a silently wrong demand space. The empty-poly
    # check alone already catches it -- p[4:-4] over a 5-part token is an empty
    # slice -- but it catches it for a reason that would evaporate the moment a
    # polyhedron label with three underscore-separated parts appeared, so the
    # trailing fields are validated directly as well:
    #   * f5/dmin/a_bin must be integers. On a 5-field token p[-3] is the
    #     neighbour list or the polyhedron, which does not cast.
    #   * dmin = 999 <=> a_bin = 999. Every exit of _distance_ratio_bins() sets
    #     DIST_BIN_UNDEFINED for BOTH fields or for neither, so a token with one
    #     of them undefined cannot have come from the tokeniser. This is also the
    #     cheapest possible detector of an off-by-one in the slice bounds, since
    #     a shifted parse breaks the pairing long before it breaks the casts.
    bad = conn.execute("""
        SELECT COUNT(*) FILTER (WHERE poly IS NULL OR poly = ''),
               COUNT(*) FILTER (WHERE elem IS NULL OR elem = ''),
               COUNT(*) FILTER (WHERE cn < 1),
               COUNT(*) FILTER (WHERE TRY_CAST(f5 AS INTEGER) IS NULL),
               COUNT(*) FILTER (WHERE TRY_CAST(dmin AS INTEGER) IS NULL),
               COUNT(*) FILTER (WHERE TRY_CAST(a_bin AS INTEGER) IS NULL),
               COUNT(*) FILTER (WHERE (dmin = '999') <> (a_bin = '999'))
        FROM v10_tok
    """).fetchone()
    if any(bad):
        raise RuntimeError(
            f"[v11-dists] token parse failure: empty poly={bad[0]}, "
            f"empty elem={bad[1]}, cn<1={bad[2]}, non-integer f5={bad[3]}, "
            f"non-integer dmin={bad[4]}, non-integer a_bin={bad[5]}, "
            f"dmin/a_bin 999-mismatch={bad[6]}.\n"
            f"This script expects the token "
            f"<elem>_<spin>_<nb>_<poly>_<f5>_<dmin>_<a_bin> emitted by "
            f"the stage 01 tokeniser. If these "
            f"databases were built by the older five-field tokeniser, use "
            f"an earlier, pre-distance version of this script "
            f"instead -- do NOT 'fix' this check, because the old script's "
            f"parser accepts the new token silently and mis-assigns poly/f5.")

    # ---------------------------------------------------------------------
    # _and_A FORMAT GUARD. This is the check that matters most in this file.
    #
    # The _dists token and the _and_A token have the SAME SHAPE and differ only in
    # what the LAST FIELD MEANS. Every test above passes on either one. Without
    # this guard a stale frequency table parses perfectly, the run completes, and
    # it selects a subset against a distance bin it believed was an asymmetry bin
    # -- wrong, and silent.
    #
    # The VALUE RANGES do separate them, decisively:
    #
    #     field          _dists token        _and_A token
    #     p[-1]          dmax  0..3, 999     a_bin  0..2, 999
    #     p[-2]          dmin  0..3, 999     dmin   0..4, 999
    #
    # a_bin is digitized against TWO edges (1.0, 3.0), so 3 is unreachable: ONE
    # token with 3 in the last field PROVES the table came from the _dists
    # tokeniser. That test is one-sided and cannot produce a false positive.
    # dmin is digitized against FOUR edges (0.60, 0.70, 0.85, 1.15), so a 4 proves
    # the opposite; that one is only a WARNING, because a small or unlucky table
    # could legitimately contain no dmin=4 (about 2.5% of sites corpus-wide).
    n_dmax_like, n_dmin_over, n_dmin4 = conn.execute("""
        SELECT COUNT(*) FILTER (WHERE TRY_CAST(a_bin AS INTEGER) = 3),
               COUNT(*) FILTER (WHERE TRY_CAST(dmin AS INTEGER) BETWEEN 5 AND 998),
               COUNT(*) FILTER (WHERE TRY_CAST(dmin AS INTEGER) = 4)
        FROM v10_tok
    """).fetchone()
    if n_dmax_like:
        raise RuntimeError(
            f"[v13-and_A] WRONG TOKEN GENERATION: {n_dmax_like:,} tokens carry a 3 "
            f"in the last field. The asymmetry bin is digitized against two edges "
            f"(1.0, 3.0) and can only be 0, 1, 2 or 999, so a 3 there means this "
            f"frequency table was built by "
            f"an earlier tokeniser, whose last field "
            f"is dmax (0..3, 999).\n"
            f"Nothing else would have caught this -- the two tokens have the same "
            f"shape, so the run would have finished and produced a subset selected "
            f"against the wrong quantity. Either use "
            f"the matching earlier version of this script "
            f"for a _dists table, or rebuild the frequency table with "
            f"the stage 01 tokeniser.")
    if n_dmin_over:
        raise RuntimeError(
            f"[v13-and_A] {n_dmin_over:,} tokens have dmin outside 0..4 and not "
            f"999. dmin is digitized against four edges (0.60, 0.70, 0.85, 1.15), "
            f"so 0..4 plus the 999 sentinel are the only legal values. Either the "
            f"table is corrupt or the slice bounds above are wrong.")
    if not n_dmin4:
        print("[v13-and_A] WARNING: no token has dmin = 4 (shortest contact "
              "stretched beyond 1.15 of the covalent-radius sum). That bin holds "
              "~2.5% of sites corpus-wide, so its complete absence suggests either "
              "a very small table or a _dists table that happens to contain no "
              "dmax = 3. Check the occupancy report below before trusting the run.",
              flush=True)
    # Distance-bin occupancy, weighted by SITES rather than by distinct tokens --
    # the ladder's support threshold is counted in sites, so this is the
    # distribution that actually predicts how far the distance rungs coarsen.
    # Printed unconditionally: it is one cheap aggregate over a table that is
    # already resident, and it is the first place a mis-binned corpus shows up.
    print("[v11-dists] distance-bin occupancy (% of site observations):", flush=True)
    print(conn.execute("""
        SELECT dmin, SUM(frequency) AS sites,
               ROUND(100.0 * SUM(frequency) / SUM(SUM(frequency)) OVER (), 2) AS pct
        FROM v10_tok GROUP BY 1 ORDER BY 1
    """).df().to_string(index=False), flush=True)
    print("[v13-and_A] asymmetry-bin occupancy (% of site observations):", flush=True)
    print(conn.execute("""
        SELECT a_bin, SUM(frequency) AS sites,
               ROUND(100.0 * SUM(frequency) / SUM(SUM(frequency)) OVER (), 2) AS pct
        FROM v10_tok GROUP BY 1 ORDER BY 1
    """).df().to_string(index=False), flush=True)
    return int(n_tok)


def build_demand_mapping(conn, T, design, n_coverage,
                         ladder=LADDER, enable_floors=False, persist_path=None,
                         support_metric='sites', floor_sigs=FLOOR_SIGS):
    """Assign every token to a demand class with support >= T. Creates TEMP table
    `demand_mapping(env_index, demand_index, rung)`.

    design='a'  support counted over UNRESOLVED tokens only.
                Strict partition: exactly one row per token, so the material x
                class matrix has the same nnz as today (usually fewer, since
                sibling environments merge). A class can only be satisfied by a
                site whose own token resolved there, so genuinely rare chemistry
                really enters the subset.

    design='d'  support counted over ALL tokens sharing the signature.
                A class may then be satisfied by a FREQUENT token that happens to
                share the coarse signature, so the rare token that motivated the
                class need not be selected -- only its chemistry class is
                guaranteed. Requires emitting one row per (token, active
                ancestor) pair. The '~9x' inflation this used to quote was for the
                13-rung pre-_dists ladder; the bound is one row per RUNG, so it is
                now up to 33x plus one per active floor. Measured on the _and_A
                fixture with all four floors: 6.29x, against 5.00x for design 'a'.
                'a' is strongly preferred at this ladder length.

    Both keep min_freq_env == 1 semantics: no token is ever discarded.

    support_metric='sites'      support = total site observations of the class.
                               Cheap (frequency table only). Does NOT guarantee
                               zero forced materials -- see the header note.
    support_metric='materials'  support = number of DISTINCT materials containing
                               the class. Guarantees >= T independent suppliers,
                               hence zero forced materials, but needs one pass
                               over material_vectors per rung with an exact
                               COUNT(DISTINCT) per class.
    """
    if design not in ('a', 'd'):
        raise ValueError(f"coarsening_design must be 'a' or 'd', got {design!r}")
    if support_metric not in ('sites', 'materials'):
        raise ValueError(f"support_metric must be 'sites' or 'materials', "
                         f"got {support_metric!r}")
    if T < n_coverage + 1:
        raise ValueError(
            f"coarsening_T={T} must be >= n_coverage+1={n_coverage+1}; otherwise a "
            f"demand class can have zero slack and its materials become forced, "
            f"which is exactly the blow-up this design removes.")

    n_tok = _build_parsed_token_table(conn)
    print(f"[v10] building demand mapping: T={T}, design='{design}', "
          f"support='{support_metric}', floors={enable_floors}", flush=True)

    if support_metric == 'materials':
        t0 = time.time()
        conn.execute("""
            CREATE OR REPLACE TEMP TABLE v10_inc AS
            SELECT mat_index, unnest(indices) AS env_index FROM material_vectors
        """)
        n_inc = conn.execute("SELECT COUNT(*) FROM v10_inc").fetchone()[0]
        print(f"[v10] incidence rows (material x environment): {n_inc:,}   "
              f"({time.time()-t0:.1f}s)", flush=True)

    # _dists: dmin/dmax added. This list is what the still-unresolved pool carries
    # from rung to rung, so a rung expression can only reference a column named
    # here -- omitting them would make every distance-bearing rung fail with
    # 'Referenced column not found' at the first coarsening step.
    cols = ("env_index, frequency, elem, spin, nb, poly, f5, cn, nbset, "
            "dmin, a_bin, poly_reliable, nb_binned, env_id")
    conn.execute(f"CREATE OR REPLACE TEMP TABLE v10_unres AS SELECT {cols} FROM v10_tok")
    conn.execute("CREATE OR REPLACE TEMP TABLE v10_resolved "
                 "(env_index INTEGER, rung INTEGER, sig VARCHAR)")

    for i, (name, expr) in enumerate(ladder):
        n_left = conn.execute("SELECT COUNT(*) FROM v10_unres").fetchone()[0]
        if n_left == 0:
            break
        t0 = time.time()
        # design 'a': support from the still-unresolved pool, so rare tokens
        # aggregate among themselves. design 'd': support from the whole
        # vocabulary, so frequent siblings count towards the threshold.
        src = "v10_unres" if design == 'a' else "v10_tok"
        if support_metric == 'sites':
            conn.execute(f"""
                CREATE OR REPLACE TEMP TABLE v10_sup AS
                SELECT ({expr}) AS sig, SUM(frequency) AS tot FROM {src} GROUP BY 1
            """)
        else:
            # Distinct suppliers, not sites: this is what makes slack > 0 and so
            # guarantees the class cannot force any single material.
            conn.execute(f"""
                CREATE OR REPLACE TEMP TABLE v10_sup AS
                SELECT sig, COUNT(DISTINCT mat_index) AS tot
                FROM (SELECT i.mat_index, ({expr}) AS sig
                      FROM v10_inc i JOIN {src} t USING (env_index))
                GROUP BY 1
            """)
        conn.execute(f"""
            CREATE OR REPLACE TEMP TABLE v10_step AS
            SELECT u.*, ({expr}) AS sig FROM v10_unres u
        """)
        conn.execute(f"""
            INSERT INTO v10_resolved
            SELECT s.env_index, {i}, s.sig
            FROM v10_step s JOIN v10_sup p USING (sig)
            WHERE p.tot >= {T}
        """)
        conn.execute(f"""
            CREATE OR REPLACE TEMP TABLE v10_unres AS
            SELECT {cols} FROM v10_step s JOIN v10_sup p USING (sig)
            WHERE p.tot < {T}
        """)
        n_res = conn.execute(
            f"SELECT COUNT(*) FROM v10_resolved WHERE rung = {i}").fetchone()[0]
        n_cls = conn.execute(
            f"SELECT COUNT(DISTINCT sig) FROM v10_resolved WHERE rung = {i}").fetchone()[0]
        print(f"[v10]   {name:18s} resolved {n_res:>12,} tokens into "
              f"{n_cls:>12,} classes   ({time.time()-t0:.1f}s)", flush=True)

    # Anything the ladder could not place (possible under design 'a', where even
    # the coarsest rung may lack support) goes to one explicit catch-all so that
    # NO token is ever silently dropped.
    n_left = conn.execute("SELECT COUNT(*) FROM v10_unres").fetchone()[0]
    if n_left:
        # PER-ELEMENT catch-all: never merge different central elements, not even
        # as a last resort. (Under design 'a' even R8 can fail, because support is
        # counted over the unresolved pool only.)
        print(f"[v10]   WARNING: {n_left:,} tokens unresolved by the whole ladder; "
              f"assigning them to PER-ELEMENT catch-all classes. Their materials "
              f"may still be forced if that support is < T.", flush=True)
        conn.execute(f"INSERT INTO v10_resolved SELECT env_index, "
                     f"{RUNG_CATCHALL}, '__CATCHALL__|' || elem FROM v10_unres")

    # Dense, contiguous class ids -> sparse-matrix column indices.
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE v10_class AS
        SELECT rung, sig,
               CAST(ROW_NUMBER() OVER (ORDER BY rung, sig) - 1 AS INTEGER) AS demand_index
        FROM (SELECT DISTINCT rung, sig FROM v10_resolved)
    """)
    n_classes = conn.execute("SELECT COUNT(*) FROM v10_class").fetchone()[0]

    if design == 'a':
        conn.execute("""
            CREATE OR REPLACE TEMP TABLE demand_mapping AS
            SELECT r.env_index, c.demand_index, r.rung
            FROM v10_resolved r JOIN v10_class c USING (rung, sig)
        """)
    else:
        # design 'd': a token belongs to its class at EVERY rung, so it may
        # satisfy any ACTIVE coarse class it matches. Emit one row per
        parts = []
        for i, (name, expr) in enumerate(ladder):
            parts.append(f"""
                SELECT t.env_index, c.demand_index, {i} AS rung
                FROM v10_tok t
                JOIN v10_class c ON c.rung = {i} AND c.sig = ({expr})
            """)
        parts.append(f"""
                SELECT r.env_index, c.demand_index, r.rung
                FROM v10_resolved r JOIN v10_class c USING (rung, sig)
                WHERE r.rung = {RUNG_CATCHALL}
        """)
        conn.execute("CREATE OR REPLACE TEMP TABLE demand_mapping AS "
                     "SELECT DISTINCT env_index, demand_index, rung FROM ("
                     + " UNION ALL ".join(parts) + ")")

    if enable_floors:
        # Approach F. Coarse rungs (R5 drops the polyhedron, R6 drops the
        # neighbour set) merge chemically distinct sites, so covering a class does
        # not guarantee every element/geometry combination appears. These floors
        # make those combinations targets in their own right.
        # COST: one extra row per token per floor -> matrix nnz grows by
        # module default. With both optional floors on, that is 5x, not the 3x the
        # two-floor default would suggest (measured on the _and_A fixture: 5.00x
        # under design 'a'). Off by default; the post-run check below reports floor
        # coverage regardless, and is usually enough.
        #
        # HAZARD: these classes get NO support test, so they can be monopolised and
        # re-force materials -- see the weak-class invariant below for the measured
        # evidence, and prefer enable_floor_topup=True.
        base = n_classes
        # floor_sigs, NOT the module global. Before this was a parameter, a
        # caller passing a custom floor list got it applied to
        # topup_floor_coverage (which has always taken floor_sigs=) but NOT
        # here, so the demand mapping silently used the module default and the
        # fingerprint recorded a floor set the mapping never built. That trap
        # was hit for real while measuring this change: a run intended to test
        # three floors as demand classes built two and reported gaps that the
        # third would have closed.
        for j, (fname, fexpr) in enumerate(floor_sigs):
            # Distinct floor signatures get contiguous ids continuing after `base`.
            conn.execute(f"""
                CREATE OR REPLACE TEMP TABLE v10_floor AS
                SELECT sig, CAST({base} + ROW_NUMBER() OVER (ORDER BY sig) - 1
                                 AS INTEGER) AS demand_index
                FROM (SELECT DISTINCT ({fexpr}) AS sig FROM v10_tok)
            """)
            n_f = conn.execute("SELECT COUNT(*) FROM v10_floor").fetchone()[0]
            conn.execute(f"""
                INSERT INTO demand_mapping
                SELECT t.env_index, f.demand_index, {FLOOR_RUNG_BASE - j}
                FROM v10_tok t JOIN v10_floor f ON f.sig = ({fexpr})
            """)
            base += n_f
            n_classes += n_f
            print(f"[v10]   floor {fname:14s} added {n_f:,} classes", flush=True)

    n_rows = conn.execute("SELECT COUNT(*) FROM demand_mapping").fetchone()[0]
    print(f"[v10] demand universe: {n_classes:,} classes "
          f"(from {n_tok:,} raw tokens -> x{n_tok/max(n_classes,1):.1f} smaller)",
          flush=True)
    print(f"[v10] demand_mapping rows: {n_rows:,} "
          f"(expansion x{n_rows/max(n_tok,1):.2f} vs one-row-per-token)", flush=True)
    if n_rows > 2 * n_tok:
        print(f"[v10] WARNING: the mapping expands each token {n_rows/n_tok:.1f}x. "
              f"Matrix nnz and peak RAM grow by the same factor.", flush=True)

    # Invariant 1: site availability per class, as the matrix column will see it.
    # Per-element catch-all classes (rung -1) are reported separately: they are the
    # residue that NO coarsening can fix, because under design 'a' the support of a
    # rung is counted over unresolved tokens only, so if an element's leftover
    # tokens total fewer than T sites there is simply no support to be had. The
    # alternative would be merging across central elements, which is worse.
    n_catch_cls, n_catch_tok = conn.execute(f"""
        SELECT COUNT(DISTINCT demand_index), COUNT(*)
        FROM demand_mapping WHERE rung = {RUNG_CATCHALL}
    """).fetchone()
    # PER-CLASS AVAILABILITY, materialised ONCE. This aggregate (the
    # demand_mapping x v10_tok join over >=124M rows) is wanted twice: here, for
    # the >= T guarantee, and at the very end of main(), for the capped-target
    # verification. Computing it a second time during postprocessing meant a full
    # re-scan at the worst possible moment -- after mat_vecs_dict and the CSR are
    # resident -- so it is kept instead. The table is n_classes rows (~370 MB at
    # 14.4M classes), it survives the cleanup below, and main() reads it directly.
    #
    # `required` is the contract the SELECTOR enforces, min(n_coverage, avail).
    # NOTE ON WHICH availability: `avail` is corpus-wide, whereas the selector caps
    # against availability in the LOADED, DEDUPED pool. The two agree, because
    # collapse_identical_materials keeps min(copies, n_coverage) representatives
    # and therefore preserves min(n_coverage, achievable) exactly (see its
    # docstring); corpus availability is in any case an upper bound, so this check
    # can only ever be conservative, never falsely pass.
    #
    # is_catch/is_floor stay separate MAX(CASE ...) aggregates rather than being
    # derived from MIN(rung): a class spanning both families would be
    # misclassified by a single rung value.
    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE v10_class_avail AS
        SELECT dm.demand_index,
               MIN(dm.rung)        AS rung,
               SUM(t.frequency)    AS avail,
               LEAST({n_coverage}, SUM(t.frequency)) AS required,
               MAX(CASE WHEN dm.rung = {RUNG_CATCHALL} THEN 1 ELSE 0 END) AS is_catch,
               MAX(CASE WHEN dm.rung <= {FLOOR_RUNG_BASE} THEN 1 ELSE 0 END) AS is_floor
        FROM demand_mapping dm JOIN v10_tok t USING (env_index)
        GROUP BY 1
    """)
    # FLOOR classes are excluded from the guarantee, not merely from the message.
    # enable_floors inserts them WITHOUT ever testing their support against T, so a
    # rare (element, polyhedron) or (element, CN) pair becomes a demand class with
    # arbitrarily little support. Counting them as `is_catch = 0` made the
    # '<-- BUG, should be 0' banner fire on classes that were never in scope.
    # Demonstrated on a purpose-built fixture with one rare (Fe, C:12) pair:
    # floors=OFF gave weak_non_catchall=0 / monopolised=3; floors=ON gave
    # weak_non_catchall=2 / monopolised=5, and both extra classes had ALL their
    # sites inside a single material -- zero slack, so floors can re-force
    # materials, precisely what the coarsening exists to prevent.
    weak, weak_non_catch, weak_floor = conn.execute(f"""
        SELECT COUNT(*),
               COUNT(*) FILTER (WHERE is_catch = 0 AND is_floor = 0),
               COUNT(*) FILTER (WHERE is_floor = 1)
        FROM v10_class_avail WHERE avail < {T}
    """).fetchone()
    print(f"[v10] classes with site-availability < T: {weak:,} "
          f"(of which non-catch-all, non-floor: {weak_non_catch:,} "
          f"{'OK' if weak_non_catch == 0 else '<-- BUG, should be 0'})", flush=True)
    if n_catch_cls:
        print(f"[v10] per-element catch-all: {n_catch_cls:,} classes covering "
              f"{n_catch_tok:,} tokens (irreducible residue; may be forced)",
              flush=True)
    if weak_floor:
        print(f"[v10] WARNING: {weak_floor:,} FLOOR classes have fewer than T={T} "
              f"sites. Floors are not subject to the support threshold, so these "
              f"have little or no slack and WILL force their supplying materials. "
              f"Prefer enable_floor_topup=True, which repairs the same gaps after "
              f"selection without creating demand classes.", flush=True)

    # Invariant 2 -- THE one that actually predicts forced materials. A class is
    # forced iff a single material holds all of its sites, because then
    # count_m[e] >= col_sums(e) and slack collapses to 0. This is the honest
    # measure of whether the coarsening did its job.
    conn.execute("DROP TABLE IF EXISTS v10_inc2")
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE v10_inc2 AS
        SELECT u.mat_index, dm.demand_index, SUM(u.cnt) AS cnt
        FROM (SELECT mat_index, unnest(indices) AS env_idx, unnest(counts) AS cnt
              FROM material_vectors) u
        JOIN demand_mapping dm ON dm.env_index = u.env_idx
        GROUP BY 1, 2
    """)
    mono_cls, mono_mats = conn.execute("""
        SELECT COUNT(*) FILTER (WHERE max_one >= total),
               (SELECT COUNT(DISTINCT mat_index) FROM (
                    SELECT demand_index, mat_index, cnt,
                           SUM(cnt) OVER (PARTITION BY demand_index) AS tot2
                    FROM v10_inc2)
                WHERE cnt >= tot2)
        FROM (SELECT demand_index, SUM(cnt) AS total, MAX(cnt) AS max_one,
                     COUNT(DISTINCT mat_index) AS n_mats
              FROM v10_inc2 GROUP BY 1)
    """).fetchone()
    print(f"[v10] monopolised classes (all sites inside ONE material): "
          f"{mono_cls:,} / {n_classes:,}", flush=True)
    # HONEST LABELLING. This is a LOWER BOUND on the number the selector will
    # actually force, for two independent reasons, both benign:
    #   1. It is the round-0 condition only. forced_selection_cascade iterates:
    #      each batch it selects shrinks `available`, which can push further
    #      materials to zero slack.
    #   2. It is computed on the FULL material pool, whereas the selector runs
    #      after collapse_identical_materials. Dedup removes interchangeable
    #      copies, which lowers col_sums for their classes and can therefore
    #      manufacture new monopolies. Those forcings are harmless -- the kept
    #      representative is by construction interchangeable with the dropped
    #      ones -- but they are real and they are counted by [FORCED] below.
    # Observed gap on the first full run: 297 predicted here vs 38,281 actually
    # pre-selected (still 0.06% of OMat24, versus 20,143,347 with no coarsening).
    print(f"[v10] => LOWER BOUND on forced materials: {mono_mats:,} "
          f"(round-0, pre-dedup). The actual count is reported by [FORCED] "
          f"below and will be higher; compare that one against 20,143,347.",
          flush=True)
    conn.execute("DROP TABLE IF EXISTS v10_inc2")

    print("[v10] classes per rung:", flush=True)
    print(conn.execute("""
        SELECT rung, COUNT(DISTINCT demand_index) AS n_classes,
                     COUNT(*) AS n_mapping_rows
        FROM demand_mapping GROUP BY 1 ORDER BY 1
    """).df().to_string(index=False), flush=True)
    if persist_path:
        conn.execute(f"COPY (SELECT dm.env_index, dm.demand_index, dm.rung, "
                     f"c.sig FROM demand_mapping dm LEFT JOIN v10_class c "
                     f"ON c.demand_index = dm.demand_index) "
                     f"TO '{persist_path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        print(f"[v10] demand mapping written to {persist_path}", flush=True)

    # The rung tables are large; release them before the matrix is built.
    # v10_class_avail is deliberately NOT dropped -- main()'s final verification
    # reads it instead of recomputing the same join.
    for tbl in ("v10_sup", "v10_step", "v10_unres", "v10_floor", "v10_inc"):
        conn.execute(f"DROP TABLE IF EXISTS {tbl}")
    gc.collect()
    # ladder_sql_sha256 is computed from the ladder ACTUALLY USED, so it is correct
    # by construction for a caller passing a custom ladder. restart_config consumes
    # it alongside n_classes as the record of what the demand space looks like.
    return {"n_tokens": n_tok, "n_classes": int(n_classes),
            "n_rows": int(n_rows), "weak_classes": int(weak),
            "weak_non_catchall": int(weak_non_catch),
            "weak_floor_classes": int(weak_floor),
            "catchall_classes": int(n_catch_cls),
            "catchall_tokens": int(n_catch_tok),
            "monopolised_classes": int(mono_cls),
            "predicted_forced_materials": int(mono_mats),
            "ladder_sql_sha256": ladder_fingerprint(ladder, floor_sigs)}


def topup_floor_coverage(conn, selected_mat_indices, floor_sigs=FLOOR_SIGS):
    """Approach F applied AFTER selection, instead of as extra demand classes.

    WHY THIS EXISTS
    ---------------
    Coarse rungs merge chemically distinct sites: R5 drops the polyhedron, R6
    drops the neighbour set. Because the demand classes are a PARTITION (each
    token belongs to exactly one), covering every demand class does NOT guarantee
    that every (element, geometry) combination is present -- the representative
    that satisfied a coarse class may carry a different polyhedron or CN than the
    token that needed it.

    `enable_floors` closes that by making the floor signatures extra demand
    classes, but it maps EVERY token to EVERY floor: ~3x the demand_mapping rows
    and ~2x matrix nnz, for +0.04% classes. Measured on the first full run
    (T=30, design 'a', 1,566,657 materials): 172 of 3,997 elem|poly_reliable and
    111 of 1,481 elem|cn combinations were absent. The gap needs a few HUNDRED
    materials, not a doubling of the matrix -- hence this.

    SCOPE: this enforces PRESENCE (>= 1 witnessing site), which is what the floor
    report measures. It is deliberately not a multiplicity guarantee; the
    n_coverage contract applies to demand classes, not to floors.

    Returns the list of added mat_index values (possibly empty).
    """
    sel = list(selected_mat_indices)
    added_all = []
    conn.execute("DROP TABLE IF EXISTS v10_tu_selmat")
    conn.execute("CREATE TEMP TABLE v10_tu_selmat (mat_index BIGINT)")
    conn.register('v10_tu_df', pd.DataFrame(
        {'mat_index': np.asarray(sel, dtype=np.int64)}))
    conn.execute("INSERT INTO v10_tu_selmat SELECT mat_index FROM v10_tu_df")

    for fname, fexpr in floor_sigs:
        t0 = time.time()
        # Environments witnessed by the CURRENT selection (grows as we add).
        conn.execute("""
            CREATE OR REPLACE TEMP TABLE v10_tu_sel AS
            SELECT DISTINCT unnest(mv.indices) AS env_index
            FROM material_vectors mv
            JOIN v10_tu_selmat s ON s.mat_index = mv.mat_index
        """)
        conn.execute(f"""
            CREATE OR REPLACE TEMP TABLE v10_tu_missing AS
            SELECT sig FROM (SELECT DISTINCT ({fexpr}) AS sig FROM v10_tok)
            EXCEPT
            SELECT DISTINCT ({fexpr}) AS sig
            FROM v10_tok t JOIN v10_tu_sel s USING (env_index)
        """)
        n_missing = conn.execute("SELECT COUNT(*) FROM v10_tu_missing").fetchone()[0]
        if n_missing == 0:
            print(f"[v10] floor top-up {fname}: nothing missing", flush=True)
            continue

        # Only the tokens carrying a missing signature -- a small set, so the
        # scan of material_vectors below is a cheap semi-join.
        conn.execute(f"""
            CREATE OR REPLACE TEMP TABLE v10_tu_envs AS
            SELECT t.env_index, ({fexpr}) AS sig FROM v10_tok t
            WHERE ({fexpr}) IN (SELECT sig FROM v10_tu_missing)
        """)
        # Determinism is GUARANTEED BY THE TIE-BREAK in the greedy below, not by
        # this ORDER BY. Originally the max() key was (gain, -cost), and max()
        # returns the FIRST maximum, i.e. dict insertion order -- which is this
        # result set's row order, and the connection runs with
        # preserve_insertion_order=False. Replaying the same greedy under
        # different candidate orderings gave different material sets. Adding
        # -mat_index as a third key makes the
        # winner a total order over distinct materials, so row order no longer
        # matters. The sort is kept as cheap defence-in-depth and to give stable
        # inputs when debugging -- removing it would not break reproducibility.
        cand = conn.execute("""
            SELECT e.sig AS sig, u.mat_index AS mat_index, u.n_sites AS n_sites
            FROM (SELECT mat_index, unnest(indices) AS env_index,
                         list_sum(counts) AS n_sites
                  FROM material_vectors) u
            JOIN v10_tu_envs e ON e.env_index = u.env_index
            ORDER BY sig, mat_index
        """).df()

        # Small weighted set cover: fewest atoms per newly-supplied signature.
        by_mat = defaultdict(set)
        cost = {}
        for r in cand.itertuples(index=False):
            m = int(r.mat_index)
            by_mat[m].add(r.sig)
            cost[m] = int(r.n_sites)
        need = set(conn.execute(
            "SELECT sig FROM v10_tu_missing").df()['sig'].tolist())
        added, unreachable = [], 0
        while need and by_mat:
            # Final tie-break on mat_index so the pick cannot depend on dict order
            # even if two materials match on both gain and cost.
            m, gain = max(
                ((m, len(s & need)) for m, s in by_mat.items()),
                key=lambda kv: (kv[1], -cost.get(kv[0], 0), -kv[0]))
            if gain == 0:
                break
            added.append(m)
            need -= by_mat.pop(m)
        unreachable = len(need)
        if added:
            conn.register('v10_tu_add', pd.DataFrame(
                {'mat_index': np.asarray(added, dtype=np.int64)}))
            conn.execute("INSERT INTO v10_tu_selmat SELECT mat_index FROM v10_tu_add")
        added_all.extend(added)
        print(f"[v10] floor top-up {fname}: {n_missing:,} missing -> added "
              f"{len(added):,} materials"
              + (f"; {unreachable:,} UNREACHABLE (no material supplies them)"
                 if unreachable else "")
              + f"   ({time.time()-t0:.1f}s)", flush=True)

    for t in ("v10_tu_sel", "v10_tu_missing", "v10_tu_envs", "v10_tu_selmat"):
        conn.execute(f"DROP TABLE IF EXISTS {t}")
    return added_all


def main():
    # ---------------- v10 coarsening parameters ----------------
    # T: minimum support a demand class must have. Replaces min_freq_env as the
    #    size knob. MUST be >= n_coverage + 1. Larger T -> coarser classes,
    #    smaller demand universe, smaller subset.
    coarsening_T = 100
    # 'a' = strict partition, rare sites genuinely sampled, matrix nnz unchanged.
    # 'd' = coarse class satisfiable by any matching site; nnz up to ~9x. See
    #       build_demand_mapping docstring. 'a' is recommended.
    coarsening_design = 'a'
    # Approach F, two mutually-exclusive routes:
    #   enable_floors=True       floors become extra demand classes. Correct but
    #                            costly: ~3x demand_mapping rows, ~2x matrix nnz.
    #   enable_floor_topup=True  floors are repaired AFTER selection by adding the
    #                            few hundred materials that supply the missing
    #                            combinations. Same presence guarantee, ~0.02%
    #                            subset growth, no matrix cost. PREFERRED.
    enable_floors = False
    enable_floor_topup = True
    # Whether the distance floor F_elem_cn_dmin (elem|CN|dmin) is enforced.
    #
    # It guarantees that every (element, CN, dmin) combination is present, however
    # rare. The ladder cannot do this on its own: a rung is support-tested, so a
    # combination too rare to reach T at any rung is never demanded. Compressed
    # sites are a small share of atoms but carry a large share of the high forces,
    # and their classes are exactly the rare ones, so this is the residue the floor
    # exists for.
    #
    # V10_DIST_FLOOR=0 ablates it.
    enable_dist_floor = True
    # _and_A: presence guarantee on (element, CN, asymmetry state). This is the
    # mechanism that most directly attacks the measured high-force under-sampling
    # (0.036% of atoms above 50 eV/A vs 0.054% random, 0.082% DIRECT) -- see
    # FLOOR_SIGS_ASYM for why a floor and not a rung.
    #
    # ON as of this revision. Its cardinality is now measured on the full corpus:
    # 4,136 signatures, 59 of them single-site (1.43%), median 53,518 sites. That
    # is CHEAPER ON EVERY AXIS than the distance floor already enforced above
    # (5,675 / 122 / 2.15%), and its singleton fraction sits between the two
    # long-standing production floors (1.58% and 0.71%) whose top-up costs ~0.02%
    # subset growth. The only stated reason for leaving it off was that nobody had
    # measured it. Set V13A_ASYM_FLOOR=0 to compare against a run without it.
    #
    # NOTE the ~0.02% top-up cost is TRANSFERRED from the v10 production run on the
    # old vocabulary for a different floor; the rarity profiles match closely, but
    # this floor's own top-up cost has not been measured directly.
    enable_asym_floor = True
    # Whether the FULL TOKEN is the top rung. It is what the modifier rule emits at
    # the finest identity level (see THE MODIFIER RULE), so True is the rule applied
    # without exception and False is an ABLATION that deliberately breaks it.
    #
    # This is a REAL knob: main() builds `ladder_active` from it below and passes
    # that everywhere, so editing this line works. (It briefly was a dead assignment
    # that read the module constant and was never used -- editing it looked like it
    # would ablate the rung and did nothing. The module-level LADDER still exists,
    # but only as the baseline for default arguments; main() never uses it.)
    # V13A_FULL_TOKEN=0 overrides per run.
    enable_full_token_rung = True
    persist_demand_mapping = None  # e.g. 'demand_mapping.parquet' for inspection
    # 'sites'     : support = site observations. Cheap; a class whose sites all sit
    #               in one material still forces it (see header note).
    # 'materials' : support = distinct supplying materials. Guarantees ZERO forced
    #               materials; costs one pass over material_vectors per rung.
    support_metric = 'materials'

    # min_freq_env IS NO LONGER A KNOB. In v9 it DISCARDED every token below the
    # threshold; v10 discards nothing and instead coarsens rare tokens until their
    # demand class is supported, so the size control is `coarsening_T`. The variable
    # is retained only so that a v9 value carried over by habit fails loudly rather
    # than being silently ignored (verified: setting it to 1000 changed neither the
    # demand universe, nor the subset, nor the output file).
    min_freq_env = 1
    # 'uniform-numba' is the only working option. 'simple' never assigns
    # values that build_sparse_matrix cannot unpack. Both are inherited from v9.
    search_algo = 'uniform-numba'
    n_coverage = 1
    uniformity_weight=0.0
    batch_size=1   # enforced: any other value raises

    # ---------------- per-atom cost model ----------------
    # Exponent on the per-material atom cost in the greedy rule
    #     score(m) = (help(m) + uw*uniformity(m)) / site_count(m) ** cost_alpha
    #
    #   0.0  ignore material size; optimise demand classes per material.
    #   1.0  textbook weighted set-cover greedy, normalising by total ATOMS.
    #   0.5  in between; no approximation guarantee.
    #
    # 1.0 is the production value: for an MLFF training set the cost you actually
    # pay is atoms, not structures, and the approximation guarantee attaches to
    # whatever you normalise by.
    #
    # EXPECT THE MATERIAL COUNT TO RISE as cost_alpha rises, because the rule
    # starts preferring several small cells over one large one. MACE batches by
    # structure, so that changes steps per epoch and preprocessing cost even as the
    # atom total falls. Judge a run on the (total atoms, n_materials) pair, never on
    # either number alone.
    #
    # Sweep it with V11_COST_ALPHA if that trade needs tuning.
    cost_alpha = 1.0

    #v9 preprocessing reductions (both are safe for any min_freq_env/n_coverage;
    #automatically skipped when resuming from a restart file)
    enable_dedup=False
    enable_forced_selection=True

    # _and_A: POST-SELECTION REDUNDANCY PRUNE. After the greedy finishes, drop any
    # min(n_coverage, availability) without it. The greedy never revisits a pick,
    # so materials taken early for one or two classes can be completely subsumed
    # by the surplus of later picks and stay in the subset regardless.
    #
    # ON by default because it is free in both senses: it costs seconds (measured:
    # 13 s for 901,452 materials / 6.87M material-class entries, converging in one
    # sweep) and it cannot remove anything the coverage contract needed -- see
    # prune_redundant_materials for the test and why forced materials and floor
    # classes are protected without special-casing.
    #
    # The prune removes materials whose classes are all still covered without
    # them, leaving every target met and every floor still witnessed.
    #
    # READ IT AS A COMPANION TO cost_alpha, NOT A GENERAL WIN. At cost_alpha=0 the
    # same step recovers 0.32% and 0.00% of atoms on the two fixtures; at
    # cost_alpha=1.0 it recovers 4.54% and 7.08%. cost_alpha=1 prefers many small
    # cells bought for one or two classes, which is exactly the population that
    # later gets subsumed. Set V13A_PRUNE=0 to reproduce the pre-prune subset
    # exactly (the output filename then carries '_noprune').
    #
    # UNLIKE the two reductions above it is NOT skipped on resume: it runs in
    # main() over the whole selection rather than inside the selector, so a
    # resumed run is pruned exactly like an uninterrupted one and the fresh ==
    # resumed subset guarantee is preserved. See the call site.
    enable_redundancy_prune = True

    # Check for restart file
    restart_path = "restart.npz"
    resume_from = restart_path if os.path.exists(restart_path) else None
    initial_selected = None
    initial_coverage = None
    restart_interval = 50000

    
    # Paths and DuckDB resources come from the USER CONFIGURATION block at the
    # top of this file (V10_TOKEN_VEC_DB / V10_ENV_FREQ_DB / V10_THREADS /
    # V10_MEMORY). Nothing site-specific is hardcoded here.
    token_vec_db = TOKEN_VEC_DB_DEFAULT     # stage 03 output
    env_freq_db  = ENV_FREQ_DB_DEFAULT      # stage 02 output
    duck_threads = DUCK_THREADS_DEFAULT
    # 400GB, not 500GB. The node has 503 GiB and the Python side alone peaks at
    # ~165 GiB holding mat_vecs_dict + the CSR matrix (measured system-wide peak on
    # the T=30 run: 188 GiB). Authorising DuckDB to take the entire node while
    # Python already owns a third of it is an OOM trap for the queries that run
    # AFTER the dict exists (the floor top-up scan and the per-env_id distribution
    # query). 400GB still sits far above anything observed, so it costs no
    # spilling; it just stops the two allocators from being allowed to overcommit.
    duck_memory  = DUCK_MEMORY_DEFAULT
    coarsening_T = int(os.environ.get('V10_T', coarsening_T))
    coarsening_design = os.environ.get('V10_DESIGN', coarsening_design)
    enable_floors      = _env_flag('V10_FLOORS',     enable_floors)
    enable_floor_topup = _env_flag('V10_TOPUP',      enable_floor_topup)
    enable_dist_floor  = _env_flag('V10_DIST_FLOOR', enable_dist_floor)
    enable_asym_floor  = _env_flag('V13A_ASYM_FLOOR', enable_asym_floor)
    enable_redundancy_prune = _env_flag('V13A_PRUNE', enable_redundancy_prune)
    enable_full_token_rung  = _env_flag('V13A_FULL_TOKEN', enable_full_token_rung)
    support_metric = os.environ.get('V10_SUPPORT', support_metric)
    cost_alpha = float(os.environ.get('V11_COST_ALPHA', cost_alpha))
    n_coverage = int(os.environ.get('V10_NCOV', n_coverage))

    # Fail loudly on v9 habits rather than ignoring them silently.
    if min_freq_env != 1:
        raise ValueError(
            f"min_freq_env={min_freq_env} has NO EFFECT in v10 and is not a knob. "
            f"v10 discards no tokens; rare tokens are coarsened until supported. "
            f"Use coarsening_T (currently {coarsening_T}) to control subset size, "
            f"and leave min_freq_env = 1.")
    if search_algo != 'uniform-numba':
        raise ValueError(
            f"search_algo={search_algo!r} is not supported. Only 'uniform-numba' "
            f"works: 'simple' never assigns selected_mat_indices, and "
            f"'uniform'/'explore-uniform' build Counter values that "
            f"build_sparse_matrix cannot unpack. Both are pre-existing v9 defects.")

    for _label, _path in (('V10_TOKEN_VEC_DB', token_vec_db),
                          ('V10_ENV_FREQ_DB', env_freq_db)):
        if not os.path.isfile(_path):
            raise FileNotFoundError(
                f"{_path} not found. Set {_label} to the database written by "
                f"stage {'03' if _label.endswith('VEC_DB') else '02'}.")

    time_connect_start = time.time()
    conn_vectors = duckdb.connect(token_vec_db,
            read_only = True,
            config={'threads' : duck_threads,
                    'memory_limit' : duck_memory,
                    'preserve_insertion_order' : False})
    conn_vectors.execute(f"ATTACH '{env_freq_db}' AS env_freq_db (READ_ONLY)")
    time_connect_end = time.time()

    print(f"Time for connecting to new db : {time_connect_end - time_connect_start} s", flush=True)
    
    #Extract only those environments, which appear at least min_freq_env times
    #env_ids = conn_vectors.execute(""" 
    #        """, [min_freq_env]).fetchnumpy()['env_id']
    
    # v10: instead of DISCARDING tokens below min_freq_env, map every token to a
    # demand class with support >= coarsening_T. The coverage universe is now the
    # set of demand classes, not the set of frequent tokens.
    # Resolved ONCE and passed explicitly everywhere. Not read from the module
    # global FLOOR_SIGS, which is frozen into default arguments at def time and
    # would ignore V10_DIST_FLOOR entirely.
    # THE LADDER THIS RUN USES, derived from the knob above and passed explicitly
    # everywhere. Not the module-level LADDER, which is fixed at import time and
    # exists only so build_demand_mapping/ladder_fingerprint have a default
    # argument to bind -- reading it here would ignore `enable_full_token_rung`.
    ladder_active = (list(LADDER_BASE) if enable_full_token_rung
                     else list(LADDER_BASE[1:]))
    floor_sigs_active = active_floor_sigs(enable_dist_floor, enable_asym_floor)
    # WHICH floors are selected is `enable_dist_floor`; WHETHER any of them is
    # enforced is a separate question, because both enforcement routes can be off
    # at once. Conflating the two made the end-of-run report claim every floor was
    # enforced and then print '<-- BUG' for each one on a run that had promised
    # nothing -- a false alarm on a correct run. Computed here so the startup line
    # and the closing report cannot disagree.
    floors_enforced = bool(enable_floors or enable_floor_topup)
    # UNCONDITIONAL, and separate from the floor line. The rung count used to be
    # appended to the floors-enforced message, so a run with both floor routes off
    # never reported which ladder it was using -- and the ablation (V13A_FULL_TOKEN=0,
    # which removes a rung and changes the whole demand space) announced itself
    # nowhere in the startup banner. Both are headline facts about the run.
    # v14: the ORDERING is now a headline fact about the run, for the same reason
    # the rung count and the full-token ablation are -- v13 and v14 differ in
    # nothing else, so a log without it cannot be told apart from a v13 log.
    # `_dA_run` is the length of the leading block in which every rung carries
    # dmin; it is derived from the ladder rather than hard-coded, so the ablation
    # (which removes rung 0) reports 10 instead of 11 without a special case.
    _dA_run = 0
    for _n, _e in ladder_active:
        if not _e.endswith("||'|'||dmin||'|'||a_bin"):
            break
        _dA_run += 1
    print(f"[v14-and_A-deep] ladder: {len(ladder_active)} rungs, MODIFIER-OUTER "
          f"ordering (dmin+A survive the whole identity chain; first {_dA_run} "
          f"rungs all carry dmin), "
          f"{'FULL-TOKEN RUNG ON (rule applied without exception)' if enable_full_token_rung else 'FULL-TOKEN RUNG OFF -- ABLATION, the modifier rule is deliberately broken at the top'}"
          f"   | first rung: {ladder_active[0][0]}"
          f" | last dmin-bearing rung: {ladder_active[_dA_run-1][0] if _dA_run else 'NONE'}",
          flush=True)
    if floors_enforced:
        print(f"[v11-dists] floors enforced "
              f"({'as demand classes' if enable_floors else 'by post-selection top-up'}): "
              f"{', '.join(n for n, _ in floor_sigs_active)}"
              f"{'' if enable_dist_floor else '   (F_elem_cn_dmin DISABLED, gap reported at end of run)'}"
              f"{'' if enable_asym_floor else '   (F_elem_cn_abin DISABLED, gap reported at end of run)'}",
              flush=True)
    else:
        print("[v11-dists] NO floors enforced (enable_floors and "
              "enable_floor_topup are both off). Floor coverage is still MEASURED "
              "at the end of the run, but nothing guarantees it.", flush=True)

    time_demand_start = time.time()
    demand_stats = build_demand_mapping(
            conn_vectors,
            T=coarsening_T,
            design=coarsening_design,
            n_coverage=n_coverage,
            ladder=ladder_active,
            enable_floors=enable_floors,
            persist_path=persist_demand_mapping,
            support_metric=support_metric,
            floor_sigs=floor_sigs_active)
    print(f"Time for building demand mapping : "
          f"{time.time() - time_demand_start} s", flush=True)

    env_indices = conn_vectors.execute("""
            SELECT DISTINCT demand_index FROM demand_mapping ORDER BY demand_index
            """).fetchnumpy()['demand_index'].astype(np.int64)
    print(f"Covering {len(env_indices)} demand classes "
          f"(min_freq_env is effectively 1: no token discarded)", flush=True)

    # Fingerprint of everything that defines the demand_index space. Coverage in a
    # restart file is keyed by demand_index, so resuming with any of these changed
    # would silently credit coverage to the wrong classes. The SQL hash comes from
    # build_demand_mapping, which computed it from the ladder it actually used --
    # see ladder_fingerprint() for why rung NAMES alone are not sufficient.
    restart_config = {
        'coarsening_T': coarsening_T,
        'coarsening_design': coarsening_design,
        'support_metric': support_metric,
        'enable_floors': bool(enable_floors),
        # Only affects the demand_index space when enable_floors=True; in top-up
        # mode the floors are applied after selection and a restart.npz is valid
        # across the toggle. Recorded (and folded into ladder_sql_sha256 via
        # floor_sigs) anyway, so a mismatch is refused rather than reasoned about
        # -- refusing a resume that would in fact have been fine is the cheap
        # error here, crediting coverage to the wrong classes is not.
        'enable_dist_floor': bool(enable_dist_floor),
        # _and_A: both new structural knobs are recorded here, for the same reason
        # enable_dist_floor is -- they change the demand space, so a subset file
        # that does not name them cannot be reproduced from its own metadata.
        # ENABLE_FULL_TOKEN_RUNG is also implicit in 'ladder' below (R0_full is
        # either in that list or it is not), but naming it makes a diff of two
        # metadata blocks readable without comparing 20-element lists.
        'enable_asym_floor': bool(enable_asym_floor),
        'enable_full_token_rung': bool(enable_full_token_rung),
        'n_coverage': n_coverage,
        # v14. STRICTLY REDUNDANT and deliberately kept: 'ladder' below already
        # differs from v13's (same names, different order) and so does
        # ladder_sql_sha256, so a v13 restart.npz is refused with or without this
        # key. It exists for the same reason 'enable_full_token_rung' does -- to
        # make a diff of two metadata blocks readable without eyeballing two
        # 33-element name lists to spot that they are permutations of each other.
        'ladder_order': 'modifier_outer',
        'ladder': [name for name, _ in ladder_active],
        'ladder_sql_sha256': demand_stats['ladder_sql_sha256'],
        'n_classes': int(demand_stats['n_classes']),
        # v11: not part of the demand_index space, but resuming under a different
        # cost-greedy suffix, which is not a greedy solution to either objective.
        # Recorded here so that mismatch is refused loudly by the check below.
        # Side effect (intended): a v10-written restart.npz lacks this key and so
        # cannot be resumed by v11.
        'cost_alpha': float(cost_alpha),
        # Not a demand-space knob (demand_index means the same thing either way),
        # but it changes the SUBSET, so a resume must not silently switch it.
        # Consequence, deliberate: a restart.npz written before this key existed
        # can no longer be resumed -- the same trade cost_alpha already makes.
        'enable_redundancy_prune': bool(enable_redundancy_prune),
        # _dists: ladder_sql_sha256 hashes the LADDER, not the parser, so a change
        # to the token grammar with an unchanged ladder would leave the
        # fingerprint intact and a stale restart.npz would be accepted with its
        # coverage attributed to differently-defined classes. That gap is closed
        # explicitly rather than relying on the ladder text happening to move at
        # the same time (it does move for this change -- but that is luck, not a
        # mechanism).
        'token_grammar': 'elem_spin_nb_poly_f5_dmin_a_bin',
    }

    if resume_from:
        # A leftover restart.npz silently turns an intended fresh run into a
        # resume, so say so unmissably. NOTE on reproducibility: the resulting
        # SUBSET is expected to be identical to an uninterrupted run, and was
        # measured to be so (1,194 materials, identical sets, on a fixture where
        # dedup removed 66.8% of materials). The reason is that
        # collapse_identical_materials keeps sorted(ids)[0] -- the lowest
        # mat_index of each duplicate group -- while the heap breaks score ties by
        # lowest row index, and row order IS mat_index order. So when the resume
        # pool still contains duplicates that dedup would have removed, the greedy
        # picks the same representative anyway. CELF re-scores on pop, so heap
        # history does not matter either. What DOES change is runtime: the skipped
        # reductions mean a larger candidate pool and no forced pre-selection.
        print("=" * 78, flush=True)
        print(f"[RESTART] THIS IS NOT A FRESH RUN. Resuming from {resume_from}.",
              flush=True)
        print("[RESTART] collapse_identical_materials and the forced-selection "
              "cascade are", flush=True)
        print("[RESTART] SKIPPED on resume, so this run is SLOWER than a fresh "
              "one (larger pool,", flush=True)
        print("[RESTART] no forced pre-selection). The final subset is expected "
              "to be identical.", flush=True)
        print("[RESTART] (The redundancy prune runs in main() over the WHOLE "
              "selection, so it is", flush=True)
        print("[RESTART]  applied on resume too and does not disturb that.)",
              flush=True)
        print(f"[RESTART] Delete {resume_from} if you wanted a fresh run.",
              flush=True)
        print("=" * 78, flush=True)
        state = load_restart_file(resume_from)
        initial_selected = state['selected']
        initial_coverage = state['current_coverage']

        saved_cfg = state.get('config')
        if saved_cfg is None:
            print("[RESTART] WARNING: this restart file has no coarsening "
                  "fingerprint (written by v9 or an early v10). It CANNOT be "
                  "validated; coverage may be attributed to the wrong demand "
                  "classes. Delete it and start fresh unless you are certain the "
                  "settings match.", flush=True)
        elif saved_cfg != restart_config:
            diff = {k: (saved_cfg.get(k), restart_config.get(k))
                    for k in set(saved_cfg) | set(restart_config)
                    if saved_cfg.get(k) != restart_config.get(k)}
            raise RuntimeError(
                f"[RESTART] refusing to resume: the coarsening configuration "
                f"changed since {resume_from} was written. demand_index values "
                f"would mean something different, silently corrupting coverage.\n"
                f"  differing keys (saved -> current): {diff}\n"
                f"  Either restore the old settings or delete {resume_from}.")
        else:
            print("[RESTART] coarsening fingerprint matches.", flush=True)

        print(f"[RESTART] Loaded {len(initial_selected)} selected materials")
        print(f"[RESTART] Loaded coverage for {len(initial_coverage)} environments")
        
        # Filter env_indices to only those still needing coverage
        env_indices_original = env_indices.copy()
        env_indices_filtered = np.array([env for env in env_indices 
                               if initial_coverage.get(int(env), 0) < n_coverage])
        
        print(f"[RESTART] Filtered environments: {len(env_indices_original)} -> {len(env_indices_filtered)}")
    else:
        print("Starting fresh run")

   
    #Put the subset of material_vectors in a new temporary table
    time_save_matsubset_start = time.time()
#    #Only select materials, which contain at least one relevant environment
#    conn_vectors.execute(""" 
#              mat_index,
#              indices,
#              counts
#            """,
#            [env_indices.tolist()])

    #Only select materials, which contain at least one relevant environment,
    #and only save the counts of the relevant environments
    # v10: the columns are DEMAND CLASSES, not env_indices. The inner
    # several environments of a material collapse into one demand class, their site
    # counts must ADD, because the material really does contain that many sites of
    # that class.
    #
    # There is deliberately no `WHERE env_idx = ANY(?::INTEGER[])` any more. With
    # min_freq_env == 1 nothing is excluded, so the multi-million-element list
    # parameter that dominated this query in v9 disappears entirely. Materials are
    # dropped only if they contain no mapped environment at all (impossible unless
    # environment_mapping is incomplete), which the JOIN handles.
    if resume_from:
        # Exclude already-selected materials, and keep only materials that still
        # touch a demand class needing coverage.
        #
        # These use HASH ANTI/SEMI JOINs against temp tables rather than the
        # bound-list forms (`mat_index != ALL(?)` /
        # `HAVING list_has_any(list(demand_index), ?)`).
        #
        # DuckDB does optimise the bound-list forms into hash lookups, but the
        # anti-join is still faster; list_has_any against a large
        # array over 200k groups took 0.94s vs 0.11s, and scaled only weakly with
        # array size (0.66/0.70/0.94s for 1k/10k/100k). So the list forms would NOT
        # have hung -- extrapolated to this scale they cost seconds to ~a minute.
        # The joins are simply 7-8x cheaper for free, and the id sets move in as
        # registered DataFrames (bulk columnar transfer) instead of element-by-
        # element SQL list binding.
        #
        # The REAL bug fixed here is the guard immediately below: an empty
        # env_indices_filtered (resuming an already-complete run) produced an empty
        # relevant_mat_vectors and then died inside matrix construction.
        if len(env_indices_filtered) == 0:
            print("[RESTART] every demand class is already covered by the "
                  "restart file; nothing left to select.", flush=True)
            print("[RESTART] NO SUBSET FILE WAS WRITTEN. The restart file holds a "
                  "complete selection; delete it and re-run to regenerate the "
                  "subset from scratch.", flush=True)
            conn_vectors.close()
            return

        conn_vectors.execute("DROP TABLE IF EXISTS v10_sel_prev")
        conn_vectors.execute("CREATE TEMP TABLE v10_sel_prev (mat_index BIGINT)")
        conn_vectors.register('v10_sel_prev_df', pd.DataFrame(
            {'mat_index': np.asarray(initial_selected, dtype=np.int64)}))
        conn_vectors.execute(
            "INSERT INTO v10_sel_prev SELECT mat_index FROM v10_sel_prev_df")

        conn_vectors.execute("DROP TABLE IF EXISTS v10_needy")
        conn_vectors.execute("CREATE TEMP TABLE v10_needy (demand_index INTEGER)")
        conn_vectors.register('v10_needy_df', pd.DataFrame(
            {'demand_index': np.asarray(env_indices_filtered, dtype=np.int64)}))
        conn_vectors.execute(
            "INSERT INTO v10_needy SELECT demand_index FROM v10_needy_df")

        conn_vectors.execute("""
            CREATE OR REPLACE TEMP TABLE v10_inc_resume AS
            SELECT u.mat_index AS mat_index,
                   dm.demand_index AS demand_index,
                   -- CAST, not bare SUM: material_vectors.counts is BIGINT[], so
                   -- SUM() returns HUGEINT, which Arrow exports as
                   -- decimal128(38,0) and numpy turns into an OBJECT array of
                   -- 120-byte Decimals -- ~81 GiB for this dataset's 675.7M
                   -- nonzeros. Result-neutral (verified byte-identical: counts
                   -- are small integers, exact in float64, and
                   -- mat_vecs_from_arrow coerces to float64 anyway); it just
                   -- moves the int->float conversion into DuckDB's C++
                   -- aggregate. The old .fetchall() loader was immune because
                   -- CPython interns the small ints DuckDB hands back.
                   CAST(SUM(u.cnt) AS DOUBLE) AS cnt
            FROM (
              SELECT mv.mat_index,
                     unnest(mv.indices) as env_idx,
                     unnest(mv.counts)  as cnt
              FROM material_vectors mv
              WHERE NOT EXISTS (SELECT 1 FROM v10_sel_prev s
                                WHERE s.mat_index = mv.mat_index)
            ) u
            JOIN demand_mapping dm ON dm.env_index = u.env_idx
            GROUP BY 1, 2
        """)
        conn_vectors.execute("""
            CREATE TEMP TABLE relevant_mat_vectors AS
            SELECT i.mat_index AS mat_index,
                   list(i.demand_index) as indices,
                   list(i.cnt) as counts
            FROM v10_inc_resume i
            WHERE i.mat_index IN (
                SELECT j.mat_index FROM v10_inc_resume j
                WHERE j.demand_index IN (SELECT demand_index FROM v10_needy))
            GROUP BY i.mat_index
            ORDER BY array_length(indices) DESC
        """)
        conn_vectors.execute("DROP TABLE IF EXISTS v10_inc_resume")


    else:
        conn_vectors.execute("""
                CREATE TEMP TABLE relevant_mat_vectors AS
                SELECT
                  mat_index,
                  list(demand_index) as indices,
                  list(cnt) as counts
                FROM (
                  SELECT u.mat_index AS mat_index,
                         dm.demand_index AS demand_index,
                         -- CAST for the same reason as the resume query above:
                         -- SUM() over BIGINT is HUGEINT -> arrow decimal128 ->
                         -- numpy object array of Decimals. Result-neutral.
                         CAST(SUM(u.cnt) AS DOUBLE) AS cnt
                  FROM (
                    SELECT
                      mat_index,
                      unnest(indices) as env_idx,
                      unnest(counts) as cnt
                    FROM material_vectors
                  ) u
                  JOIN demand_mapping dm ON dm.env_index = u.env_idx
                  GROUP BY 1, 2
                )
                GROUP BY mat_index
                ORDER BY array_length(indices) DESC
                """)
    time_save_matsubset_end = time.time()
    print(f"Time for saving relevant mat_ids : {time_save_matsubset_end - time_save_matsubset_start} s", flush=True)
    
    #Transfer the relevant rows to a python dict
    time_matsubset_to_dict_start = time.time()
    if search_algo == 'simple':
        mat_vecs_dict = {int(mat_index): set(indices) for mat_index, indices in conn_vectors.execute("SELECT mat_index, indices FROM relevant_mat_vectors ORDER BY mat_index").fetchall()}
    elif search_algo == 'uniform' or search_algo == 'explore-uniform':
        mat_vecs_dict = {int(mat_index): Counter( dict(zip(list(indices), list(counts))) ) 
                for mat_index, indices, counts in 
                conn_vectors.execute("SELECT mat_index, indices, counts FROM relevant_mat_vectors ORDER BY mat_index").fetchall()}
    elif search_algo == 'uniform-numba':
        # Arrow instead of .fetchall() -- see mat_vecs_from_arrow for why, and for
        # the byte-identity contract it preserves. to_arrow_table() is the current
        # name; fetch_arrow_table() is its long-standing alias, kept for older
        # duckdb builds.
        _res = conn_vectors.execute("SELECT mat_index, indices, counts "
                                    "FROM relevant_mat_vectors ORDER BY mat_index")
        _tbl = (_res.to_arrow_table() if hasattr(_res, 'to_arrow_table')
                else _res.fetch_arrow_table())
        mat_vecs_dict = mat_vec_pool_from_arrow(_tbl)
        del _tbl, _res          # release the Arrow buffers; the slices own copies
        gc.collect()
    time_matsubset_to_dict_end = time.time()

    print(f"Time to load matvecs as dict : {time_matsubset_to_dict_end - time_matsubset_to_dict_start} s", flush=True)
    # sys.getsizeof on a dict reports only the hash table, not the 42.8M tuples
    # and slices hanging off it; the pool's nbytes is the real, complete figure.
    _pool_bytes = (mat_vecs_dict.nbytes if isinstance(mat_vecs_dict, MatVecPool)
                   else sys.getsizeof(mat_vecs_dict))
    print(f"Size of dict : {_pool_bytes/1024/1024/1024} GB")
    
    print(f"mat_vecs_dict is of size {len(mat_vecs_dict)}")
    
    env_indices = set(env_indices)
  
    #TODO: Adapt algorithm to n-coverage and to new mat_vecs_dict
    if search_algo == "uniform" or search_algo == "uniform-numba":
        selected_mat_indices = parallel_uniform_n_coverage(
                env_indices, 
                mat_vecs_dict, 
                n_coverage=n_coverage, 
                n_proc=1,
                uniformity_weight=uniformity_weight,
                cost_alpha=cost_alpha,
                batch_size=batch_size,
                restart_interval=restart_interval,  # Save every 1000 iterations
                restart_path=restart_path,
                initial_selected=initial_selected,
                initial_coverage=initial_coverage,
                restart_config=restart_config,
                enable_dedup=enable_dedup,
                enable_forced_selection=enable_forced_selection
                )

        # Combine with previously selected materials if resuming
        #if resume_from:

    elif search_algo == 'explore-uniform':
        selected_mat_indices = parallel_uniform_n_coverage(
                env_indices, 
                mat_vecs_dict, 
                n_coverage=n_coverage, 
                n_proc=1,
                explore_first=True,
                uniformity_weight=uniformity_weight,
                cost_alpha=cost_alpha,
                enable_dedup=enable_dedup,
                enable_forced_selection=enable_forced_selection
                )
    
    #Postprocessing

    # ---- _and_A: POST-SELECTION REDUNDANCY PRUNE ------------------------------
    # Runs HERE, in main(), rather than inside the selector, for three reasons
    # that only became clear once the tests were written:
    #
    #  1. THE SELECTOR STAYS BYTE-IDENTICAL to v12/v13. Two separate tests assert
    #     that (section A of test_v13_dists_and_A.py, and the byte-identity check
    #     in section H against the module test_v11_per_atom_cost validates). The
    #     prune is post-processing, not selection, and putting it inside would
    #     have broken a property worth keeping.
    #  2. IT WORKS ON RESUMED RUNS. Inside the selector, the materials chosen
    #     before the restart are not rows of the loaded matrix, so a resumed run
    #     could only prune its own suffix -- making the result depend on WHERE the
    #     run was interrupted and breaking the fresh == resumed subset guarantee
    #     that the Phase 0b acceptance rule exists to provide. The tests caught
    #     this. Rebuilding the incidence from DuckDB over just the selected
    #     materials costs ~1 s and prunes fresh and resumed runs identically.
    #  3. IT IS ADJACENT TO THE FLOOR TOP-UP, so the required ordering --
    #     prune first, top up second -- is local and obvious. That ordering is
    #     what lets prune_redundant_materials ignore floors entirely: the top-up
    #     then restores any presence guarantee the prune disturbed.
    if enable_redundancy_prune and selected_mat_indices:
        t_prune = time.time()
        # The selection as a deduplicated, ordered universe. Everything below is
        # decided against THIS list, not against whatever the incidence query
        # happens to return -- see the keep/drop construction.
        _sel = sorted({int(m) for m in selected_mat_indices})
        conn_vectors.execute("DROP TABLE IF EXISTS v10_prune_sel")
        conn_vectors.execute("CREATE TEMP TABLE v10_prune_sel (mat_index BIGINT)")
        conn_vectors.register('v10_prune_df', pd.DataFrame(
            {'mat_index': np.asarray(_sel, dtype=np.int64)}))
        conn_vectors.execute(
            "INSERT INTO v10_prune_sel SELECT mat_index FROM v10_prune_df")
        # Only the SELECTED materials are unnested, so this is ~1e7 rows rather
        # than the ~9e8 of the full incidence.
        _pin = conn_vectors.execute("""
            SELECT u.mat_index, dm.demand_index, SUM(u.cnt) AS cnt
            FROM (SELECT mv.mat_index, unnest(mv.indices) AS env_idx,
                         unnest(mv.counts) AS cnt
                  FROM material_vectors mv
                  JOIN v10_prune_sel s ON s.mat_index = mv.mat_index) u
            JOIN demand_mapping dm ON dm.env_index = u.env_idx
            GROUP BY 1, 2 ORDER BY 1, 2
        """).df()
        # `required` is min(n_coverage, corpus availability) -- the identical
        # expression the selector's target_vector uses, read from the table
        # build_demand_mapping already materialised for its own >= T invariant.
        _req_df = conn_vectors.execute(
            "SELECT demand_index, required FROM v10_class_avail").df()
        _n_cls = int(demand_stats['n_classes'])
        _req = np.zeros(_n_cls, dtype=np.float64)
        _req[_req_df['demand_index'].to_numpy(np.int64)] = \
            _req_df['required'].to_numpy(np.float64)

        _mats = np.sort(_pin['mat_index'].unique().astype(np.int64))
        _mrow = {int(m): i for i, m in enumerate(_mats)}
        _r = _pin['mat_index'].map(_mrow).to_numpy(np.int64)
        _c = _pin['demand_index'].to_numpy(np.int64)
        _d = _pin['cnt'].to_numpy(np.float64)
        _S = sparse.csr_matrix((_d, (_r, _c)), shape=(len(_mats), _n_cls))
        # ATOMS COME FROM material_vectors.counts, NOT from _S.sum(axis=1).
        # Summing the incidence counts an atom ONCE PER DEMAND CLASS it maps to:
        # that equals the atom count only under design 'a' with enable_floors off.
        # With floors as demand classes every token also maps to each floor, so
        # the sum is (1 + n_floors)x the atoms; under design 'd' it is up to one
        # row per rung. Same trap the '[v11] FINAL SUBSET SIZE' query documents
        # and avoids. It would inflate the reported numbers and, under design 'd',
        # order the prune by a quantity that is not monotone in atoms.
        _nat = conn_vectors.execute("""
            SELECT mv.mat_index, list_sum(mv.counts) AS natoms
            FROM material_vectors mv
            JOIN v10_prune_sel s ON s.mat_index = mv.mat_index
        """).df().set_index('mat_index')['natoms']
        _atoms = np.nan_to_num(_nat.reindex(_mats).to_numpy(np.float64),
                               nan=0.0).round().astype(np.int64)
        _n_before = len(_sel)
        _a_before = int(np.nan_to_num(_nat.reindex(_sel).to_numpy(np.float64),
                                      nan=0.0).sum())
        # A selected material can only be MISSING from the incidence if none of
        # its environments reached demand_mapping, which needs an incomplete
        # environment_mapping (build_demand_mapping warns when that happens).
        # Such a material was never evaluated, so it must be KEPT, not quietly
        # lost -- which is what rebuilding the selection from `_mats` alone would
        # have done. Drop only what was proved redundant.
        if len(_mats) != len(_sel):
            print(f"[PRUNE] WARNING: {len(_sel) - len(_mats):,} selected materials "
                  f"have no environment in demand_mapping and were not evaluated; "
                  f"they are KEPT.", flush=True)
        _keep, _npr, _apr, _sw = prune_redundant_materials(
                _S, np.arange(len(_mats), dtype=np.int64), _req, _atoms)
        if _npr:
            _drop = {int(m) for m in _mats[~_keep]}
            selected_mat_indices = [m for m in _sel if m not in _drop]
        print(f"[PRUNE] redundancy prune: {_n_before:,} -> "
              f"{len(selected_mat_indices):,} materials "
              f"(-{_npr:,}, -{100.0*_npr/max(_n_before,1):.2f}%), "
              f"{_a_before:,} -> {_a_before-_apr:,} atoms "
              f"(-{_apr:,}, -{100.0*_apr/max(_a_before,1):.2f}%) "
              f"in {_sw} sweep(s), {time.time()-t_prune:.1f}s", flush=True)
        print(f"[PRUNE]   every dropped material's classes are still covered to "
              f"min(n_coverage, availability) by the survivors. This is largely a "
              f"COMPANION TO cost_alpha={cost_alpha:g}: the same step recovers "
              f"~0-0.3% of atoms at cost_alpha=0 and ~5-8% at 1.0, because only "
              f"the small cells bought for one or two classes get subsumed.",
              flush=True)
        conn_vectors.execute("DROP TABLE IF EXISTS v10_prune_sel")
    elif not enable_redundancy_prune:
        print("[PRUNE] disabled (enable_redundancy_prune=False / V13A_PRUNE=0); "
              "the subset is the greedy's output unchanged.", flush=True)
    else:
        print("[PRUNE] nothing to prune: the selection is empty.", flush=True)

    # Approach F repair, BEFORE anything downstream consumes the selection, so the
    # added materials appear in best_mat_ids, the distribution file and the written
    # subset. Skipped when enable_floors already made the floors demand classes.
    if enable_floor_topup and not enable_floors:
        n_before = len(selected_mat_indices)
        added = topup_floor_coverage(conn_vectors, selected_mat_indices,
                                     floor_sigs=floor_sigs_active)
        if added:
            selected_mat_indices = list(selected_mat_indices) + added
            print(f"[v10] floor top-up total: {n_before:,} -> "
                  f"{len(selected_mat_indices):,} materials "
                  f"(+{len(added):,}, +{100.0*len(added)/max(n_before,1):.3f}%)",
                  flush=True)
        else:
            print("[v10] floor top-up: no materials needed", flush=True)

    #Convert mat_indices back to mat_ids
    best_mat_ids = conn_vectors.execute(""" 
            SELECT mat_id
            FROM material_vectors
            WHERE mat_index = ANY(?)
            """,
            [selected_mat_indices]).fetchnumpy()['mat_id']

    # This translation is a FILTER, so it silently (a) drops any mat_index absent
    # written subset differ from what was actually selected, with no other symptom
    # at all. Check it rather than assume it.
    n_unique_sel = len({int(x) for x in selected_mat_indices})
    if len(best_mat_ids) != n_unique_sel:
        print(f"[v10] WARNING: selection/translation mismatch: "
              f"{n_unique_sel:,} unique selected mat_index -> "
              f"{len(best_mat_ids):,} mat_id "
              f"({n_unique_sel - len(best_mat_ids):,} untranslatable, absent from "
              f"material_vectors?). The written subset is SMALLER than the "
              f"selection and the coverage guarantee may not hold for it.",
              flush=True)
    if len(selected_mat_indices) != n_unique_sel:
        print(f"[v10] NOTE: selection contained "
              f"{len(selected_mat_indices) - n_unique_sel:,} duplicate mat_index "
              f"values (collapsed by the translation).", flush=True)
    
    #Count how many chemical elements are in the subset
    species_subset = set(conn_vectors.execute(""" 
            SELECT DISTINCT unnest(elements) as all_species
            FROM env_freq_db.material_elements
            WHERE mat_id = ANY(?)
            """,
            [best_mat_ids.tolist()]).fetchnumpy()['all_species']
            )
    
    print(f"Current subset contains {len(species_subset)} species : {species_subset}")
    
    # ---------------- v10 verification ----------------
    # The v9 check ("every env_id with frequency >= min_freq_env is present") is
    # meaningless here: min_freq_env is 1, so it would demand all 124M tokens --
    # precisely the forced-inclusion blow-up this version removes. The correct
    # invariant is that every DEMAND CLASS is covered n_coverage times.
    #
    # Selected materials go into a temp table rather than a bound list: the subset
    # can be millions of rows, and `= ANY(?)` on that is the same anti-pattern
    # that made the v9 load query stall.
    conn_vectors.execute("DROP TABLE IF EXISTS v10_selected")
    conn_vectors.execute("CREATE TEMP TABLE v10_selected (mat_index BIGINT)")
    conn_vectors.register('v10_sel_df',
                          pd.DataFrame({'mat_index':
                                        np.asarray(selected_mat_indices, dtype=np.int64)}))
    conn_vectors.execute("INSERT INTO v10_selected SELECT mat_index FROM v10_sel_df")

    # ---------------- v11: atom count of the FINAL subset ----------------
    # The quantity cost_alpha exists to control, reported for the subset that is
    # actually written -- i.e. AFTER the floor top-up, so it matches the .txt file
    # rather than the selector's return value.
    #
    # Counted from material_vectors.counts directly, NOT through demand_mapping.
    # Every atom carries exactly one env token, so SUM(list_sum(counts)) over the
    # demand_mapping would multi-count under coarsening_design='d' (one row per
    # token per active ancestor) and under enable_floors (one extra row per token
    # per floor), i.e. exactly where the site_count used by the greedy stops being
    # an atom count. This number is therefore correct for every configuration.
    #
    # DISTINCT because the mat_index list can in principle carry duplicates (the
    # translation below collapses them and says so); double-counting a material's
    # atoms here would silently inflate the headline figure.
    n_sel_mats, n_sel_atoms = conn_vectors.execute("""
        SELECT COUNT(*), SUM(natoms) FROM (
            SELECT mv.mat_index, list_sum(mv.counts) AS natoms
            FROM material_vectors mv
            JOIN (SELECT DISTINCT mat_index FROM v10_selected) s
              ON s.mat_index = mv.mat_index)
    """).fetchone()
    n_sel_atoms = int(n_sel_atoms or 0)
    n_sel_mats = int(n_sel_mats or 0)
    print(f"[v11] FINAL SUBSET SIZE: {n_sel_atoms:,} atoms in {n_sel_mats:,} "
          f"materials (mean {n_sel_atoms/max(n_sel_mats,1):.2f} atoms/material)",
          flush=True)
    # Atoms spent per demand class covered. At n_coverage=1 under design 'a' each
    # atom supplies exactly one class, so 1.00 is the unreachable floor and this
    # ratio is the cost-efficiency of the selection -- the headline number to
    # compare across cost_alpha values.
    print(f"[v11]   atoms per demand class: "
          f"{n_sel_atoms/max(int(demand_stats['n_classes']),1):.2f} "
          f"over {int(demand_stats['n_classes']):,} classes", flush=True)

    conn_vectors.execute("""
        CREATE OR REPLACE TEMP TABLE v10_subset_cov AS
        SELECT dm.demand_index, SUM(u.cnt) AS n_sites
        FROM (
            SELECT mv.mat_index,
                   unnest(mv.indices) AS env_idx,
                   unnest(mv.counts)  AS cnt
            FROM material_vectors mv
            JOIN v10_selected s ON s.mat_index = mv.mat_index
        ) u
        JOIN demand_mapping dm ON dm.env_index = u.env_idx
        GROUP BY 1
    """)
    # The contract the SELECTOR enforces is target = min(n_coverage, achievable),
    # not n_coverage flat. Checking against n_coverage alone raises a FALSE alarm on
    # any class whose total site availability is below n_coverage. Non-catch-all
    # classes cannot hit that (availability >= T >= n_coverage+1), but per-element
    # catch-all classes (rung -1) carry NO support guarantee. A rung -1 class
    # with availability below n_coverage is fully covered against the capped
    # target, but would be reported as
    # UNDER-COVERED, with zero genuine shortfalls in the run. Dormant at
    # n_coverage=1, where availability >= 1 makes min(n_coverage, avail) == 1.
    #
    # v10_class_avail (demand_index, rung, avail, required, is_catch, is_floor) was
    # materialised by build_demand_mapping, which needed the identical aggregate for
    # its >= T invariant. Reading it here rather than rebuilding it avoids a second
    # full pass over the >=124M-row demand_mapping x v10_tok join at the point of
    # peak memory, and keeps the reported `required` provably the same expression
    # the filter below uses.
    n_classes_total = int(demand_stats['n_classes'])
    n_cov_ok, n_capped_cls = conn_vectors.execute(f"""
        SELECT COUNT(*) FILTER (WHERE COALESCE(v.n_sites, 0) >= a.required),
               COUNT(*) FILTER (WHERE a.avail < {n_coverage})
        FROM v10_class_avail a
        LEFT JOIN v10_subset_cov v USING (demand_index)
    """).fetchone()
    print(f"[v10] demand classes covered >= min(n_coverage({n_coverage}), "
          f"availability): {n_cov_ok:,} / {n_classes_total:,}", flush=True)
    if n_capped_cls:
        print(f"[v10]   ({n_capped_cls:,} of them have fewer than n_coverage sites "
              f"in the whole corpus, so their target is capped at availability -- "
              f"these are per-element catch-all classes, not a shortfall)",
              flush=True)
    if n_cov_ok < n_classes_total:
        # Enumerate from demand_mapping, NOT from v10_class: when enable_floors is
        # on, the floor classes (rungs FLOOR_RUNG_BASE - j, i.e. -100 down to -103
        # with all four floors active) exist only in demand_mapping, so listing from
        # v10_class would count a floor shortfall above but fail to show it here.
        # v10_class is joined only to label the signature.
        missing = conn_vectors.execute("""
            SELECT a.rung, c.sig, a.avail AS availability,
                   COALESCE(v.n_sites, 0) AS n_sites, a.required
            FROM v10_class_avail a
            LEFT JOIN v10_class c USING (demand_index)
            LEFT JOIN v10_subset_cov v USING (demand_index)
            WHERE COALESCE(v.n_sites, 0) < a.required
            LIMIT 20
        """).df()
        print(f"[v10] UNDER-COVERED CLASSES (first 20 of "
              f"{n_classes_total - n_cov_ok:,}):", flush=True)
        print(missing.to_string(index=False), flush=True)
    else:
        print("[v10] All demand classes satisfied.", flush=True)

    # Informational: how much of the RAW vocabulary ended up present. This is
    # expected to be far below 100% -- rare tokens are represented by their
    # coarsened class, not necessarily by themselves (design 'd' especially).
    raw_present = conn_vectors.execute("""
        SELECT COUNT(DISTINCT u.env_idx)
        FROM (
            SELECT unnest(mv.indices) AS env_idx
            FROM material_vectors mv
            JOIN v10_selected s ON s.mat_index = mv.mat_index
        ) u
    """).fetchone()[0]
    print(f"[v10] distinct raw env_ids present in subset: {raw_present:,} "
          f"of {demand_stats['n_tokens']:,} "
          f"({100.0*raw_present/max(demand_stats['n_tokens'],1):.2f}%)", flush=True)

    # Approach F as a cheap post-check: are all coarse element/geometry
    # combinations represented, even though coarse rungs merge them?
    # Reported over ALL floors, including any that were not enforced this run: an
    # unenforced floor's gap is exactly the number needed to decide whether to
    # enforce it next time, and measuring it costs one COUNT DISTINCT.
    # A floor counts as enforced only if it was SELECTED and an enforcement route
    # actually ran. With enable_floors and enable_floor_topup both off, nothing was
    # promised, so a gap is expected and must not be labelled a bug.
    enforced_names = ({n for n, _ in floor_sigs_active} if floors_enforced
                      else set())
    for fname, fexpr in ALL_FLOOR_SIGS_FOR_REPORT:
        tot, got = conn_vectors.execute(f"""
            SELECT
              (SELECT COUNT(DISTINCT ({fexpr})) FROM v10_tok),
              (SELECT COUNT(DISTINCT ({fexpr})) FROM v10_tok t
               WHERE t.env_index IN (
                   SELECT unnest(mv.indices) FROM material_vectors mv
                   JOIN v10_selected s ON s.mat_index = mv.mat_index))
        """).fetchone()
        if fname in enforced_names:
            flag = "OK" if got == tot else f"MISSING {tot-got}  <-- BUG, floor was enforced"
            mode = "enforced"
        else:
            # Not a defect: nothing promised this floor. The gap is the price of
            # having it switched off, printed so it never has to be re-measured.
            flag = "OK" if got == tot else f"gap {tot-got} (not enforced)"
            mode = "REPORT  "
        print(f"[v10] floor {fname:14s} [{mode}]: {got:,} / {tot:,} present   {flag}",
              flush=True)

    #env_ids_in_subset_all = conn_vectors.execute("""
    #        """, [best_mat_ids.tolist()]).fetchnumpy()['env_id']



#    env_id_counts_detailed = conn_vectors.execute("""
#           SUM(unnested_counts.count) as total_count,
#           COUNT(DISTINCT m.mat_id) as mat_count
#""", [best_mat_ids.tolist()]).fetchall()








    env_id_counts_detailed = conn_vectors.execute("""
        SELECT em.env_id,
               SUM(paired.count_val) as total_count,
               COUNT(DISTINCT paired.mat_id) as mat_count,
               MAX(cf.frequency) as original_count
        FROM (
            SELECT
                m.mat_id,
                unnest(m.indices) as env_index,
                unnest(m.counts) as count_val
            FROM material_vectors m
            WHERE m.mat_id = ANY(?)
        ) as paired
        JOIN environment_mapping em ON paired.env_index = em.env_index
        LEFT JOIN env_freq_db.chemenv_frequency_table cf ON em.env_id = cf.env_id
        GROUP BY em.env_id
    """, [best_mat_ids.tolist()]).fetchall()


    # For the distribution file
    env_ids_found = np.array([row[0] for row in env_id_counts_detailed], dtype=str)
    total_counts = np.array([row[1] for row in env_id_counts_detailed], dtype=int)
    mat_counts = np.array([row[2] for row in env_id_counts_detailed], dtype=int)
    # Occurrences of each env_id in the original (full) set; 0 if absent from the
    # frequency table. Compare with total_count to see which envs were sampled
    # below n_coverage and whether that was because too few existed overall.
    original_counts = np.array([row[3] if row[3] is not None else 0
                                for row in env_id_counts_detailed], dtype=int)

    # Sort by descending total count
    sorted_indices = np.argsort(-total_counts)
    env_ids_found = env_ids_found[sorted_indices]
    total_counts = total_counts[sorted_indices]
    mat_counts = mat_counts[sorted_indices]
    original_counts = original_counts[sorted_indices]

    # max() on an empty sequence raises; an empty result set is reachable if the
    max_env_id_len = max((len(s) for s in env_ids_found), default=1)
    env_id_distribution = np.zeros(len(env_ids_found),
                               dtype=[('env_id', f'U{max_env_id_len}'),
                                      ('total_count', 'i8'),
                                      ('mat_count', 'i8'),
                                      ('original_count', 'i8')])

    env_id_distribution['env_id'] = env_ids_found
    env_id_distribution['total_count'] = total_counts
    env_id_distribution['mat_count'] = mat_counts
    env_id_distribution['original_count'] = original_counts

    np.savetxt('env_ids_found_distribution.txt', env_id_distribution,
           fmt='%s %d %d %d',
           header='env_id total_count mat_count original_count')





    #Find by how many materials a given env_id is represented
    #env_id_counts = conn_vectors.execute("""
    #""", [best_mat_ids.tolist()]).fetchall()
    # Convert to dictionary

    #  v10 verification block above for the correct demand-class invariant.)

    #Save frequency distribution of env_ids in file
    #Sort by descending frequency
    #Combine to array and save in txt file
    # Map the mat_counts to your existing env_ids

    
    # Every parameter that CHANGES THE RESULT must appear here, or two different
    # runs silently overwrite each other's subset. support_metric and n_coverage
    # both materially change the selection (measured on the fixture: n_coverage
    # 1/2/3 -> 1194/1325/1589 materials; support 'sites' vs 'materials' -> 3244 vs
    # 961 demand classes), yet neither was in the name before.
    #
    # v11 adds cost_alpha and uniformity_weight, both of which were missing and
    # both of which change the selection. cost_alpha is the whole point of v11
    # materials), so an alpha sweep run in one directory silently destroyed its
    # own earlier arms -- reproduced before the fix: the 332-material alpha=0
    # subset was overwritten by the 620-material alpha=1 one, leaving a single
    # file. uniformity_weight is the same class of omission, inherited from v10.
    #
    # The tags are emitted UNCONDITIONALLY, not only when non-default. A v11
    # cost_alpha=0 run is NOT the same subset as a v10 run (Phase 0 and Phase 0b
    # both move tie resolution), so suppressing the tag at the default value
    # would let those two collide instead -- the same bug one level up.
    # ':g' keeps the tag short and stable: 0.0 -> '0', 0.5 -> '0.5', 1.0 -> '1'.
    #
    # NOTE, deliberately NOT solved here: env_ids_found_distribution.txt and
    # restart.npz are still fixed names, so parallel arms of a sweep must each
    # run in their own working directory regardless of this filename. A stale
    # restart.npz from another arm is at least refused loudly, because
    # restart_config carries cost_alpha.
    # v13 MARKER, deliberately in the filename. v12 and v13 build DIFFERENT demand
    # spaces from the same knobs, so without this they write the same filename and a
    # comparison run in one directory silently overwrites one of the two results.
    # refuses a foreign restart file rather than resuming against the wrong ladder.)
    # _and_A: its own marker, for exactly the reason the v13 marker exists. An
    # _and_A run and a v13 run with identical knobs build DIFFERENT demand spaces
    # overwrite the other's result. The two optional structural knobs are in the
    # name too, since they change the demand space rather than just the run.
    #
    # v14 CHANGES THE MARKER to 'v14andAdeep'. This is not cosmetic and it is the
    # whole reason the marker convention exists: v14 and _and_A take the SAME
    # vocabulary and the SAME knobs and differ only in rung order, so they produce
    # different subsets under filenames that would otherwise be character-for-
    # character identical -- the one case where an overwrite is completely silent
    # and the two results are meant to be compared. ('deep' = the modifiers survive
    # deep into the ladder instead of being dropped first.)
    out_name = (f'best_subset_coarsened_v14andAdeep_T{coarsening_T}'
                f'{"_R0full" if enable_full_token_rung else ""}'
                f'{"_asymfloor" if enable_asym_floor else ""}'
                f'_design{coarsening_design}'
                f'_sup{support_metric}'
                f'_ncov{n_coverage}'
                f'_alpha{cost_alpha:g}'
                f'_uw{uniformity_weight:g}'
                f'{"_floors" if enable_floors else ""}'
                f'{"_topup" if (enable_floor_topup and not enable_floors) else ""}'
                # Marked only when OFF, so the default filename is unchanged and
                # stays comparable with the subsets already produced.
                f'{"" if enable_dist_floor else "_nodistfloor"}'
                f'{"" if enable_redundancy_prune else "_noprune"}'
                f'.txt')
    np.savetxt(out_name, best_mat_ids, fmt='%s')
    print(f"Subset written to {out_name} ({len(best_mat_ids):,} materials)", flush=True)
    print(f"Subset search finished", flush=True)
    
    conn_vectors.close()

if __name__ == '__main__':
    main()
