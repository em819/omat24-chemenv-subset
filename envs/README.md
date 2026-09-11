# Environments

Four Python environments are needed. Each directory here records one of them
exactly as it existed on the cluster when the published results were produced.

| Id | Directory | Python | Used by |
|---|---|---|---|
| **A** | `envA-chemenv/` | 3.10.13 | 01, 02, 03, 04, 05, 10.2 |
| **B** | `envB-mace/` | 3.12.12 | 06, 07, 08, 09/matpes, 10.1, 10.3, 10.4 |
| **C** | `envC-matcalc/` | 3.12.3 | 09/matcalc |
| **D** | `envD-mliparena/` | 3.12.3 | 09/stability, 09/nebdft2k |

Every dependency in all four resolves to a released PyPI version — there are no
git checkouts or local-path installs, so the `requirements.txt` files are
complete.

## Rebuilding

### Env A (conda)

Conda supplies only the interpreter and system libraries here; everything
scientific is pip-installed on top. Either rebuild the whole thing from the
pinned export:

```bash
conda env create -f envA-chemenv/environment.yml
```

or build a minimal interpreter and layer the pip requirements on it — more
portable across platforms, since the pinned build strings in `environment.yml`
may not exist on another OS:

```bash
conda create -n chemenv python=3.10.13
conda activate chemenv
pip install -r envA-chemenv/requirements.txt
```

`environment-from-history.yml` records only the packages that were explicitly
requested, which is the right starting point if you are adapting rather than
reproducing.

> `mpi4py` and `h5py` are pip wheels here, built against the OpenMPI and HDF5
> supplied by the cluster modules below. If you rebuild against a different MPI
> you will need to reinstall both from source.

> **`fairchem-core` is only needed to read the OMat24 ASE databases.** If your
> copy of OMat24 is extxyz, stage 01 never imports it and you can leave it out —
> which also drops `torch` and `torch-geometric` from env A. Everything else in
> stages 01–05 needs only pymatgen, ase, numpy, mpi4py, h5py + pytables, duckdb,
> pyarrow and numba.

### Envs B, C, D (virtualenvs)

```bash
python3.12 -m venv envB && source envB/bin/activate
pip install -r envB-mace/requirements.txt
```

`pyvenv.cfg` records the interpreter each venv was created from. Envs B and D
were created with [uv](https://github.com/astral-sh/uv) and contain no `pip`;
`uv pip install -r …` works equally well.

## Cluster modules

These are load-bearing and invisible to pip and conda. They are recorded in the
`module load` lines of the SLURM scripts, which are the authoritative version —
this table is a summary. Module names are [MeluXina](https://docs.lxp.lu/)
EasyBuild names; substitute your site's equivalents.

| Stages | Modules | Why |
|---|---|---|
| 01, 10.2 | `env/release/2024.1`, `OpenMPI/5.0.3-GCC-13.3.0`, `HDF5/1.14.5-gompi-2024a` | `mpi4py` and parallel `h5py` are linked against these |
| 06 | `env/release/2024.1` only | CUDA is loaded **deliberately not** — mapping the CUDA libraries into 128 fork workers exhausts the commit limit |
| 07, 08, 09/matpes, 10.1, 10.3, 10.4 | `env/release/2024.1`, `CUDA/12.6.0` | torch + `cuequivariance` acceleration |
| 09/matcalc, 09/stability, 09/nebdft2k | `env/release/2024.1`, `Python/3.12.3-GCCcore-13.3.0`, `CUDA/12.6.0` | these venvs were built on the module Python |
| 02, 03, 04, 05 | none — conda env only | pure DuckDB / numba, no MPI, no GPU |

## Key versions

Useful for a quick compatibility check:

| Package | A | B | C | D |
|---|---|---|---|---|
| python | 3.10.13 | 3.12.12 | 3.12.3 | 3.12.3 |
| numpy | 2.2.6 | 2.4.2 | 2.4.6 | 2.5.1 |
| torch | 2.6.0 | 2.5.0 | 2.12.0+cu126 | 2.13.0+cu126 |
| ase | 3.25.0 | 3.27.0 | 3.28.0 | 3.29.0 |
| pymatgen | 2025.6.14 | — | 2026.5.4 | 2026.5.4 |
| mace-torch | — | 0.3.14 | 0.3.16 | 0.3.16 |
| cuequivariance | — | 0.8.1 | — | 0.8.1 |
| duckdb / numba | 1.4.2 / 0.61.2 | — | — | — |
| mpi4py / h5py | 4.1.0 / 3.14.0 | — | — | — |
| matcalc / phonopy | — | — | 0.5.1 / 4.1.0 | — |
| mlip-arena / litraj | — | — | — | 0.2.0 / 0.1 |

The model was **trained** with `mace-torch` 0.3.14 (env B) and **benchmarked**
partly with 0.3.16 (envs C and D). That is not a problem — the exported
`.model` file is self-contained — but it is worth knowing if you compare
timings.
