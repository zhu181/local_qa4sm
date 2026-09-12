# QA4SM Local Validation

Local command-line tool for validating soil moisture products against
reference datasets. Part of the [QA4SM](https://qa4sm.eu) ecosystem.

[![Python](https://img.shields.io/badge/python-3.13%2B-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()
[![uv](https://img.shields.io/badge/managed%20with-uv-purple.svg)]()

## What is this

`qa4sm-validate` is a CLI companion to the [QA4SM web
service](https://qa4sm.eu) that runs the same validation framework
locally. It uses the [pytesmo](https://github.com/TUW-GEO/pytesmo)
validation framework (vendored as a git submodule under `pytesmo/`)
and adds:

- A Dask-based parallel backend that streams results in batches
  (default) instead of accumulating everything in the client.
- Optional GPU acceleration via CuPy inside the Dask workers
  (`-Gpu` / `QA4SM_USE_GPU=1`).
- A classic `ThreadPoolExecutor` path (escape hatch for environments
  where Dask doesn't fit).
- Robust handling of corrupt input files (per-gpi read retries),
  transient worker failures (per-job retries inside Dask batches),
  and stale crash-resume caches (zarr quarantine on repeated load
  failures).
- A preset-driven PowerShell wrapper (`run_validation.ps1`) that
  handles bbox overlays, log file naming, memory-budget persistence,
  and interactive mode on Windows.

## Features

- **Dask streaming by default** — `qa4sm-validate` uses
  `pytesmo.parallel.DaskParallelExecutor` and streams per-batch
  results back to the run netCDF so neither client nor workers
  hold the full result set in memory.
- **Crash-resume batch cache** — completed batches are persisted
  client-side to `outputs/.dask_batch_cache/`. Re-runs with the
  same config skip done batches and only recompute missing ones.
- **Corrupt-input read retries** — `Validation.calc` wraps
  `data_manager.get_data()` in a retry loop; on `_is_corruption_error`
  the read is retried up to `QA4SM_MAX_READ_RETRIES` times before
  the gpi is recorded as no-data and the run continues.
- **GPU acceleration** — pairwise metrics, triple collocation, and
  bootstrap confidence intervals dispatch to CuPy automatically when
  `QA4SM_USE_GPU=1` and CUDA is available.
- **Classic escape hatch** — `QA4SM_USE_CLASSIC=1` switches to the
  per-job `ThreadPoolExecutor` path with coalesced netCDF writes and
  corruption-tolerant reopen-validation.
- **ISMN-safe** — the ISMN reader is not fork/thread-safe; the
  validator automatically limits `n_workers=1` when the reference
  reader is ISMN.

## Installation

Requires **Python ≥ 3.13** and [uv](https://docs.astral.sh/uv/).

```bash
# Clone with submodules — pytesmo is a pinned fork and is REQUIRED
git clone --recursive https://github.com/zhu181/local_qa4sm.git
cd local_qa4sm

# Install dev + runtime deps into .venv/
uv sync --extra dev
```

If you forgot `--recursive`:

```bash
git submodule update --init --recursive
```

### GPU acceleration

Optional. Only install if you have an NVIDIA GPU and CUDA ≥ 12.x:

```bash
# CUDA 13.x (recommended for newer toolkits)
uv pip install cupy-cuda13x==14.1.1
# Re-pin numpy afterwards — cupy bumps it to 2.5.x which breaks numba
uv pip install "numpy==2.4.6"
```

> **Never install both `cupy-cuda12x` and `cupy-cuda13x`** — they
> share the `cupy` namespace directories and clobber each other's
> bundled CUDA DLLs. Recovery: uninstall both and reinstall only the
> one that matches your toolkit.

Without CUDA / CuPy the validator still runs — Dask uses NumPy
metrics transparently.

## Quick start

### Dry-run (parse config, no compute)

```bash
uv run qa4sm-validate validator/validation_run.template.json --dry-run
```

### Real run on a preset

The `presets/` directory holds ready-made configs. On Windows:

```powershell
.\run_validation.ps1 -List                       # show available presets
.\run_validation.ps1 spl3smpe                    # full-preset run, Dask default
.\run_validation.ps1 ismn-spl3 -BBox 33,110,38,115 -Gpu
.\run_validation.ps1 spl3 -Classic -MaxWorkers 4 # escape hatch
.\run_validation.ps1 -Interactive                # guided setup
```

Cross-platform (Linux / macOS / WSL):

```bash
uv run qa4sm-validate presets/spl3smpe.json
uv run qa4sm-validate presets/ismn-spl3.json \
    --override validation_run.interval_from=2017-01-01 \
    --override validation_run.interval_to=2021-12-31
```

The script accepts a `storage_path` override via `-StoragePath` /
`--storage-path`, or the JSON field of the same name. The CLI fails
fast if the storage path doesn't exist.

## Behaviour matrix

Dask is now the default parallel backend; classic is an opt-in
escape hatch; GPU is an opt-in metric accelerator on top of Dask.

| `QA4SM_USE_DASK` | `QA4SM_USE_GPU` | `QA4SM_USE_CLASSIC` | Path |
|---|---|---|---|
| `1` (default) | `0` (default) | `0` (default) | Dask streaming + NumPy metrics |
| `1` | `1` | `0` | Dask streaming + GPU metrics (CuPy) |
| `0` | `1` | `0` | Dask streaming + GPU metrics (GPU implies Dask) |
| `0` | `0` | `1` | Classic `ThreadPoolExecutor` (escape hatch) |
| `0` | `0` | `0` | Dask streaming + NumPy metrics (default) |

Run-validation.ps1 shortcuts:

| Flag | Effect |
|---|---|
| `-Gpu` | `QA4SM_USE_GPU=1` + `QA4SM_USE_DASK=1` |
| `-Dask` | `QA4SM_USE_DASK=1` (no-op on default runs) |
| `-Classic` | `QA4SM_USE_CLASSIC=1`, unsets `QA4SM_USE_DASK` |

## Configuration

Validation runs are defined in a JSON file. See
`validator/validation_run.template.json` for the full schema. Key
fields:

| Field | Description |
|---|---|
| `validation_run.dataset_configurations[].dataset.reader` | Reader class name (e.g. `"SMAPL3_V9Reader"`, `"ISMN_Interface"`, `"GriddedNcOrthoMultiTs"`) |
| `storage_path` | Path to local data directory (CLI fails fast if missing) |
| `interval_from` / `interval_to` | Validation time range (ISO date) |
| `scaling_method` | `"none"`, `"min_max"`, `"linreg"`, `"mean_std"`, `"cdf_beta_match"` |
| `temporal_matching` | Tolerance window in hours (e.g. `12`) |
| `temporal_stability` | Enable stability / intra-annual metrics |
| `tcol` | Enable triple collocation (`snr` / `beta` / `err_std`) |
| `bootstrap_tcol_cis` | Bootstrap CIs on tcol metrics (slower; needs GPU for big runs) |
| `intra_annual_metrics` | Enable intra-annual subsets |
| `stability_metrics` | Enable stability subsets |
| `anomalies` | `"none"` or `"climatology"` |
| `upscaling_method` | `"none"` or `"nearest"` |

Readers are selected by the `reader` string in the config, not by
the dataset `short_name`. Adding a new dataset means writing a
reader factory in `validator/readers.py` and registering it in
`_READER_REGISTRY`.

## CLI

```
qa4sm-validate [config] [--dry-run] [--log-level LEVEL]
                [--log-file PATH] [--max-workers N]
```

- `--dry-run` — parse and validate config without executing.
- `--log-level` — `DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL`
  (default `INFO`).
- `--log-file PATH` — write logs to file in addition to console.
- `--max-workers N` — override parallel worker count (env:
  `QA4SM_MAX_PARALLEL_WORKERS`, default `min(2, CPU count)`).
  ISMN forces `1` in both paths because the reader isn't
  fork/thread-safe.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `QA4SM_USE_DASK` | `1` | Use the Dask streaming path |
| `QA4SM_USE_CLASSIC` | `0` | Escape hatch: classic `ThreadPoolExecutor` (requires `QA4SM_USE_DASK=0`) |
| `QA4SM_USE_GPU` | `0` | Enable CuPy/NumPy-metric acceleration inside Dask workers (implies Dask) |
| `QA4SM_MAX_PARALLEL_WORKERS` | `min(2, CPU count)` | Dask cluster size / classic thread-pool size |
| `QA4SM_HEARTBEAT_INTERVAL_SECONDS` | `60` | Classic-path heartbeat log interval |
| `QA4SM_DASK_MEMORY_LIMIT` | `""` (4 GB/worker default) | Total budget, divided equally per worker (e.g. `"16GB"`) |
| `QA4SM_DASK_BATCH_SIZE` | `100` | Gpis per Dask batch — drop to `25` on big ISMN×SMAP runs |
| `QA4SM_MAX_READ_RETRIES` | `3` | Per-gpi retries on corrupt-input-file errors |
| `QA4SM_READ_RETRY_DELAY_SECONDS` | `1.0` | Sleep between read retries |
| `QA4SM_DASK_BATCH_RETRIES` | `3` | Dask-level retries for whole batch tasks |
| `QA4SM_THREAD_TIMEOUT_SECONDS` | `0` (disabled) | Per-job timeout on the classic path |
| `QA4SM_CACHE_LOAD_RETRIES` | `3` | Consecutive zarr load failures before quarantine |

`run_validation.ps1 -MemoryLimit <SIZE>` accepts a bare number
(e.g. `8`) and normalizes it to `8GB`; the value is persisted to
`outputs/.dask_memory_limit` so later runs without the flag reuse it.
Without any persisted value, the script defaults to `16GB` total.

## Outputs

Each run writes a timestamped directory under `outputs/`:

```
outputs/<run_id>/
  <run_id>.nc           # run netCDF (consolidated; safe to open read-only)
  <run_id>_plots/       # qa4sm-reader-generated PNGs (overview maps,
                        #   scatter, box, tcol status barplot)
  *.log                 # per-run log if -LogFile was used
```

Stale `.nc` files in the run dir are auto-cleaned on re-run. A
persistent crash-resume cache lives in
`outputs/.dask_batch_cache/<sha1(config)>_b<batch_size>/`.

Logs from `run_validation.ps1` invocations go to
`logs/validation_run_<ts>.log` by default (script-level events +
validation output in one file).

## Presets

`presets/validation_run.<name>.json` are ready-made configs for
common dataset combinations (e.g. `spl3smpe`, `ismn-spl3`,
`ismn-spl3-nsmc`, `spl3smpe_nsmcsmc`). Run
`.\run_validation.ps1 -List` to see them all. Each preset can be
narrowed with `-BBox <lat0,lon0,lat1,lon1>` (overlaid onto a
temporary copy; full-preset runs stay small and fast).

## Architecture

```
validator.cli.main
  └─ orchestrator.parse_validation_run_config
       └─ validator.validation.run_validation
            ├─ _dask_requested()
            │   ├─ Dask path  →  pytesmo.Validation.calc(parallel='dask',
            │   │                batch_callback=..., max_read_retries=...)
            │   └─ Classic    →  ThreadPoolExecutor loop with write
            │                    coalescing + corruption reopen
            └─ _post_process_run  →  qa4sm_reader netcdf transcription
                                     →  plots via qa4sm_reader.graphics
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full
request lifecycle, the parallel-backend decision matrix, and the
pytesmo integration boundary (which symbols we import, which
`Validation.calc()` kwargs we pass, why we monkey-patch
`qa4sm_reader.netcdf_transcription.DATASETS` for tcol metric keep
lists, and where the streaming batch callback hooks in).

## Submodules

This repository embeds [pytesmo](https://github.com/TUW-GEO/pytesmo)
as a git submodule under `pytesmo/`. It is a pinned fork with GPU
acceleration, Dask streaming + crash-resume, and corrupt-input read
retries. pytesmo is licensed separately under BSD-3-Clause — see
`pytesmo/LICENSE.txt`. Always run `git submodule update --init
--recursive` after cloning.

`qa4sm-reader` and `qa4sm-preprocessing` are NOT vendored. They are
installed from PyPI as `qa4sm_reader` and `qa4sm_preprocessing`;
optional local clones can be set up via `scripts/setup.ps1`.

## Development

```bash
# Sync deps
uv sync --extra dev

# Run the parent test suite (uses tests/, no external data)
uv run pytest

# Lint + format
uv run ruff check .
uv run ruff format .

# Lint check only (CI mode)
uv run ruff format --check .
```

The pytesmo submodule has its own venv and test suite. To run its
tests:

```bash
cd pytesmo
uv run pytest --ignore=tests/test_docs/test_examples.py \
              --ignore=tests/test_validation_framework/test_adapters.py \
              --ignore=tests/test_validation_framework/test_validation.py \
              --ignore=tests/test_io_formats.py \
              --ignore=tests/test_timedate/test_julian.py
```

(The five `--ignore` flags deselect modules that depend on optional
reader packages or notebook libraries not in the default venv.)

### Conventions

- Conventional commits: `feat:`, `fix:`, `chore:`, `docs:`, `style:`,
  `test:`, `refactor:`.
- Branch naming: `feat/*`, `fix/*`, `chore/*`, `docs/*`.
- AGENTS.md is the agent memory file; update it when behaviour
  changes so future agent sessions stay in sync.
- Don't edit `pytesmo/` from the parent — commit inside the
  submodule's own branch, then bump the pointer from here.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full guide.

## Troubleshooting

**`QA4SM_USE_GPU=1` but no GPU detected.**
The validator logs a warning and stays on the Dask path with NumPy
metrics — the run completes; no fallback to the classic path. Install
`cupy-cuda13x==14.1.1` and `numpy==2.4.6` to enable GPU.

**`KilledWorker` on a 100-gpi ISMN×SMAP batch.**
Peak worker memory exceeded. Drop `QA4SM_DASK_BATCH_SIZE=25` (or
smaller) and/or raise `QA4SM_DASK_MEMORY_LIMIT`.

**Classic-path `STATUS_ACCESS_VIOLATION` inside `netcdf-*.dll`.**
The classic path already coalesces writes into 10-job batches and
reopens the netCDF between flushes to detect (and delete) silently
corrupted files. If you still see this, switch to the Dask path
(`-Dask` / unset `QA4SM_USE_CLASSIC`).

**Stale crash-resume cache.**
Delete `outputs/.dask_batch_cache/<sha1(config)>_b<batch_size>/` to
force a clean recompute, or quarantine a single batch by deleting
just that `batch_<i>.zarr/` directory. Corrupt batches that fail to
load `QA4SM_CACHE_LOAD_RETRIES` times are auto-quarantined under
`<cache>/.quarantine/`.

**`uv sync` fails behind a mirror.**
`uv.lock` currently points at `pypi.tuna.tsinghua.edu.cn`. Set
`UV_INDEX_URL=https://pypi.org/simple` (or revert the `chore(deps):`
commit) if you're outside that mirror's reach.

## License

[MIT](LICENSE) for this repository. The vendored
[`pytesmo/`](pytesmo/) submodule is BSD-3-Clause; attribution and
upstream license terms are preserved at `pytesmo/LICENSE.txt`.

## Acknowledgements

- [QA4SM](https://qa4sm.eu) — the validation service this CLI
  mirrors.
- [pytesmo](https://github.com/TUW-GEO/pytesmo) — the validation
  framework (TU Wien, Department of Geodesy and Geoinformation).
- [qa4sm-reader](https://pypi.org/project/qa4sm-reader/) and
  [qa4sm-preprocessing](https://pypi.org/project/qa4sm-preprocessing/)
  — output transcription + reader ecosystem.

## Related projects

- [qa4sm.eu](https://qa4sm.eu) — the web service.
- [pytesmo](https://github.com/TUW-GEO/pytesmo) — the upstream
  framework.
- [qa4sm-reader](https://pypi.org/project/qa4sm-reader/) — NetCDF
  reading + plotting.
- [qa4sm-preprocessing](https://pypi.org/project/qa4sm-preprocessing/)
  — data reader ecosystem.