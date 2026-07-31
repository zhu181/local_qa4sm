# QA4SM Local Validation

## Quick start
- Run: `uv run qa4sm-validate <config.json>`
- Or: `uv run python -m validator.cli <config.json>`
- Dry-run: add `--dry-run` flag

## Config
- JSON config file, see `validator/validation_run.template.json`
- Reader field selects the data reader class (e.g. `"SMAPL3_V9Reader"`, `"ISMN_Interface"`, `"GriddedNcOrthoMultiTs"`)
- `storage_path` must point to local data directories (use your own paths)

## Architecture (4 packages)
| Package | Path | Role |
|---------|------|------|
| `validator` | `./validator/` | CLI + validation engine (primary package) |
| `qa4sm-gpu-validation` | `./qa4sm-gpu-validation/` | Standalone pytesmo-compatible batched GPU metric calculators (`qa4sm_gpu_validation`) |
| `qa4sm-reader` | `./qa4sm-reader/` | NetCDF result reading & plotting (separate git history) |
| `qa4sm-preprocessing` | `./qa4sm-preprocessing/` | Data preprocessing readers (separate git history) |

## GPU / parallelism
- `qa4sm-gpu-validation` owns: `qa4sm_gpu_validation/gpu_backend.py` (backend singleton), `gpu_metrics.py` (`BatchedPairwiseMetrics`, `BatchedTCAMetrics`, `GPUBatchedValidation`), `read_lock.py` (`HDF5_READ_LOCK`)
- `validator/validation.py` re-exports `_hdf5_read_lock` and builds `GPUBatchedValidation` in `_create_gpu_validation`; per-thread instances cached via `threading.local()`
- Env: `QA4SM_MAX_PARALLEL_WORKERS`, `QA4SM_HEARTBEAT_INTERVAL_SECONDS`, `QA4SM_GPU_ENABLED`, `QA4SM_GPU_DEVICE_ID`, `QA4SM_GPU_BATCH_SIZE`, `QA4SM_TS_CACHE_SIZE_MB`

## CLI
- `qa4sm-validate [config] [--dry-run] [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [--log-file PATH] [--max-workers N] [--no-gpu] [--batch-size N] [--gpu-device N]`
- Executor selection: GPU batched thread pool when GPU available, process pool for ISMN/Windows (HDF5 thread-safety), otherwise thread pool

## Dev
- Python >=3.13, uv package manager
- Lint: `ruff check .`  |  Format: `ruff format .`  |  Tests: `pytest`  |  Types: `mypy validator`
- Sub-projects have their own `tox`/`pytest`/CI configs
- Key entrypoint: `validator/cli.py:main`
- Models mimic Django ORM (in-memory dataclasses with `.objects.filter()`, `.save()`, `.delete()`)
- Output dir: `validator/media/` (`settings.py:MEDIA_ROOT`)
- Stale `.nc` files in output dir are auto-cleaned on re-run

## Gotchas
- Readers selected by string name in JSON config, not by dataset short_name
- Some noise is auto-suppressed (first 5 occurrences of repetitive messages)
- Validation tasks tracked in-memory only (no database)
