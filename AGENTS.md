# QA4SM Local Validation

## Quick start
- Run: `uv run qa4sm-validate <config.json>`
- Or: `uv run python -m validator.cli <config.json>`
- Dry-run: add `--dry-run` flag

## Config
- JSON config file, see `validator/validation_run.template.json`
- Reader field selects the data reader class (e.g. `"SMAPL3_V9Reader"`, `"ISMN_Interface"`, `"GriddedNcOrthoMultiTs"`)
- `storage_path` must point to local data directories (use your own paths)

## Architecture (3 packages)
| Package | Path | Role |
|---------|------|------|
| `validator` | `./validator/` | CLI + validation engine (primary package) |
| `qa4sm-reader` | `./qa4sm-reader/` | NetCDF result reading & plotting (separate git history) |
| `qa4sm-preprocessing` | `./qa4sm-preprocessing/` | Data preprocessing readers (separate git history) |

## CLI
- `qa4sm-validate [config] [--dry-run] [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [--log-file PATH] [--max-workers N] [--no-gpu] [--gpu-device ID] [--batch-size N] [--cache-size-mb N]`
- Env: `QA4SM_MAX_PARALLEL_WORKERS`, `QA4SM_HEARTBEAT_INTERVAL_SECONDS`, `QA4SM_GPU_DEVICE_ID`
- Default workers: `max(1, os.cpu_count() // 2)`
- Executor selection (`validator/validation.py:_determine_executor`):
  - **gpu** (GPU enabled + available): `ThreadPoolExecutor` (max 4 workers) + CuPy batched metrics via pytesmo `GPUBatchedValidation`
  - **process**: `ProcessPoolExecutor` (default for CPU path, ISMN reference, and Windows)
  - **thread**: `ThreadPoolExecutor` (CPU fallback)

## Dev
- Python >=3.13, uv package manager
- Lint: `ruff check .`  |  Format: `ruff format .`  |  Tests: `pytest`  |  Types: `mypy validator`
- Sub-projects have their own `tox`/`pytest`/CI configs
- Key entrypoint: `validator/cli.py:main`
- Models mimic Django ORM (in-memory dataclasses with `.objects.filter()`, `.save()`, `.delete()`)
- Output dir: `validator/media/` (`settings.py:MEDIA_ROOT`)
- Stale `.nc` files in output dir are auto-cleaned on re-run

## GPU (CuPy, in vendored pytesmo)
- GPU acceleration lives **inside vendored `pytesmo`** (branch `feat/gpu-cupy`), not in a separate package:
  - `pytesmo/src/pytesmo/gpu_backend.py` — lazy thread-safe `GPUBackend` singleton (CuPy with NumPy fallback; `to_gpu`/`to_cpu`/`synchronize`/`empty_cache`/`mem_info`)
  - `pytesmo/src/pytesmo/validation_framework/gpu_metrics.py` — `BatchedPairwiseMetrics`, `BatchedTCAMetrics` (padded masked tensors, float64 intermediates, mirror numpy references)
  - `pytesmo/src/pytesmo/validation_framework/gpu_validation.py` — `GPUBatchedValidation` (drop-in for `Validation.calc()`, byte-compatible result format)
- Install GPU extras: `uv sync --extra gpu` (root) → installs `cupy-cuda13x[ctk]>=14`. Must be CUDA 13.x wheel for RTX 50-series (Blackwell sm_120); `cupy-cuda12x` does NOT work on it.
- Config fields (optional): `gpu: true/false`, `gpu_device`, `batch_size`, `cache_size_mb`

## HDF5 thread safety (Windows crashes `0xC0000005`)
- `libhdf5`/netCDF-C are **not thread-safe** in the PyPI wheels (`netCDF4` has no internal lock; `h5py` is safe only via its internal `phil` lock). Never run netCDF4 reads concurrently without protection.
- Mitigation (single source of truth: `pytesmo/.../validation_framework/read_lock.py`):
  - `HDF5_READ_LOCK = threading.RLock()` wraps all netCDF4 reads at `DataManager.get_data()`/`get_other_data()` in pytesmo
  - GPU path: threads + lock (short critical sections; compute overlaps reads)
  - CPU/ISMN/Windows path: `ProcessPoolExecutor` (separate processes, no shared HDF5 state, no lock needed)
- Do NOT "optimize" by removing the lock or switching the GPU path to processes

## Gotchas
- Readers selected by string name in JSON config, not by dataset short_name
- Some noise is auto-suppressed (first 5 occurrences of repetitive messages)
- Validation tasks tracked in-memory only (no database)
- Config JSON: `gpu: false` disables GPU even if CuPy + device present; `--no-gpu` overrides
