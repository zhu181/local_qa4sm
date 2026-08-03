# QA4SM Local Validation

## Quick start
- Run: `uv run qa4sm-validate <config.json>` (config arg optional; defaults to `validator/validation_run.template.json`)
- Or: `uv run python -m validator.cli <config.json>`; dry-run: `--dry-run`
- Windows helper: `.\run_validation.ps1 [<preset>] [-Config PATH] [-BBox lat0,lon0,lat1,lon1] [-Gpu] [-MemoryLimit SIZE] [-Heartbeat SEC] [-EnvVar NAME=value] [-DryRun] [-LogLevel X] [-LogFile PATH] [-MaxWorkers N] [-Interactive]` (uses `.venv\Scripts\python.exe`)
  - Preset names are fuzzy-matched against `presets/validation_run.*.json` (e.g. `ismn-spl3`, `spl3smpe`); `-List` shows them
  - `-BBox` overlays `min_lat/min_lon/max_lat/max_lon` onto a temp copy (keeps full-preset runs small/fast) and deletes it after a real run
  - `-Gpu` sets `QA4SM_USE_GPU=1`; `-MemoryLimit` sets `QA4SM_DASK_MEMORY_LIMIT` (e.g. `16GB`); `-Heartbeat` sets `QA4SM_HEARTBEAT_INTERVAL_SECONDS`; `-EnvVar` is a repeatable `NAME=value` passthrough; unset vars are cleared so no stale env leaks between runs; `-Config` still accepted for explicit paths
  - Every invocation writes a timestamped log to `logs\validation_run_<ts>.log` by default (script-level events + validation output in one file; dry-runs too) unless `-LogFile` overrides it; script prints the path at the end
  - `-Interactive` prompts through preset/bbox/GPU/memory/dry-run choices (GPU defaults ON) then confirms before running

## Repo layout
| Path | Role | Git status |
|------|------|-----------|
| `validator/` | CLI + validation engine (primary package) | committed |
| `pytesmo/` | Git submodule = fork `github.com/zhu181/pytesmo` with GPU/Dask work; has a nested `tests/test-data` submodule; read `pytesmo/AGENTS.md` before editing | submodule |
| `qa4sm-reader/`, `qa4sm-preprocessing/` | NOT vendored; installed from PyPI as `qa4sm_reader`/`qa4sm_preprocessing`; optional local clones via `scripts/setup.ps1` | gitignored |
| `presets/`, `data/`, `outputs/`, `logs/`, `dist/` | Local configs / runtime dirs (presets has ready-made run configs) | gitignored |

## Config
- JSON config; `reader` selects the reader class by string name (e.g. `"SMAPL3_V9Reader"`, `"ISMN_Interface"`, `"GriddedNcOrthoMultiTs"`), NOT by dataset short_name
- `storage_path` must point to your own local data dirs (CLI fails fast if missing); `SMAPTs` reader also requires `grid.nc` there

## CLI & env
- `qa4sm-validate [config] [--dry-run] [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [--log-file PATH] [--max-workers N]`
- Env: `QA4SM_MAX_PARALLEL_WORKERS` (default = CPU count), `QA4SM_HEARTBEAT_INTERVAL_SECONDS` (default 60), `QA4SM_USE_GPU`, `QA4SM_DASK_MEMORY_LIMIT` (Dask-compatible string like "16GB"; default 60% of system RAM)
- Classic path: `ThreadPoolExecutor`; ISMN reference reader forces `max_workers=1` (not thread-safe)
- Repetitive log/warning noise auto-suppressed after first 5 occurrences; pytesmo logger forced to CRITICAL; `dask`/`distributed` loggers forced to WARNING so their INFO chatter doesn't drown validator output
- Progress is logged to the log file (not just tqdm-to-stderr): GPU/Dask path logs `GPU/Dask progress: <done>/<total> gpis (<pct>%) elapsed=… ETA≈…` at INFO every ~15 s plus 25/50/75/100% milestones; classic path heartbeat includes ok/error counts + ETA
- `run_validation.ps1 -MemoryLimit` accepts a bare number (e.g. `8`) and normalizes it to `8GB` (logs the assumption); a bare int passed to Dask is an invalid memory spec

## GPU acceleration
- pytesmo metrics auto-dispatch to CuPy when available (`_GPU_AVAILABLE` checked at import); no per-call wiring needed
- `QA4SM_USE_GPU=1` switches to the Dask-parallel GPU path (single `Validation.calc(..., parallel="dask", use_gpu=True)`); if GPU/CuPy is unavailable it logs a warning and falls back to the classic threaded path
- **Memory-bounded streaming (M3/E4)**: the GPU path streams results in batches instead of accumulating everything in the client.
  - `Validation.calc(..., batch_callback=fn)`: dask branch routes through `_run_streaming` → `executor.map_batches_streaming`; each completed batch is compacted (`_compact_batch`) and handed to the callback, then released. `calc()` returns a small summary dict `{"n_gpis", "batches", "keys"}` instead of full results (backward compatible — without `batch_callback` it returns the full merged dict as before).
  - qa4sm `_run_gpu_dask_validation` passes a callback that converts each batch (`_pytesmo_to_qa4sm_results`) and appends it to the run netCDF (`check_and_store_results`), so peak client/worker memory is ~one batch (`batch_size=100`).
  - **Crash-resume cache**: completed batches are persisted client-side to `outputs/.dask_batch_cache/<sha1(config)>/batch_<i>.zarr/` (per-combo zarr groups + `combos.json` + `metrics.json` + `.complete` sentinel written last). Same config re-run skips done batches; any config change (datasets/interval/bbox/metrics) gets a fresh hash dir. Cache is per-config and persists across runs; no cleanup yet. The batch store handles the real per-gpi result structure `{combo: [{metric: array}, ...]}` (list of per-window metric dicts); error gpis are positionally aligned as absent on reload. Metric names with filesystem-incompatible characters (e.g. `2016|BIAS`) are sanitized for zarr DirectoryStore with a `metrics.json` mapping.
  - Worker memory fractions are set via `dask.config` (`distributed.worker.memory.{target,spill,pause,terminate}`, defaults 0.7/0.7/0.9/0.95) — modern distributed no longer accepts them as `LocalCluster` kwargs; qa4sm passes `memory_target_fraction=0.6, memory_spill_fraction=0.55` via `parallel_kwargs`, and the config propagates to worker subprocesses at startup.
- Dask worker `memory_limit` is set explicitly (default 60% of RAM) because Dask's `auto` (~40%) triggers `KilledWorker` on memory-hungry readers (e.g. `GriddedNcOrthoMultiTs` reads add ~1 GB RSS per gpi). Override via `QA4SM_DASK_MEMORY_LIMIT`
- Root venv uses Python 3.13 + `cupy-cuda13x==14.1.1` (matches the installed CUDA 13.3 toolkit; the only `cublas64_12.dll` on this machine is a stale app bundle that fails to load, so `cupy-cuda12x` cannot do tcol/matmul). Never mix `cupy-cuda12x` and `cupy-cuda13x` — they share the `cupy` namespace dir
- On this machine Dask auto-limits to 1 worker (1 GPU); the win is GPU-accelerated metric/bootstrap compute, not multi-worker parallelism
- Verified on real data (SPL3SMPE x NSMCSMC, 132 points with bbox): Dask+GPU streaming path runs end-to-end (~34 s validation, ~2 min total), 132/132 ok, 0 errors, output netCDF + plots. Resume verified: second run with same config loads 2 cached batches and completes correctly.
- ISMN presets (ISMN x SPL3SMPE x NSMCSMC, 68 stations, tcol=True) also verified end-to-end on both classic and Dask+GPU paths (41/41 ok on valid stations; remaining 27 = stations outside the 2017-2021 interval). Required fixes: tcol `refname` must be the prefixed spatial-ref name (`0-ISMN`), missing `frm_class` metadata filled with template defaults instead of raising, and stability-adapter status keys (`bulk|status`) recognized in pytesmo error paths

## Dev
- Python >=3.13, uv; dev deps are pytest + ruff only
- Lint: `ruff check .` | Format: `ruff format .` (line-length 120, double quotes, select E,F,I,N,W,UP)
- Tests: `uv run pytest` runs only `tests/` (no external data); files: `tests/test_{cli,filters,models,orchestrator}.py`
- pytesmo has its own venv/uv.lock/tests — run `uv run pytest` inside `pytesmo/` (Windows notebook tests fail on GBK; deselect)
- `scripts/transcribe_results.py`: pytesmo NetCDF → QA4SM NetCDF; must call `get_transcribed_dataset()` before `write_to_netcdf()`

## Architecture notes
- Entrypoint: `validator/cli.py:main`; flow: `orchestrator.parse_validation_run_config` → `validation.run_validation`
- Models are in-memory dataclasses mimicking Django ORM: `ValidationTask.objects.filter()/.get()/.save()/.delete()` (no DB)
- Output dir: `outputs/` (`settings.py:MEDIA_ROOT`); stale `.nc` files in run dir auto-cleaned on re-run
