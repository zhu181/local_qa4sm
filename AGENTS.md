# QA4SM Local Validation

## Quick start
- Run: `uv run qa4sm-validate <config.json>` (config arg optional; defaults to `validator/validation_run.template.json`)
- Or: `uv run python -m validator.cli <config.json>`; dry-run: `--dry-run`
- Windows helper: `.\run_validation.ps1 [-Config PATH] [-DryRun] [-LogLevel X] [-MaxWorkers N]` (uses `.venv\Scripts\python.exe`)

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
- Env: `QA4SM_MAX_PARALLEL_WORKERS` (default = CPU count), `QA4SM_HEARTBEAT_INTERVAL_SECONDS` (default 60), `QA4SM_USE_GPU`
- Classic path: `ThreadPoolExecutor`; ISMN reference reader forces `max_workers=1` (not thread-safe)
- Repetitive log/warning noise auto-suppressed after first 5 occurrences; pytesmo logger forced to CRITICAL

## GPU acceleration
- pytesmo metrics auto-dispatch to CuPy when available (`_GPU_AVAILABLE` checked at import); no per-call wiring needed
- `QA4SM_USE_GPU=1` switches to the Dask-parallel GPU path (single `Validation.calc(..., parallel="dask", use_gpu=True)`); if GPU/CuPy is unavailable it logs a warning and falls back to the classic threaded path
- Root venv uses Python 3.13 + `cupy-cuda12x` (works on this machine: RTX 5070, CUDA 12.9). Never mix `cupy-cuda12x` and `cupy-cuda13x` — they share the `cupy` namespace dir
- On this machine Dask auto-limits to 1 worker (1 GPU); the win is GPU-accelerated metric/bootstrap compute, not multi-worker parallelism

## Dev
- Python >=3.13, uv; dev deps are pytest + ruff only
- Lint: `ruff check .` | Format: `ruff format .` (line-length 120, double quotes, select E,F,I,N,W,UP)
- Tests: `uv run pytest` runs only `tests/` (no external data); files: `tests/test_{cli,filters,models,orchestrator}.py`
- pytesmo has its own venv/uv.lock/tests — run `uv run pytest` inside `pytesmo/` (Windows notebook tests fail on GBK; deselect)
- `scripts/transcribe_results.py`: pytesmo NetCDF → QA4SM NetCDF; must call `get_transcribed_dataset()` before `write_to_netcdf()`

## Architecture notes
- Entrypoint: `validator/cli.py:main`; flow: `orchestrator.parse_validation_run_config` → `validation.run_validation`
- Models are in-memory dataclasses mimicking Django ORM: `ValidationTask.objects.filter()/.get()/.save()/.delete()` (no DB)
- Output dir: `validator/media/` (`settings.py:MEDIA_ROOT`); stale `.nc` files in run dir auto-cleaned on re-run
