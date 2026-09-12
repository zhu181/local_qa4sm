# Changelog

All notable changes to this repository will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
for the public-facing tooling (`validator/` package, `qa4sm-validate`
CLI, `run_validation.ps1`). The vendored `pytesmo/` submodule has its
own version line in `pytesmo/CHANGELOG.rst`.

## [Unreleased]

### Added

- **Dask is now the default parallel backend.** The Dask streaming
  path (`pytesmo.parallel.DaskParallelExecutor` + per-batch
  callback) is enabled out of the box; the classic
  `ThreadPoolExecutor` path is opt-in via
  `QA4SM_USE_CLASSIC=1`.
- **Classic escape hatch.** `QA4SM_USE_CLASSIC=1` (requires
  `QA4SM_USE_DASK=0`) switches back to the per-job
  `ThreadPoolExecutor` path with coalesced netCDF writes and
  corruption-tolerant reopen-validation.
- **GPU acceleration is an opt-in accelerator, not a separate
  path.** `QA4SM_USE_GPU=1` implies Dask and adds CuPy/NumPy-metric
  acceleration inside the Dask workers. When CUDA/CuPy is missing
  on a non-CUDA box, the validator logs a warning and continues on
  the Dask path with NumPy metrics (no fallback to the classic path,
  which used to discard finished streaming results).
- **Corrupt-input read retries.** `Validation.calc` wraps
  `data_manager.get_data()` in a retry loop. On
  `_is_corruption_error` (heuristic for `EOFError`, `struct.error`,
  `OSError` mentioning `netcdf`/`hdf`/`truncate`/`corrupt`/...,
  `TypeError` on `NoneType`) the read is retried up to
  `QA4SM_MAX_READ_RETRIES` times with `QA4SM_READ_RETRY_DELAY_SECONDS`
  sleep between attempts. After the retries the gpi is degraded to
  `NoGpiDataError` (status `7`) so the run continues. Off by default
  (`QA4SM_MAX_READ_RETRIES=0`) for backward compatibility.
- **Per-job retries inside Dask batches.** New env
  `QA4SM_DASK_BATCH_RETRIES` (default `3`) controls how many times a
  Dask worker retries a whole batch task after a transient failure
  (e.g. worker death, OOM).
- **Zarr cache quarantine.** `QA4SM_CACHE_LOAD_RETRIES` (default
  `3`) — consecutive `_load_batch_zarr` failures before a corrupt
  batch directory is moved under `<cache>/.quarantine/` and
  recomputed from scratch.
- **Classic-path per-job timeout.** `QA4SM_THREAD_TIMEOUT_SECONDS`
  (default `0` = disabled). When set, jobs that don't finish within
  the window are logged, counted as error, and resubmitted. The
  original thread continues in the background; Python cannot kill
  it.
- **Classic-path netCDF write coalescing.** NetCDF writes are
  coalesced into batches of 10 jobs per open/append/close cycle.
  Between flushes the run netCDF is reopened read-only; silently
  corrupted files (Windows `STATUS_ACCESS_VIOLATION` inside
  `netcdf-*.dll`) are deleted so the next write recreates them
  cleanly. Extended `OSError` recoverable messages cover more
  netCDF4 failure modes.
- **`run_validation.ps1` mode selector.** New switches `-Dask`
  (default; no-op on default runs) and `-Classic` (escape hatch;
  unsets `QA4SM_USE_DASK`). `-Gpu` now implies `-Dask`. `-MemoryLimit`
  is ignored under `-Classic`.
- **`run_validation.ps1` retry flags.** `-MaxReadRetries`,
  `-DaskBatchRetries`, `-ThreadTimeout`, `-CacheLoadRetries` map
  directly to their `QA4SM_*` envs.
- **`run_validation.ps1 -Interactive`** now prompts through the
  Dask / Dask+GPU / Classic mode selector (Dask default), memory
  budget, worker count, dry-run, then confirms before running.
- **Public-readiness docs.** MIT `LICENSE`; rewritten `README.md`
  (~326 lines) covering install, behaviour matrix, CLI, env vars,
  outputs, presets, architecture pointer, troubleshooting;
  `CHANGELOG.md` (this file); `CONTRIBUTING.md`; `docs/ARCHITECTURE.md`.
- **pytesmo submodule** bumped from `257aada` to `385277e`. Five new
  commits on `feat/batch-resilience-2026-09` (submodule branch) add:
  - `DaskParallelExecutor(use_gpu=...)` gating flag (no CuPy probe
    crash on non-CUDA boxes).
  - Per-job retries inside `_make_batch_processor`.
  - Zarr cache quarantine on repeated load failures.
  - `Validation.calc(max_read_retries=, retry_delay_seconds=)`
    corrupt-input retry.
  - `SubsetsMetricsAdapter` status-key-safe empty-result patch
    (regression test included).
  - `docs/gpu_acceleration.rst` streaming / retry section; README.rst
    fork notice + badge strip; CHANGELOG.rst Unreleased entries.

### Changed

- `_run_gpu_dask_validation` renamed to `_run_dask_validation` with
  a backward-compat alias.
- `_gpu_dask_requested` split into `_dask_requested` (path selection)
  + `_gpu_dask_requested` (CUDA detection only). The path is now
  Dask by default; the helper downgrades `USE_GPU` silently when
  CUDA is unavailable instead of falling back to classic.
- Progress log prefix changed from `"GPU/Dask progress: ..."` to
  `"Dask progress: ..."`; a per-worker memory summary line was
  added on the same throttled cadence.
- `validator/cli.py`: centralised `silenced_prefixes` tuple covers
  `"Sending large graph of size"` from Dask in addition to the
  pytesmo per-gpi warnings, with a matching
  `warnings.filterwarnings("ignore", ...)` rule.
- `AGENTS.md`: behaviour matrix + new env docs.
- `pytesmo/README.rst`: stripped upstream-pointing `ci/cov/pip/doc`
  badges (they pointed at TUW-GEO and could mislead); added fork
  notice + "Recent additions in this fork" section.

## Notable earlier changes

This branch (`feat/gpu-pytesmo-integrate`) accumulated the following
work since it was forked from `main`. See `git log` for full
detail:

- Dask streaming validation with crash-resume batch cache (`5954c6f`).
- ISMN+tcol path fixes (`27cb5bf`).
- `-MemoryLimit` persistence across script invocations (`3a188bf`).
- Dask GPU path memory handling (`b09eb41`).
- Worker memory leak hardening (`4132cd8`, `1118f99`).
- Auto-log every `run_validation.ps1` invocation (`5e73c0e`).
- Preset-friendly `run_validation.ps1` with bbox / env / interactive
  modes (`62c1411`).
- Output dir moved to repo-root `outputs/` (`9c740ec`).
- GPU/Dask parallel validation with `QA4SM_USE_GPU` toggle
  (`8b6060d`).

For older history see `git log` on `main` and `origin/main`.

[Unreleased]: https://github.com/zhu181/local_qa4sm/compare/main...HEAD