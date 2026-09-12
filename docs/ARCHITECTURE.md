# Architecture

This document describes the runtime architecture of `local_qa4sm`
from a code-reader's perspective. For environment variables, CLI
flags, and behaviour matrix see [`README.md`](../README.md); for the
agent memory file see [`AGENTS.md`](../AGENTS.md).

## 1. Request lifecycle

A single `qa4sm-validate <config.json>` invocation flows through
six layers. Each layer is in its own module for testability.

```
  ┌─────────────────────────────────────────────────────────────┐
  │  1. CLI  (validator/cli.py:main)                            │
  │     - argparse (config, --dry-run, --log-level, ...)        │
  │     - logging setup; warning suppression                     │
  │     - termination-signal handlers                           │
  └────────────────────────┬────────────────────────────────────┘
                           ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  2. Config parse  (validator/orchestrator.py)               │
  │     JSON file → ValidationRun dataclass + child objects     │
  │     (DatasetConfiguration, ParametrisedFilter, ...).        │
  │     Fails fast if storage_path is missing.                  │
  └────────────────────────┬────────────────────────────────────┘
                           ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  3. Plan  (validator/validation.py)                         │
  │     - run_validation() builds per-gpi/lon/lat jobs.         │
  │     - Selects parallel backend via _dask_requested().       │
  │     - _determine_max_workers(jobs, ref_reader) returns 1    │
  │       when the ISMN reader is the spatial reference (the    │
  │       reader is not fork/thread-safe).                      │
  └────────────────────────┬────────────────────────────────────┘
                           ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  4. Execute  (validator/validation.py)                      │
  │     Dask path:                                              │
  │       _run_dask_validation()  →                             │
  │         pytesmo.Validation.calc(parallel='dask',            │
  │           batch_callback=_batch_cb,                         │
  │           max_read_retries=..., retry_delay_seconds=...)    │
  │         each batch → _pytesmo_to_qa4sm_results →            │
  │         check_and_store_results (netCDF append).            │
  │     Classic path:                                           │
  │       _run_classic_validation() with ThreadPoolExecutor,    │
  │       write coalescing queue, _reopen_netcdf_or_delete.     │
  └────────────────────────┬────────────────────────────────────┘
                           ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  5. Post-process  (validator/validation.py:_post_process_run)│
  │     - Transcribe pytesmo netCDF → QA4SM netCDF via          │
  │       qa4sm_reader. _extend_transcriber_datasets monkey-    │
  │       patches the DATASETS list so tcol metrics (snr,       │
  │       beta, err_std) survive for datasets outside the       │
  │       hard-coded list.                                      │
  │     - Generate plots via qa4sm_reader.graphics.             │
  └────────────────────────┬────────────────────────────────────┘
                           ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  6. Outputs  (outputs/<run_id>/)                            │
  │     - <run_id>.nc (consolidated; safe to open read-only)    │
  │     - <run_id>_plots/ (overview maps, scatter, box,         │
  │       tcol status barplot)                                  │
  │     - logs/validation_run_<ts>.log (per-PS1-run script log) │
  └─────────────────────────────────────────────────────────────┘
```

Key invariants:

- `_dask_requested()` is the single source of truth for backend
  selection. Both `cli.py` dry-run output and `run_validation()` use
  it.
- `check_and_store_results()` is the only writer of the run netCDF;
  the classic-path write queue funnels every result through it so
  the recovery logic (`OSError` → reopen / delete / retry) and the
  Dask-path streaming callback share one code path.
- Stale `.nc` files in the run dir are auto-cleaned on re-run by
  `_post_process_run()`.

## 2. Parallel backends

### Decision matrix

| `QA4SM_USE_DASK` | `QA4SM_USE_GPU` | `QA4SM_USE_CLASSIC` | Path |
|---|---|---|---|
| `1` (default) | `0` (default) | `0` (default) | **Dask streaming + NumPy metrics** |
| `1` | `1` | `0` | Dask streaming + GPU metrics (CuPy) |
| `0` | `1` | `0` | Dask streaming + GPU metrics (GPU implies Dask) |
| `0` | `0` | `1` | Classic `ThreadPoolExecutor` (escape hatch) |
| `0` | `0` | `0` | Dask streaming + NumPy metrics (default) |

`_dask_requested()` returns `True` whenever Dask is in play
(USE_DASK=1 OR USE_GPU=1, AND USE_CLASSIC is not set). When both
USE_DASK and USE_GPU are off AND USE_CLASSIC is off, it still
returns `True` because Dask is now the default fallback — only an
explicit USE_CLASSIC=1 opts out.

`_gpu_dask_requested()` is a thinner check: does the user want GPU
metrics AND is CUDA/CuPy actually available? When CuPy is missing
on a non-CUDA box, the helper downgrades `settings.USE_GPU` to
`False` (so metrics dispatch to NumPy) and the Dask path still
runs. The previous behaviour — return `False` and fall back to the
classic `ThreadPoolExecutor` path — is gone: that fallback used to
discard finished streaming results.

### Dask path internals

`pytesmo.Validation.calc(parallel="dask", batch_callback=_batch_cb,
...)` invokes `pytesmo.parallel.DaskParallelExecutor.map_batches_streaming`
under the hood. Each completed batch is:

1. Compacted by `_compact_batch` (drops executor-level error dicts).
2. Handed to `_batch_cb`.
3. Released from worker memory.

The qa4sm batch callback converts the per-gpi result dicts to
QA4SM shape (`_pytesmo_to_qa4sm_results`) and appends them to the
run netCDF via `check_and_store_results`. Peak client and per-worker
memory is therefore ~one batch (`batch_size=settings.DASK_BATCH_SIZE`,
default `100`; drop to `25` for memory-hungry ISMN×SMAP runs).

The crash-resume cache lives at
`outputs/.dask_batch_cache/<sha1(config)>_b<batch_size>/batch_<i>.zarr/`.
Each batch dir has `combos.json`, `metrics.json`, per-combo zarr
groups, and a `.complete` sentinel written last. A re-run with the
same config skips done batches and only recomputes missing ones.
The pytesmo resume path length-checks each cached batch against the
current batch as belt-and-braces (covered by
`test_map_batches_streaming_resume_batch_size_mismatch`).

If `_load_batch_zarr` raises on the same batch
`QA4SM_CACHE_LOAD_RETRIES` (default `3`) times in a row, the batch
directory is moved under `<cache>/.quarantine/batch_<idx>_<ts>/` and
recomputed on the next run.

### Classic path internals

Classic runs use a `ThreadPoolExecutor` job loop. The wait/poll loop
is reshaped from the original "submit-all-at-once" style to a
"submit-on-completion" style so a slow job doesn't block fast ones,
and so the ISMN-fork-safe `n_workers=1` constraint is honoured
without over-submitting.

When `QA4SM_THREAD_TIMEOUT_SECONDS > 0`, the wait loop polls every
`thread_timeout` seconds instead of the original 10 s. Jobs that
exceed the window are logged, counted as error, and resubmitted; the
hung thread continues in the background (Python cannot kill it).

NetCDF writes are coalesced into batches of 10 jobs per
open/append/close cycle (write queue + `_flush_results_queue`).
Between flushes the run netCDF is reopened read-only
(`_reopen_netcdf_or_delete`); silently corrupted files (Windows
`STATUS_ACCESS_VIOLATION` inside `netcdf-*.dll`) are deleted so the
next write recreates them cleanly. `PermissionError` on Windows
file-locks is treated as transient and surfaced for the next flush.

## 3. pytesmo integration boundary

The parent repo imports only the public pytesmo surface. The
boundary is small enough to enumerate:

### Imports from `pytesmo`

```python
from pytesmo.validation_framework.validation import Validation
from pytesmo.validation_framework.data_manager import DataManager
from pytesmo.validation_framework.metric_calculators import (
    PairwiseIntercomparisonMetrics,
    TripleCollocationMetrics,
)
from pytesmo.validation_framework.metric_calculators_adapters import (
    SubsetsMetricsAdapter,
    StabilityMetricsAdapter,
    IntraAnnualMetricsAdapter,
)
from pytesmo.validation_framework.temporal_matchers import (
    make_combined_temporal_matcher,
)
from pytesmo.io.zarr_writer import ...    # via executor; not imported here
from pytesmo.parallel import DaskParallelExecutor  # not directly imported
                                                   # (the executor is constructed
                                                   # inside pytesmo.Validation.calc)
from pytesmo.gpu import is_gpu_available   # imported inside validator/validation.py
                                           # to short-circuit USE_GPU when CUDA is missing
```

### `Validation.calc()` kwargs we pass

| Kwarg | Source | Effect |
|---|---|---|
| `parallel` | `settings.USE_DASK` | `"dask"` or `None` |
| `use_gpu` | `settings.USE_GPU` | GPU metrics dispatch (or NumPy fallback) |
| `n_workers` | `_determine_max_workers()` | Dask cluster size |
| `batch_size` | `settings.DASK_BATCH_SIZE` | Gpis per Dask batch |
| `output_format` | hard-coded `"zarr"` (parent) / `None` (classic) | Intermediate store |
| `output_path` | tempdir | Intermediate store path |
| `progress` | `True` | Progress bar |
| `progress_callback` | `_make_gpu_progress_callback()` | Throttled log line |
| `batch_callback` | `_batch_cb` | Stream each batch to netCDF |
| `max_read_retries` | `settings.MAX_READ_RETRIES` | Corrupt-input read retries |
| `retry_delay_seconds` | `settings.READ_RETRY_DELAY_SECONDS` | Sleep between read retries |
| `parallel_kwargs` | dict | `dashboard=False`, `memory_limit`, `memory_*_fraction`, `use_gpu`, `retries`, `max_job_retries`, `retry_delay_seconds`, `cache_load_retries` |

### The `_extend_transcriber_datasets` monkey-patch

`qa4sm_reader.globals.DATASETS` is a hard-coded list (e.g. `ISMN`,
`ASCAT`, ...) consulted by `qa4sm_reader.netcdf_transcription.is_valid_tcol_metric_name`
to decide which datasets' tcol metrics (`snr`/`beta`/`err_std`)
survive the transcription into the QA4SM-shaped netCDF. Datasets
outside the list silently lose their tcol metrics.

This is a problem when running with datasets that the upstream
transcriber doesn't know about (e.g. `SPL3SMPE`, `NSMCSMC`,
`SMMerge`). Without intervention the transcribed netCDF would
contain only `0-ISMN` for tcol metrics even when the run included
`SPL3SMPE × NSMCSMC × ISMN`.

Fix: `_extend_transcriber_datasets(run)` patches
`qa4sm_reader.netcdf_transcription.DATASETS` (the module namespace
`is_valid_tcol_metric_name` reads at call time) before
`_post_process_run` calls `transcriber.get_transcribed_dataset()`.
Patching `qa4sm_reader.globals.DATASETS` alone does NOT work —
the transcriber imports a copy at module load.

### Streaming batch callback

The streaming batch callback wires the per-batch pytesmo result
into the qa4sm write path:

```python
def _batch_cb(batch_result):
    # 1. Convert per-gpi pytesmo dict → qa4sm dict (status, metrics, gpi list)
    qa4sm_batch = _pytesmo_to_qa4sm_results(batch_result)
    # 2. Append to run netCDF (errors recoverable as in classic path)
    check_and_store_results(task_id, qa4sm_batch, run_dir)
    # 3. Bookkeeping
    state["seen"] += len(qa4sm_batch)
    state["ok"] += ok_count
    state["error"] += error_count
```

The `task_id` is the run-level UUID; per-gpi gpis are inside
`qa4sm_batch`. The callback runs in the **client process** (not the
worker), so it can write to the run netCDF directly via the
netCDF4 API without IPC.

## 4. Failure modes and where they're handled

| Failure mode | Where caught | Behaviour |
|---|---|---|
| Corrupt netCDF read | `pytesmo.Validation.calc` (`max_read_retries` retry loop) | Up to `MAX_READ_RETRIES` retries; on exhaustion gpi → `NO_GPI_DATA` (status 7) |
| Dask worker dies mid-batch | `pytesmo.DaskParallelExecutor` (Dask-level `retries`) + qa4sm `DASK_BATCH_RETRIES` parallel kwarg | Up to N Dask-level retries; qa4sm increments `error_points` per gpi in the lost batch |
| `local_directory` OOM kills worker | pytesmo's `memory_terminate_fraction=0.95` triggers worker shutdown; Dask-level retries pick it up | Same as above |
| Corrupt zarr cache batch | `map_batches_streaming` quarantine logic (`QA4SM_CACHE_LOAD_RETRIES`) | Quarantine to `<cache>/.quarantine/`; recompute on next run |
| netCDF4 `STATUS_ACCESS_VIOLATION` (classic path, large runs) | `check_and_store_results` (extended OSError messages) + `_reopen_netcdf_or_delete` | Detected between flushes; corrupted file deleted and recreated |
| Hung classic-path job | `QA4SM_THREAD_TIMEOUT_SECONDS` poll loop | Logged, error-points counted, resubmitted; thread continues in background |
| `DaskParallelExecutor.close()` TimeoutError | pytesmo `DaskParallelExecutor.close()` catches + logs; qa4sm treats a post-close exception (when `state["ok"]+state["error"] == n_gpis`) as success | Results already streamed to netCDF |
| CuPy missing on non-CUDA box | `_gpu_dask_requested()` checks `pytesmo.gpu.is_gpu_available` | Logs warning; downgrades `USE_GPU` to `False`; Dask path runs with NumPy metrics |
| Tornado TCP-handshake garbage frames (`"Unable to allocate 2.95 EiB..."`) | Not silenced — `tornado.application` is left at WARNING | Benign; the Dask cluster keeps working |

## 5. Where to look when adding a new feature

| You want to... | Touch these files |
|---|---|
| Add an env var | `validator/settings.py` (parser + defaults), `AGENTS.md` (env list), `run_validation.ps1` (PS1 flag if applicable), `README.md` (env-var table), `tests/test_gpu.py` (parser test) |
| Add a dataset / reader | `validator/readers.py` (`_READER_REGISTRY` + factory), `validator/orchestrator.py` (dataset_config → reader dispatch), README (reader name) |
| Add a metric | `validator/batches.py` (per-combo metric spec), `validator/orchestrator.py` (metric config schema), README (metric description) |
| Tweak the parallel backend | `validator/validation.py` (`_dask_requested`, `_run_dask_validation`, `_run_classic_validation`), `validator/settings.py` (parallel-related envs), `pytesmo/src/pytesmo/parallel.py` (if the change crosses the boundary) |
| Add a CLI flag | `validator/cli.py` (argparse), `run_validation.ps1` (PS1 flag + env wiring), README (CLI section) |
| Add a plot type | `qa4sm_reader` (separate package); the validator only calls `qa4sm_reader.graphics.plot_all(...)` from `_post_process_run` |
| Change pytesmo behaviour | `pytesmo/` submodule, own branch + own commit; bump the pointer in a separate parent commit |