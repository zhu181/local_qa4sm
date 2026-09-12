import os

MEDIA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
DEFAULT_PARALLEL_WORKERS = min(2, os.cpu_count() or 1)

# Configure max number of local workers used in run_validation.
# Can be overridden via environment variable QA4SM_MAX_PARALLEL_WORKERS.
MAX_PARALLEL_WORKERS = int(os.getenv("QA4SM_MAX_PARALLEL_WORKERS", str(DEFAULT_PARALLEL_WORKERS)))
if MAX_PARALLEL_WORKERS < 1:
    MAX_PARALLEL_WORKERS = 1

# Emit a progress heartbeat while waiting for parallel jobs.
HEARTBEAT_INTERVAL_SECONDS = int(os.getenv("QA4SM_HEARTBEAT_INTERVAL_SECONDS", "60"))
if HEARTBEAT_INTERVAL_SECONDS < 1:
    HEARTBEAT_INTERVAL_SECONDS = 1

# Run validation through the Dask streaming path (pytesmo.parallel.DaskParallelExecutor).
# Dask is the default parallel backend. Set QA4SM_USE_DASK=0 together with
# QA4SM_USE_CLASSIC=1 to fall back to the classic ThreadPoolExecutor path.
USE_DASK = os.getenv("QA4SM_USE_DASK", "1") == "1"

# Force the classic ThreadPoolExecutor path (escape hatch). Only honoured when
# QA4SM_USE_DASK is also disabled.
USE_CLASSIC = os.getenv("QA4SM_USE_CLASSIC", "0") == "1"

# Enable GPU acceleration in the Dask path (pytesmo[gpu]). Requires CuPy and a
# CUDA device. When unset on a non-CUDA box the Dask path runs with NumPy
# metrics; the validator does NOT fall back to the classic path automatically.
USE_GPU = os.getenv("QA4SM_USE_GPU", "0") == "1"

# Total memory budget for all Dask workers. Accepts a string parsable by Dask
# (e.g. "16GB") or a plain byte count. The value is divided equally among
# workers. If unset, each worker defaults to 4 GB.
DASK_MEMORY_LIMIT = os.getenv("QA4SM_DASK_MEMORY_LIMIT", "") or None


# Number of gpis computed per Dask batch. Larger batches amortize the
# per-batch fixed costs (reader deserialization, GC, zarr save) better;
# per-gpi peak memory is small because the Dask path reads targeted slices
# (read_bulk=False) instead of bulk-loading whole datasets per batch.
def _parse_batch_size() -> int:
    try:
        return max(1, int(os.getenv("QA4SM_DASK_BATCH_SIZE", "100")))
    except (TypeError, ValueError):
        return 100


DASK_BATCH_SIZE = _parse_batch_size()


# Number of additional retries when data_manager.get_data() raises an exception
# that looks like a corrupt input data file (detected by pytesmo's
# _is_corruption_error). After the retries are exhausted the gpi is treated as
# NoGpiDataError (status 7) so the run continues. ``0`` disables the retry.
def _parse_max_read_retries() -> int:
    try:
        return max(0, int(os.getenv("QA4SM_MAX_READ_RETRIES", "3")))
    except (TypeError, ValueError):
        return 3


MAX_READ_RETRIES = _parse_max_read_retries()


def _parse_retry_delay_seconds() -> float:
    try:
        v = float(os.getenv("QA4SM_READ_RETRY_DELAY_SECONDS", "1.0"))
        return max(0.0, v)
    except (TypeError, ValueError):
        return 1.0


READ_RETRY_DELAY_SECONDS = _parse_retry_delay_seconds()


def _parse_dask_batch_retries() -> int:
    try:
        return max(0, int(os.getenv("QA4SM_DASK_BATCH_RETRIES", "3")))
    except (TypeError, ValueError):
        return 3


DASK_BATCH_RETRIES = _parse_dask_batch_retries()


def _parse_thread_timeout_seconds() -> float:
    """Per-job timeout for the classic ThreadPoolExecutor path.

    ``0`` (default) disables the timeout — the wait loop polls every 10 s, as
    before. When set, a job that does not finish within ``THREAD_TIMEOUT_SECONDS``
    is logged, counted as error, and resubmitted (the original hung thread keeps
    running in the background; Python cannot kill it).
    """
    try:
        v = float(os.getenv("QA4SM_THREAD_TIMEOUT_SECONDS", "0"))
        return max(0.0, v)
    except (TypeError, ValueError):
        return 0.0


THREAD_TIMEOUT_SECONDS = _parse_thread_timeout_seconds()


def _parse_cache_load_retries() -> int:
    try:
        return max(1, int(os.getenv("QA4SM_CACHE_LOAD_RETRIES", "3")))
    except (TypeError, ValueError):
        return 3


CACHE_LOAD_RETRIES = _parse_cache_load_retries()
