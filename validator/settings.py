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

# Run validation through the Dask-parallel GPU path (pytesmo[gpu]).
# Falls back to the classic threaded path if no GPU/CuPy is available.
USE_GPU = os.getenv("QA4SM_USE_GPU", "0") == "1"

# Total memory budget for all Dask workers in the GPU/Dask path. Accepts a
# string parsable by Dask (e.g. "16GB") or a plain byte count. The value is
# divided equally among workers. If unset, each worker defaults to 4 GB.
DASK_MEMORY_LIMIT = os.getenv("QA4SM_DASK_MEMORY_LIMIT", "") or None


# Number of gpis computed per Dask batch in the GPU/Dask path. Larger batches
# amortize the per-batch fixed costs (reader deserialization, GC, zarr save)
# better; per-gpi peak memory is small because the Dask path reads targeted
# slices (read_bulk=False) instead of bulk-loading whole datasets per batch.
def _parse_batch_size() -> int:
    try:
        return max(1, int(os.getenv("QA4SM_DASK_BATCH_SIZE", "100")))
    except (TypeError, ValueError):
        return 100


DASK_BATCH_SIZE = _parse_batch_size()
