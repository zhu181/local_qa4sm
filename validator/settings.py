import os

MEDIA_ROOT = os.path.join(os.path.dirname(__file__), "media")
DEFAULT_PARALLEL_WORKERS = os.cpu_count() or 1

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
