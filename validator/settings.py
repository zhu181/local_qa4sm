import os

MEDIA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
DEFAULT_PARALLEL_WORKERS = max(1, (os.cpu_count() or 2) // 2)

# Configure max number of local workers used in run_validation.
# Can be overridden via environment variable QA4SM_MAX_PARALLEL_WORKERS.
MAX_PARALLEL_WORKERS = int(os.getenv("QA4SM_MAX_PARALLEL_WORKERS", str(DEFAULT_PARALLEL_WORKERS)))
if MAX_PARALLEL_WORKERS < 1:
    MAX_PARALLEL_WORKERS = 1

# Emit a progress heartbeat while waiting for parallel jobs.
HEARTBEAT_INTERVAL_SECONDS = int(os.getenv("QA4SM_HEARTBEAT_INTERVAL_SECONDS", "60"))
if HEARTBEAT_INTERVAL_SECONDS < 1:
    HEARTBEAT_INTERVAL_SECONDS = 1

# GPU acceleration settings
GPU_ENABLED = os.getenv("QA4SM_GPU_ENABLED", "1") != "0"
GPU_DEVICE_ID = int(os.getenv("QA4SM_GPU_DEVICE_ID", "0"))
GPU_BATCH_SIZE = int(os.getenv("QA4SM_GPU_BATCH_SIZE", "500"))
TS_CACHE_SIZE_MB = int(os.getenv("QA4SM_TS_CACHE_SIZE_MB", "128"))
if TS_CACHE_SIZE_MB < 0:
    TS_CACHE_SIZE_MB = 0
