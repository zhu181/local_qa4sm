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

# GPU acceleration flags (can be overridden by CLI / config).
GPU_ENABLED: bool = os.getenv("QA4SM_GPU_ENABLED", "").lower() in ("1", "true", "yes")
GPU_DEVICE_ID: int = int(os.getenv("QA4SM_GPU_DEVICE_ID", "0"))
GPU_BATCH_SIZE: int = int(os.getenv("QA4SM_GPU_BATCH_SIZE", "1000"))
GPU_CACHE_SIZE_MB: int = int(os.getenv("QA4SM_GPU_CACHE_SIZE_MB", "2048"))
