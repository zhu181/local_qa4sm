import os

MEDIA_ROOT = os.path.join(os.path.dirname(__file__), "media")
DEFAULT_PARALLEL_WORKERS = os.cpu_count() or 1

# Configure max number of local workers used in run_validation.
# Can be overridden via environment variable QA4SM_MAX_PARALLEL_WORKERS.
MAX_PARALLEL_WORKERS = int(
    os.getenv("QA4SM_MAX_PARALLEL_WORKERS", str(DEFAULT_PARALLEL_WORKERS))
)
if MAX_PARALLEL_WORKERS < 1:
    MAX_PARALLEL_WORKERS = 1
