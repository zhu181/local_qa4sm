"""qa4sm-gpu-validation: thread-safe, GPU-accelerated pytesmo-compatible metrics.

This package provides drop-in replacements for pytesmo's per-grid-point
metric calculators that compute pairwise and triple collocation metrics for
N grid points in a single batched PyTorch (CUDA) pass, with an automatic
NumPy fallback when CUDA is unavailable.

Public API
----------
- :class:`GPUBatchedValidation` - drop-in replacement for
  ``pytesmo.validation_framework.validation.Validation.calc()``
- :class:`BatchedPairwiseMetrics` / :class:`BatchedTCAMetrics` - batched
  metric calculators in pytesmo result-dict format
- :class:`GPUBackend` / :func:`get_gpu_backend` - CUDA backend detection
- :data:`HDF5_READ_LOCK` - global lock serializing HDF5/netCDF4 reads
"""

from qa4sm_gpu_validation.gpu_backend import GPUBackend, get_gpu_backend, get_xp
from qa4sm_gpu_validation.gpu_metrics import (
    BatchedPairwiseMetrics,
    BatchedTCAMetrics,
    GPUBatchedValidation,
)
from qa4sm_gpu_validation.read_lock import HDF5_READ_LOCK

__version__ = "0.1.0"

__all__ = [
    "BatchedPairwiseMetrics",
    "BatchedTCAMetrics",
    "GPUBackend",
    "GPUBatchedValidation",
    "HDF5_READ_LOCK",
    "get_gpu_backend",
    "get_xp",
]
