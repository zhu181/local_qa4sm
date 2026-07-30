"""GPU backend detection and configuration for QA4SM validation.

Provides a PyTorch-backed tensor computation backend with automatic fallback
to NumPy when CUDA is unavailable. Importing this module never fails even when
PyTorch is not installed; the singleton is initialized lazily on first access.
"""

import logging
from typing import Any

import numpy

LOGGER = logging.getLogger(__name__)


class GPUBackend:
    """Detect CUDA availability via PyTorch and expose tensor computation utilities.

    Construction is safe on failure: if PyTorch is missing or no CUDA device is
    found, the instance falls back to NumPy and reports ``available == False``.
    """

    def __init__(self, device_id: int = 0) -> None:
        self.available: bool = False
        self.xp: Any = numpy
        self._device: Any = None
        self._device_id: int = device_id
        self._device_name: str = ""
        self._total_memory_bytes: int = 0
        self._torch: Any = None

        try:
            import torch
        except Exception:
            LOGGER.info("PyTorch is not installed; using NumPy fallback for array operations.")
            return

        self._torch = torch

        if not torch.cuda.is_available():
            LOGGER.info("CUDA not available via PyTorch; using NumPy fallback.")
            return

        device_count = torch.cuda.device_count()
        if device_count <= 0:
            LOGGER.info("No CUDA-capable device detected via PyTorch; using NumPy fallback.")
            return

        if device_id >= device_count:
            LOGGER.warning(
                "Requested GPU device %s but only %s devices available; using device 0.",
                device_id,
                device_count,
            )
            device_id = 0
            self._device_id = device_id

        try:
            self._device = torch.device(f"cuda:{device_id}")
            torch.cuda.set_device(device_id)
            self.available = True
            self.xp = numpy  # xp is still numpy; use to_tensor / to_numpy for GPU ops
            self._read_device_properties(device_id)
            LOGGER.info(
                "CUDA device %s ready: name=%s total_memory_bytes=%s",
                device_id,
                self._device_name,
                self._total_memory_bytes,
            )
        except Exception as exc:
            LOGGER.info(
                "CUDA device %s initialization failed (%s); using NumPy fallback.",
                device_id,
                exc,
            )
            self.available = False
            self.xp = numpy
            self._device = None

    def _read_device_properties(self, device_id: int) -> None:
        if self._torch is None:
            self._device_name = ""
            self._total_memory_bytes = 0
            return
        try:
            props = self._torch.cuda.get_device_properties(device_id)
            self._device_name = str(props.name)
            self._total_memory_bytes = int(props.total_memory)
        except Exception:
            self._device_name = ""
            self._total_memory_bytes = 0

    @property
    def device(self) -> Any:
        """Return the active torch device (cuda:N or cpu)."""
        if self.available and self._device is not None:
            return self._device
        return self._torch.device("cpu") if self._torch is not None else None

    @property
    def torch(self) -> Any:
        """Return the torch module if available, else None."""
        return self._torch

    def get_device_info(self) -> dict:
        """Return a dict with device name, total and free memory in bytes."""
        if not self.available:
            return {
                "name": "",
                "total_memory_bytes": 0,
                "free_memory_bytes": 0,
            }
        free_bytes, _total_bytes = self.mem_info()
        return {
            "name": self._device_name,
            "total_memory_bytes": self._total_memory_bytes,
            "free_memory_bytes": free_bytes,
        }

    def synchronize(self) -> None:
        """Synchronize the CUDA device when available (no-op on CPU)."""
        if not self.available or self._torch is None:
            return
        try:
            self._torch.cuda.synchronize(self._device)
        except Exception as exc:
            LOGGER.warning("torch.cuda.synchronize() failed: %s", exc)

    def mem_info(self) -> tuple[int, int]:
        """Return ``(free_bytes, total_bytes)`` for the active device."""
        if not self.available or self._torch is None:
            return (0, 0)
        try:
            free, total = self._torch.cuda.mem_get_info(self._device)
            return (int(free), int(total))
        except Exception:
            return (0, self._total_memory_bytes)

    def to_tensor(self, arr: Any, dtype: Any = None) -> Any:
        """Convert a numpy array to a torch tensor on the active device.

        Returns the input unchanged if PyTorch is not available.
        """
        if self._torch is None:
            return arr
        if dtype is None:
            tensor = self._torch.as_tensor(arr)
        else:
            tensor = self._torch.as_tensor(arr, dtype=dtype)
        if self.available and self._device is not None:
            tensor = tensor.to(self._device)
        return tensor

    def to_numpy(self, tensor: Any) -> numpy.ndarray:
        """Transfer a torch tensor back to the host as a NumPy array.

        Passes through numpy arrays unchanged.
        """
        if isinstance(tensor, numpy.ndarray):
            return tensor
        if self._torch is not None and isinstance(tensor, self._torch.Tensor):
            return tensor.detach().cpu().numpy()
        return numpy.asarray(tensor)

    def empty_cache(self) -> None:
        """Release all unoccupied cached memory from the GPU memory allocator."""
        if not self.available or self._torch is None:
            return
        try:
            self._torch.cuda.empty_cache()
        except Exception as exc:
            LOGGER.warning("torch.cuda.empty_cache() failed: %s", exc)


_gpu_backend: GPUBackend | None = None


def get_gpu_backend() -> GPUBackend:
    """Return the lazily-initialized GPU backend singleton."""
    global _gpu_backend
    if _gpu_backend is None:
        from validator import settings

        _gpu_backend = GPUBackend(device_id=settings.GPU_DEVICE_ID)
    return _gpu_backend


def get_xp() -> Any:
    """Return the active array namespace (numpy, always).

    For GPU tensor operations, use ``get_gpu_backend().to_tensor()`` and
    ``get_gpu_backend().to_numpy()`` instead.
    """
    return get_gpu_backend().xp


def __getattr__(name: str) -> Any:
    if name == "gpu_backend":
        backend = get_gpu_backend()
        globals()["gpu_backend"] = backend
        return backend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
