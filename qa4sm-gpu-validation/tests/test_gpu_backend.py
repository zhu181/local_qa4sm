from qa4sm_gpu_validation.gpu_backend import GPUBackend, get_xp


def test_gpu_backend_fallback_to_numpy():
    """When no GPU, xp should be numpy."""
    GPUBackend(device_id=0)  # trigger initialization
    # On dev machine without GPU, should fall back to numpy
    # On GPU machine, GPU ops use torch tensors
    xp = get_xp()
    arr = xp.array([1, 2, 3])
    # Both numpy and torch tensors support basic operations
    assert arr.sum() == 6


def test_gpu_backend_get_device_info():
    """get_device_info returns a dict with expected keys."""
    backend = GPUBackend(device_id=0)
    info = backend.get_device_info()
    assert "name" in info
    assert "total_memory_bytes" in info
    assert "free_memory_bytes" in info


def test_gpu_backend_synchronize_noop():
    """synchronize should not raise even without GPU."""
    backend = GPUBackend(device_id=0)
    backend.synchronize()  # should be no-op on CPU


def test_gpu_backend_mem_info():
    """mem_info returns (free, total) tuple."""
    backend = GPUBackend(device_id=0)
    free, total = backend.mem_info()
    assert isinstance(free, int)
    assert isinstance(total, int)


def test_get_xp_returns_array_module():
    """get_xp returns a module with array creation."""
    xp = get_xp()
    assert hasattr(xp, "array")
    assert hasattr(xp, "zeros")
    assert hasattr(xp, "ones")
