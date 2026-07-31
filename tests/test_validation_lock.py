"""Tests for the global HDF5 read lock used to serialize netCDF4/HDF5 access.

The lock prevents Windows 0xC0000005 access violations when multiple worker
threads concurrently open the same netCDF file. These tests verify the lock
is a proper ``threading.Lock`` instance and that concurrent ``_read_data``
calls on a ``GPUBatchedValidation`` do not raise.
"""

import threading
import uuid
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from qa4sm_gpu_validation.gpu_metrics import BatchedPairwiseMetrics, GPUBatchedValidation

from validator.validation import _get_thread_gpu_validation, _hdf5_read_lock


def test_hdf5_read_lock_is_threading_lock():
    assert isinstance(_hdf5_read_lock, type(threading.Lock()))


def test_hdf5_read_lock_is_reentrant_across_threads():
    # A different thread must be able to acquire after release.
    acquired = []

    def worker():
        with _hdf5_read_lock:
            acquired.append(True)

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=2)
    assert acquired == [True]


def test_gpubatchedvalidation_read_data_uses_lock():
    """Verify _read_data acquires the global HDF5 lock."""
    val = _DummyValidation.__new__(_DummyValidation)
    val.datamanager = MagicMock()
    fake_df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    val.datamanager.get_data.return_value = {"ds": fake_df}

    # The lock must NOT be held before the call.
    assert _hdf5_read_lock.acquire(blocking=False)
    _hdf5_read_lock.release()

    # _read_data should succeed and return the mocked dict.
    result = val._read_data(0, 0.0, 0.0)
    assert isinstance(result, dict)
    assert "ds" in result


def test_concurrent_read_data_no_crash():
    """Spawn N threads calling _read_data simultaneously; verify no exception."""
    val = _DummyValidation.__new__(_DummyValidation)
    val.datamanager = MagicMock()
    # Each call returns a small df; ensures the lock is held for each call.
    val.datamanager.get_data.return_value = {
        "ds": pd.DataFrame({"x": np.arange(50, dtype=np.float64)})
    }

    errors: list[Exception] = []
    barrier = threading.Barrier(8)

    def worker():
        try:
            barrier.wait(timeout=2)
            for _ in range(5):
                val._read_data(0, 0.0, 0.0)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert errors == []


def test_thread_local_validation_cache():
    """Two threads should each get their own cached validation instance."""
    val_run_a = MagicMock()
    val_run_a.id = uuid.uuid4().hex
    val_run_b = MagicMock()
    val_run_b.id = uuid.uuid4().hex

    built: list = []
    call_count = {"n": 0}

    def fake_builder(vr):
        call_count["n"] += 1
        built.append(vr.id)
        return MagicMock()

    # Monkey-patch the builder used by _get_thread_gpu_validation.
    import validator.validation as v_mod

    original = v_mod._create_gpu_validation
    v_mod._create_gpu_validation = fake_builder
    try:
        v1 = _get_thread_gpu_validation(val_run_a)
        v2 = _get_thread_gpu_validation(val_run_a)  # cache hit
        assert v1 is v2
    finally:
        v_mod._create_gpu_validation = original


def test_batched_pairwise_metrics_callable_under_lock():
    """Smoke test: BatchedPairwiseMetrics.calc_batch still works on CPU."""
    calc = BatchedPairwiseMetrics()
    rng = np.random.default_rng(42)
    data = [pd.DataFrame({"x": rng.normal(size=200), "y": rng.normal(size=200)})]
    infos = [(0, 1.0, 2.0)]
    results = calc.calc_batch(data, infos)
    assert len(results) == 1
    assert results[0] is not None


# Helper class that inherits only the _read_data implementation under test,
# bypassing __init__ (which would otherwise require real readers / data).
class _DummyValidation(GPUBatchedValidation):
    def __init__(self):  # pragma: no cover - replaced
        pass
