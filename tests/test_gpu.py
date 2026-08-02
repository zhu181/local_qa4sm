import types
from unittest import mock

import numpy as np
import pytest

from validator import settings, validation


@pytest.fixture(autouse=True)
def _reset_use_gpu():
    settings.USE_GPU = False
    yield
    settings.USE_GPU = False


class _RecordingVal:
    """Fake pytesmo Validation whose calc() records its call args."""

    def __init__(self):
        self.calls = {}

    def calc(self, *args, **kwargs):
        self.calls["gpis"] = np.asarray(args[0]).tolist()
        self.calls["meta"] = list(args[3]) if len(args) > 3 else []
        self.calls["kwargs"] = kwargs
        return {"(('a', 'sm'), ('b', 'sm'))": {"status": np.array([0, 0])}}


def _jobs():
    return [
        (np.array([1, 2]), np.array([10.0, 20.0]), np.array([30.0, 40.0])),
        (np.array([3]), np.array([50.0]), np.array([60.0]), [{"network": "n1"}]),
    ]


def test_run_gpu_dask_validation_concatenates_jobs_and_dispatches_to_dask(monkeypatch):
    monkeypatch.setattr(validation, "_pytesmo_to_qa4sm_results", lambda r: r)
    monkeypatch.setattr(validation, "_count_job_status", lambda r, n: (2, 1))
    stored = {}
    monkeypatch.setattr(validation, "check_and_store_results", lambda *a, **k: stored.update(a=a, k=k))
    posted = {}
    monkeypatch.setattr(validation, "_post_process_run", lambda *a, **k: posted.update(a=a))

    val_run = types.SimpleNamespace(id="run1", ok_points=0, error_points=0, progress=0, total_points=3)
    fake = _RecordingVal()
    validation._run_gpu_dask_validation(val_run, fake, _jobs(), "/tmp/run")

    assert fake.calls["gpis"] == [1, 2, 3]
    assert fake.calls["meta"] == [{}, {}, {"network": "n1"}]
    kwargs = fake.calls["kwargs"]
    assert kwargs["use_gpu"] is True
    assert kwargs["parallel"] == "dask"
    assert kwargs["parallel_kwargs"]["dashboard"] is False
    assert "memory_limit" in kwargs["parallel_kwargs"]
    assert kwargs["only_with_reference"] is True
    assert val_run.ok_points == 2
    assert val_run.error_points == 1
    assert stored["a"][0] == "run1"
    assert posted["a"][0] is val_run


def test_run_gpu_dask_validation_returns_early_on_empty_results(monkeypatch):
    class EmptyVal:
        def calc(self, *args, **kwargs):
            return {}

    not_called = mock.Mock()
    monkeypatch.setattr(validation, "_pytesmo_to_qa4sm_results", not_called)
    val_run = types.SimpleNamespace(id="run2", ok_points=0, error_points=0, progress=0, total_points=3)
    validation._run_gpu_dask_validation(val_run, EmptyVal(), _jobs(), "/tmp/run")
    not_called.assert_not_called()
    assert val_run.ok_points == 0


def test_execute_job_passes_use_gpu_from_settings(monkeypatch):
    monkeypatch.setattr(settings, "USE_GPU", True)
    fake = _RecordingVal()
    monkeypatch.setattr(validation, "create_pytesmo_validation", lambda vr: fake)

    val_run = types.SimpleNamespace(id="run3")
    job = (np.array([1]), np.array([2.0]), np.array([3.0]))
    result = validation.execute_job(val_run, job)

    assert result["(('a', 'sm'), ('b', 'sm'))"]["status"].tolist() == [0, 0]
    assert fake.calls["kwargs"]["use_gpu"] is True


def test_execute_job_passes_use_gpu_false_when_disabled(monkeypatch):
    fake = _RecordingVal()
    monkeypatch.setattr(validation, "create_pytesmo_validation", lambda vr: fake)
    val_run = types.SimpleNamespace(id="run4")
    job = (np.array([1]), np.array([2.0]), np.array([3.0]))
    validation.execute_job(val_run, job)
    assert fake.calls["kwargs"]["use_gpu"] is False


def test_gpu_dask_requested_disabled_when_setting_off(monkeypatch):
    monkeypatch.setattr(settings, "USE_GPU", False)
    assert validation._gpu_dask_requested() is False


def test_gpu_dask_requested_falls_back_when_gpu_unavailable(monkeypatch):
    monkeypatch.setattr(settings, "USE_GPU", True)
    monkeypatch.setattr("pytesmo.gpu.is_gpu_available", lambda: False)
    assert validation._gpu_dask_requested() is False
    assert settings.USE_GPU is False


def test_gpu_dask_requested_enabled_when_gpu_available(monkeypatch):
    monkeypatch.setattr(settings, "USE_GPU", True)
    monkeypatch.setattr("pytesmo.gpu.is_gpu_available", lambda: True)
    assert validation._gpu_dask_requested() is True
    assert settings.USE_GPU is True


def test_dask_memory_limit_uses_env_override(monkeypatch):
    monkeypatch.setattr(settings, "DASK_MEMORY_LIMIT", "16GB")
    assert validation._dask_memory_limit() == "16GB"


def test_dask_memory_limit_defaults_to_fraction_of_ram(monkeypatch):
    monkeypatch.setattr(settings, "DASK_MEMORY_LIMIT", None)
    import sys

    fake_psutil = types.SimpleNamespace(virtual_memory=lambda: types.SimpleNamespace(total=40 * 1024**3))
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    limit = validation._dask_memory_limit()
    assert limit == int(0.6 * 40 * 1024**3)
