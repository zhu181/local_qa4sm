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
    """Fake pytesmo Validation whose calc() records its call args and invokes
    the batch_callback once (mirroring the streaming dask path)."""

    def __init__(self):
        self.calls = {}
        self.summary = {"n_gpis": 3, "batches": 1, "keys": [("a", "sm")]}

    def calc(self, *args, **kwargs):
        self.calls["gpis"] = np.asarray(args[0]).tolist()
        self.calls["meta"] = list(args[3]) if len(args) > 3 else []
        self.calls["kwargs"] = kwargs
        cb = kwargs.get("batch_callback")
        if cb is not None:
            cb({("a", "sm"): {"status": np.array([0, 0])}}, 2)
            return dict(self.summary)
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
    assert kwargs["batch_size"] == 100
    assert kwargs["output_path"]
    assert callable(kwargs["batch_callback"])
    assert kwargs["only_with_reference"] is True
    assert kwargs["parallel_kwargs"]["dashboard"] is False
    assert "memory_limit" in kwargs["parallel_kwargs"]
    assert kwargs["parallel_kwargs"]["memory_target_fraction"] == 0.6
    assert kwargs["parallel_kwargs"]["memory_spill_fraction"] == 0.55
    assert val_run.ok_points == 2
    assert val_run.error_points == 1
    assert val_run.progress == 100
    assert stored["a"][0] == "run1"
    assert posted["a"][0] is val_run
    assert posted["a"][2] == {("a", "sm"): None}


def test_run_gpu_dask_validation_raises_on_empty_results(monkeypatch):
    class EmptyVal:
        def calc(self, *args, **kwargs):
            return {}

    not_called = mock.Mock()
    monkeypatch.setattr(validation, "_pytesmo_to_qa4sm_results", not_called)
    val_run = types.SimpleNamespace(id="run2", ok_points=0, error_points=0, progress=0, total_points=3)
    with pytest.raises(RuntimeError, match="produced no results"):
        validation._run_gpu_dask_validation(val_run, EmptyVal(), _jobs(), "/tmp/run")
    not_called.assert_not_called()
    assert val_run.ok_points == 0


def test_config_hash_stable_and_config_sensitive():
    base = types.SimpleNamespace(
        id="r1", name_tag="x", datasets=["a", "b"], interval=["2017", "2021"],
        total_points=10, ok_points=0, error_points=0, progress=0, output_file="o.nc",
    )
    h1 = validation._config_hash(base)
    # runtime-only changes (id/counters/output) do not change the hash
    base.id = "r2"
    base.ok_points = 5
    base.progress = 50
    assert validation._config_hash(base) == h1
    # config changes (datasets) do
    base.datasets = ["a", "c"]
    assert validation._config_hash(base) != h1


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


def test_setup_metric_calculators_tcol_uses_prefixed_spatial_ref_name(monkeypatch):
    captured = {}

    class FakeTcol:
        def __init__(self, refname, metadata_template=None, bootstrap_cis=False):
            captured["refname"] = refname

        def calc_metrics(self, data, gpi_info):
            pass

    monkeypatch.setattr(validation, "TripleCollocationMetrics", FakeTcol)
    val_run = types.SimpleNamespace(
        tcol=True,
        bootstrap_tcol_cis=False,
        intra_annual_metrics=False,
        stability_metrics=False,
    )
    validation._setup_metric_calculators(
        3, ["0-ISMN", "1-SPL3SMPE", "2-NSMCSMC"], val_run, {}, None, None, "0-ISMN"
    )
    assert captured["refname"] == "0-ISMN"


def test_dask_memory_limit_defaults_to_fraction_of_ram(monkeypatch):
    monkeypatch.setattr(settings, "DASK_MEMORY_LIMIT", None)
    import sys

    fake_psutil = types.SimpleNamespace(virtual_memory=lambda: types.SimpleNamespace(total=40 * 1024**3))
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    limit = validation._dask_memory_limit()
    assert limit == int(0.6 * 40 * 1024**3)


def test_format_duration():
    assert validation._format_duration(0) == "0s"
    assert validation._format_duration(59) == "59s"
    assert validation._format_duration(61) == "1m01s"
    assert validation._format_duration(3600 + 120 + 5) == "1h02m05s"


def test_format_eta_n_a_when_nothing_done():
    assert validation._format_eta(30.0, 0, 100) == "n/a"


def test_format_eta_scales_with_rate():
    assert validation._format_eta(10.0, 5, 10) == "10s"


def test_format_progress_includes_pct_and_eta():
    text = validation._format_progress(25, 100, 50.0)
    assert "25/100 gpis" in text
    assert "25.0%" in text
    assert "elapsed=50s" in text
    assert "ETA≈2m30s" in text


def test_gpu_progress_callback_throttles_and_reports_milestones(monkeypatch, caplog):
    tick = {"t": 0.0}
    monkeypatch.setattr(validation.time, "monotonic", lambda: tick["t"])
    run = types.SimpleNamespace(id="run-p")
    cb = validation._make_gpu_progress_callback(run, log_interval=15.0)

    with caplog.at_level("INFO", logger="validator.validation"):
        tick["t"] = 0.0
        cb(1, 100)  # first call -> throttled (now - 0 >= 15)
        tick["t"] = 5.0
        cb(6, 100)  # within interval, no milestone -> suppressed
        tick["t"] = 20.0
        cb(25, 100)  # throttled again
        tick["t"] = 30.0
        cb(50, 100)  # 50% milestone

    messages = [r.message for r in caplog.records if r.message.startswith("GPU/Dask progress")]
    assert len(messages) == 3
    assert "1/100" in messages[0]
    assert "25/100" in messages[1]
    assert "50/100" in messages[2]

