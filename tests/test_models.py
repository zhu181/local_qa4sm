from datetime import datetime
from uuid import uuid4

from validator.models import (
    DataFilter,
    DataVariable,
    Dataset,
    DatasetConfiguration,
    DatasetVersion,
    ParametrisedFilter,
    ValidationRun,
    ValidationTask,
)


class TestDataFilter:
    def test_defaults(self):
        f = DataFilter(id=1, name="FIL_TEST", description="A test filter", help_text="")
        assert f.name == "FIL_TEST"
        assert not f.parameterised
        assert f.to_include is None

    def test_str(self):
        f = DataFilter(id=1, name="FIL_TEST", description="Test", help_text="")
        assert str(f) == "FIL_TEST (Test)"


class TestDataVariable:
    def test_minimal(self):
        v = DataVariable(id=1, short_name="sm", pretty_name="Soil Moisture", help_text="")
        assert v.short_name == "sm"
        assert v.unit == "n.a."

    def test_with_unit(self):
        v = DataVariable(id=1, short_name="sm", pretty_name="Soil Moisture", help_text="", unit="m3/m3")
        assert v.unit == "m3/m3"


class TestDataset:
    def test_minimal(self):
        d = Dataset(
            id=1, short_name="TEST", pretty_name="Test",
            help_text="", detailed_description="", source_reference="", citation="",
        )
        assert d.short_name == "TEST"
        assert d.reader is None

    def test_resolution_in_m_default(self):
        d = Dataset(
            id=1, short_name="TEST", pretty_name="Test",
            help_text="", detailed_description="", source_reference="", citation="",
        )
        assert d.resolution_in_m == 30e3

    def test_resolution_in_m_km(self):
        d = Dataset(
            id=1, short_name="TEST", pretty_name="Test",
            help_text="", detailed_description="", source_reference="", citation="",
        )
        d.resolution = {"value": 25, "unit": "km"}
        assert d.resolution_in_m == 25e3

    def test_resolution_in_m_deg(self):
        d = Dataset(
            id=1, short_name="TEST", pretty_name="Test",
            help_text="", detailed_description="", source_reference="", citation="",
        )
        d.resolution = {"value": 0.25, "unit": "deg"}
        assert d.resolution_in_m == 0.25 * 100 * 1e3


class TestParametrisedFilter:
    def test_defaults(self):
        pf = ParametrisedFilter()
        assert pf.id == 0
        assert pf.dataset_config is None
        assert pf.filter is None
        assert pf.parameters == ""


class TestDatasetVersion:
    def test_minimal(self):
        v = DatasetVersion(id=1, short_name="v1", pretty_name="Version 1", help_text="")
        assert v.short_name == "v1"

    def test_str(self):
        v = DatasetVersion(id=1, short_name="v1", pretty_name="Version 1", help_text="")
        assert str(v) == "v1"


class TestValidationRun:
    def test_defaults(self):
        run = ValidationRun(id="test-1", name_tag="test")
        assert run.total_points == 0
        assert run.scaling_method == ValidationRun.NO_SCALING
        assert run.anomalies == ValidationRun.NO_ANOM
        assert run.temporal_matching == ValidationRun.TEMP_MATCH_WINDOW

    def test_with_interval(self):
        run = ValidationRun(
            id="test-2", name_tag="test",
            interval_from=datetime(2020, 1, 1),
            interval_to=datetime(2020, 12, 31),
        )
        assert run.interval_from is not None
        assert run.interval_to is not None

    def test_scaling_methods_constants(self):
        assert ValidationRun.MIN_MAX == "min_max"
        assert ValidationRun.LINREG == "linreg"
        assert ValidationRun.NO_SCALING == "none"
        assert ValidationRun.BETA_SCALING == "cdf_beta_match"


class TestDatasetConfiguration:
    def make_run(self):
        return ValidationRun(id="run-1", name_tag="run")

    def make_dataset(self):
        return Dataset(
            id=1, short_name="TEST", pretty_name="Test",
            help_text="", detailed_description="", source_reference="", citation="",
        )

    def test_minimal(self):
        run = self.make_run()
        ds = self.make_dataset()
        cfg = DatasetConfiguration(
            id=101,
            validation=run,
            dataset=ds,
            version=DatasetVersion(id=1, short_name="v1", pretty_name="V1", help_text=""),
            variable=DataVariable(id=1, short_name="sm", pretty_name="SM", help_text=""),
            filters=[],
            parametrised_filters=[],
        )
        assert cfg.id == 101
        assert cfg.dataset.short_name == "TEST"


class TestValidationTask:
    def test_lifecycle(self):
        task_id = uuid4().hex
        task = ValidationTask(task_id=task_id)
        task.save()
        assert ValidationTask.objects.filter(task_id=task_id).exists()
        task.delete()
        assert not ValidationTask.objects.filter(task_id=task_id).exists()

    def test_does_not_exist(self):
        import pytest
        with pytest.raises(ValidationTask.DoesNotExist):
            ValidationTask.objects.get(task_id="nonexistent")

    def test_save_requires_task_id(self):
        import pytest
        task = ValidationTask()
        with pytest.raises(ValueError, match="task_id must be set"):
            task.save()
