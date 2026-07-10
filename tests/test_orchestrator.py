import json
from pathlib import Path

import pytest

from validator.orchestrator import (
    ValidationRun,
    _parse_datetime,
    parse_validation_run_config,
)

SAMPLE_CONFIG = {
    "validation_run": {
        "id": 10001,
        "name_tag": "demo-validation",
        "scaling_method": "none",
        "interval_from": "2020-01-01",
        "interval_to": "2020-12-31",
        "anomalies": "none",
        "temporal_matching": 12,
        "dataset_configurations": [
            {
                "id": 101,
                "is_spatial_reference": True,
                "is_temporal_reference": True,
                "dataset": {
                    "id": 1,
                    "short_name": "REF",
                    "pretty_name": "Reference",
                    "help_text": "",
                    "storage_path": "/path/to/data",
                    "reader": "SMAPL3_V9Reader",
                },
                "version": {
                    "id": 11,
                    "short_name": "v1",
                    "pretty_name": "V1",
                    "help_text": "",
                },
                "variable": {
                    "id": 21,
                    "short_name": "soil_moisture",
                    "pretty_name": "Soil Moisture",
                    "help_text": "",
                    "unit": "m3/m3",
                },
                "filters": [],
                "parametrised_filters": [],
            },
            {
                "id": 102,
                "is_spatial_reference": False,
                "is_temporal_reference": False,
                "is_scaling_reference": True,
                "dataset": {
                    "id": 2,
                    "short_name": "OTHER",
                    "pretty_name": "Other",
                    "help_text": "",
                    "storage_path": "/path/to/other",
                    "reader": "GriddedNcOrthoMultiTs",
                },
                "version": {
                    "id": 12,
                    "short_name": "v1",
                    "pretty_name": "V1",
                    "help_text": "",
                },
                "variable": {
                    "id": 22,
                    "short_name": "soil_moisture",
                    "pretty_name": "Soil Moisture",
                    "help_text": "",
                    "unit": "m3/m3",
                },
                "filters": [],
                "parametrised_filters": [],
            },
        ],
        "spatial_reference_configuration": 101,
        "temporal_reference_configuration": 101,
        "scaling_ref": 102,
    }
}


class TestParseDatetime:
    def test_none(self):
        assert _parse_datetime(None) is None

    def test_empty_string(self):
        assert _parse_datetime("") is None

    def test_iso_date(self):
        dt = _parse_datetime("2020-01-01")
        assert dt is not None
        assert dt.year == 2020
        assert dt.month == 1
        assert dt.day == 1

    def test_iso_with_time(self):
        dt = _parse_datetime("2020-01-01T12:30:00")
        assert dt is not None
        assert dt.hour == 12
        assert dt.minute == 30

    def test_with_z_suffix(self):
        dt = _parse_datetime("2020-01-01T00:00:00Z")
        assert dt is not None

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Unsupported datetime"):
            _parse_datetime("not-a-date")


class TestParseValidationRunConfig:
    def test_full_config(self):
        result = parse_validation_run_config(SAMPLE_CONFIG)
        assert isinstance(result, ValidationRun)
        assert result.name_tag == "demo-validation"
        assert len(result.dataset_configurations) == 2

    def test_reference_resolution(self):
        result = parse_validation_run_config(SAMPLE_CONFIG)
        assert result.spatial_reference_configuration is not None
        assert result.spatial_reference_configuration.id == 101
        assert result.temporal_reference_configuration is not None
        assert result.temporal_reference_configuration.id == 101
        assert result.scaling_ref is not None
        assert result.scaling_ref.id == 102

    def test_dataset_fields(self):
        result = parse_validation_run_config(SAMPLE_CONFIG)
        cfg = result.dataset_configurations[0]
        assert cfg.dataset.short_name == "REF"
        assert cfg.dataset.reader == "SMAPL3_V9Reader"
        assert cfg.version.short_name == "v1"
        assert cfg.variable.short_name == "soil_moisture"

    def test_second_dataset_no_refs(self):
        result = parse_validation_run_config(SAMPLE_CONFIG)
        cfg = result.dataset_configurations[1]
        assert not cfg.is_spatial_reference
        assert not cfg.is_temporal_reference
        assert cfg.is_scaling_reference

    def test_default_spatial_ref_fallback(self):
        config = {
            "validation_run": {
                "id": 20001,
                "name_tag": "fallback",
                "dataset_configurations": [
                    {
                        "id": 1,
                        "dataset": {"id": 1, "short_name": "A", "storage_path": "/data", "reader": "Dummy"},
                        "version": {"id": 1, "short_name": "v1"},
                        "variable": {"id": 1, "short_name": "sm", "pretty_name": "SM", "help_text": ""},
                        "filters": [],
                        "parametrised_filters": [],
                    }
                ],
            }
        }
        result = parse_validation_run_config(config)
        assert result.spatial_reference_configuration is not None
        assert result.spatial_reference_configuration.id == 1

    def test_filters_as_strings(self):
        config = {
            "validation_run": {
                "id": 30001,
                "name_tag": "filters",
                "dataset_configurations": [
                    {
                        "id": 1,
                        "dataset": {"id": 1, "short_name": "A", "storage_path": "/data", "reader": "ISMN_Interface"},
                        "version": {"id": 1, "short_name": "v1"},
                        "variable": {"id": 1, "short_name": "sm", "pretty_name": "SM", "help_text": ""},
                        "filters": ["FIL_ISMN_GOOD"],
                        "parametrised_filters": [],
                    }
                ],
            }
        }
        result = parse_validation_run_config(config)
        assert len(result.dataset_configurations[0].filters) == 1
        assert result.dataset_configurations[0].filters[0].name == "FIL_ISMN_GOOD"

    def test_generates_uuid_id(self):
        result = parse_validation_run_config(SAMPLE_CONFIG)
        assert isinstance(result.id, str)
        assert len(result.id) > 0


class TestRunValidationFromJsonFile:
    def test_with_template_json(self):
        template_path = Path(__file__).parent.parent / "validator" / "validation_run.template.json"
        with template_path.open() as f:
            payload = json.load(f)
        result = parse_validation_run_config(payload)
        assert result is not None
        assert len(result.dataset_configurations) == 2
