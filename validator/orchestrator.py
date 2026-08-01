import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MethodType
from typing import Any
from uuid import uuid4

from validator.models import (
    DataFilter,
    Dataset,
    DatasetConfiguration,
    DatasetVersion,
    DataVariable,
    ParametrisedFilter,
    ValidationRun,
)


class Collection(list):
    """A tiny list wrapper that mimics the .all() pattern used in validation code."""

    def all(self):
        return self


@dataclass
class OutputFile:
    name: str = ""


def _parse_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
    raise ValueError(f"Unsupported datetime value: {value!r}")


def _parse_filter(payload: Any) -> DataFilter:
    # Compact syntax is supported: "FIL_ISMN_GOOD"
    if isinstance(payload, str):
        name = payload
        return DataFilter(id=0, name=name, description=name, help_text="")

    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported filter definition: {payload!r}")

    name = payload["name"]
    return DataFilter(
        id=payload.get("id", 0),
        name=name,
        description=payload.get("description", name),
        help_text=payload.get("help_text", ""),
    )


def _parse_variable(payload: dict[str, Any]) -> DataVariable:
    return DataVariable(
        id=payload.get("id", 0),
        short_name=payload["short_name"],
        pretty_name=payload.get("pretty_name", payload["short_name"]),
        help_text=payload.get("help_text", ""),
        unit=payload.get("unit", "n.a."),
        min_value=payload.get("min_value"),
        max_value=payload.get("max_value"),
        display_name=payload.get("display_name", "n.a."),
    )


def _parse_version(payload: dict[str, Any]) -> DatasetVersion:
    return DatasetVersion(
        id=payload.get("id", 0),
        short_name=payload["short_name"],
        pretty_name=payload.get("pretty_name", payload["short_name"]),
        help_text=payload.get("help_text", ""),
        time_range_start=payload.get("time_range_start"),
        time_range_end=payload.get("time_range_end"),
        geographical_range=payload.get("geographical_range"),
        filters=payload.get("filters", []),
        variables=payload.get("variables", []),
    )


def _parse_dataset(payload: dict[str, Any]) -> Dataset:
    return Dataset(
        id=payload.get("id", 0),
        short_name=payload["short_name"],
        pretty_name=payload.get("pretty_name", payload["short_name"]),
        help_text=payload.get("help_text", ""),
        detailed_description=payload.get("detailed_description", ""),
        source_reference=payload.get("source_reference", ""),
        citation=payload.get("citation", ""),
        storage_path=payload.get("storage_path", ""),
        is_spatial_reference=payload.get("is_spatial_reference", False),
        is_scattered_data=payload.get("is_scattered_data", False),
        versions=payload.get("versions", []),
        reader=payload.get("reader", ""),
    )


def _parse_dataset_configuration(payload: dict[str, Any], validation: ValidationRun):
    filters = Collection(_parse_filter(item) for item in payload.get("filters", []))
    parametrised_filters = Collection(
        ParametrisedFilter(
            id=0,
            dataset_config=None,
            filter=_parse_filter(item.get("filter", item.get("name"))),
            parameters=item.get("parameters", ""),
        )
        for item in payload.get("parametrised_filters", [])
    )

    config = DatasetConfiguration(
        id=payload.get("id", 0),
        validation=validation,
        dataset=_parse_dataset(payload["dataset"]),
        version=_parse_version(payload["version"]),
        variable=_parse_variable(payload["variable"]),
        filters=filters,
        parametrised_filters=parametrised_filters,
        is_spatial_reference=payload.get("is_spatial_reference", True),
        is_temporal_reference=payload.get("is_temporal_reference", True),
        is_scaling_reference=payload.get("is_scaling_reference", True),
    )
    return config


def _resolve_config_ref(raw_ref: Any, config_by_id: dict[int, DatasetConfiguration]) -> DatasetConfiguration | None:
    if raw_ref is None:
        return None
    if isinstance(raw_ref, int):
        return config_by_id.get(raw_ref)
    if isinstance(raw_ref, dict):
        ref_id = raw_ref.get("id")
        if ref_id is None:
            return None
        return config_by_id.get(ref_id)
    raise ValueError(f"Unsupported dataset configuration reference: {raw_ref!r}")


def parse_validation_run_config(payload: dict[str, Any]) -> ValidationRun:
    raw = payload.get("validation_run", payload)
    runtime_run_id = str(uuid4())
    config_run_id = raw.get("id")

    val_run = ValidationRun(
        id=runtime_run_id,
        name_tag=raw.get("name_tag", f"validation-{runtime_run_id}"),
        total_points=raw.get("total_points", 0),
        error_points=raw.get("error_points", 0),
        ok_points=raw.get("ok_points", 0),
        scaling_method=raw.get("scaling_method", ValidationRun.NO_SCALING),
        interval_from=_parse_datetime(raw.get("interval_from")),
        interval_to=_parse_datetime(raw.get("interval_to")),
        anomalies=raw.get("anomalies", ValidationRun.NO_ANOM),
        min_lat=raw.get("min_lat"),
        min_lon=raw.get("min_lon"),
        max_lat=raw.get("max_lat"),
        max_lon=raw.get("max_lon"),
        anomalies_from=_parse_datetime(raw.get("anomalies_from")),
        anomalies_to=_parse_datetime(raw.get("anomalies_to")),
        upscaling_method=raw.get("upscaling_method", ValidationRun.NO_UPSCALE),
        temporal_stability=raw.get("temporal_stability", False),
        output_file=raw.get("output_file"),
        tcol=raw.get("tcol", False),
        bootstrap_tcol_cis=raw.get("bootstrap_tcol_cis", False),
        gpu_enabled=bool(raw.get("gpu", False)),
        gpu_device_id=int(raw.get("gpu_device_id", 0)),
        batch_size=int(raw.get("batch_size", 1000)),
        cache_size_mb=int(raw.get("cache_size_mb", 2048)),
        temporal_matching=raw.get("temporal_matching", ValidationRun.TEMP_MATCH_WINDOW),
        plots_save_metadata=raw.get("plots_save_metadata", "threshold"),
        intra_annual_metrics=raw.get("intra_annual_metrics", False),
        intra_annual_type=raw.get("intra_annual_type"),
        intra_annual_overlap=raw.get("intra_annual_overlap"),
        stability_metrics=raw.get("stability_metrics", False),
    )

    raw_dataset_configs = raw.get("dataset_configurations", [])
    dataset_configs = Collection(_parse_dataset_configuration(cfg, val_run) for cfg in raw_dataset_configs)
    config_by_id = {cfg.id: cfg for cfg in dataset_configs}

    spatial_ref = _resolve_config_ref(raw.get("spatial_reference_configuration"), config_by_id)
    temporal_ref = _resolve_config_ref(raw.get("temporal_reference_configuration"), config_by_id)
    scaling_ref = _resolve_config_ref(raw.get("scaling_ref"), config_by_id)

    if spatial_ref is None and dataset_configs:
        spatial_ref = dataset_configs[0]

    val_run.spatial_reference_configuration = spatial_ref
    val_run.temporal_reference_configuration = temporal_ref
    val_run.scaling_ref = scaling_ref
    val_run.dataset_configurations = dataset_configs
    val_run.progress = raw.get("progress", 0)

    if isinstance(val_run.output_file, str):
        val_run.output_file = OutputFile(name=val_run.output_file)
    elif val_run.output_file is None:
        val_run.output_file = OutputFile()

    # run_validation expects django-like model behavior, so we add no-op save.
    def _save(self):
        return None

    val_run.save = MethodType(_save, val_run)
    val_run.config_id = config_run_id
    return val_run


def run_validation_from_json_file(json_file: str | Path) -> ValidationRun:
    from validator.validation import run_validation

    with open(json_file, encoding="utf-8") as f:
        payload = json.load(f)
    val_run = parse_validation_run_config(payload)
    return run_validation(val_run)


def run_validation_from_json_string(json_content: str) -> ValidationRun:
    from validator.validation import run_validation

    payload = json.loads(json_content)
    val_run = parse_validation_run_config(payload)
    return run_validation(val_run)
