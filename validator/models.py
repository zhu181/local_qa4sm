
from dataclasses import dataclass, field
from datetime import datetime
import threading
from typing import ClassVar, Optional


@dataclass
class DatasetConfiguration:
    # validator.models.validation_run.ValidationRun
    id: int
    validation: "ValidationRun"
    dataset: "Dataset"
    version: "DatasetVersion"
    variable: "DataVariable"
    filters: list["DataFilter"]
    parametrised_filters: list["DataFilter"]
    is_spatial_reference: bool = True
    is_temporal_reference: bool = True
    is_scaling_reference: bool = True

    def __str__(self):

        return "Data set: {}, version: {}, variable: {}".format(
            self.dataset if hasattr(self, "dataset") else "none",
            self.version if hasattr(self, "version") else "none",
            self.variable if hasattr(self, "variable") else "none",
        )


@dataclass
class DataVariable:
    id: int
    short_name: str
    pretty_name: str
    help_text: str
    unit: str = "n.a."

    min_value: Optional[float] = None
    max_value: Optional[float] = None
    display_name: str = "n.a."

    # many-to-one relationships coming from other models:
    # dataset_configuration from DatasetConfiguration

    def __str__(self):
        return self.short_name


@dataclass
class DataFilter:
    id: int
    name: str
    description: str
    help_text: str
    parameterised: bool = False
    dialog_name: Optional[str] = None
    default_set_active: bool = False
    default_parameter: Optional[str] = None
    to_include: Optional[str] = None
    to_exclude: Optional[str] = None
    readonly: bool = False

    # many-to-one relationships coming from other models:
    # dataset_configuration from DatasetConfiguration

    def __str__(self):
        return "{} ({})".format(self.name, self.description)


@dataclass
class DatasetVersion:
    id: int
    short_name: str
    pretty_name: str
    help_text: str
    time_range_start: Optional[str] = None
    time_range_end: Optional[str] = None
    geographical_range: Optional[dict] = None
    filters: list = field(default_factory=list)
    variables: list = field(default_factory=list)

    def __str__(self):
        return self.short_name


@dataclass
class Dataset:
    id: int
    short_name: str
    pretty_name: str
    help_text: str
    detailed_description: str
    source_reference: str
    citation: str

    storage_path: str = ""

    is_spatial_reference: bool = False
    is_scattered_data: bool = False

    versions: list = field(default_factory=list)

    resolution = None
    reader: Optional[str] = None
    # many-to-one relationships coming from other models:
    # dataset_configuration from DatasetConfiguration

    def __str__(self):
        return self.short_name

    @property
    def resolution_in_m(self):
        # we need the resolution in m in the distance lookup
        # as a default value we use 30km
        default = 30e3
        if not isinstance(self.resolution, dict):
            return default
        if "value" not in self.resolution or "unit" not in self.resolution:
            return default
        val = self.resolution["value"]
        unit = self.resolution["unit"]
        if unit == "km":
            return val * 1e3
        elif unit == "deg":
            # assuming 1deg approx 100km
            return val * 100 * 1e3
        else:
            return default


@dataclass
class ValidationRun:
    # scaling methods
    MIN_MAX = "min_max"
    LINREG = "linreg"
    MEAN_STD = "mean_std"
    NO_SCALING = "none"
    BETA_SCALING = "cdf_beta_match"

    SCALING_METHODS = (
        (NO_SCALING, "No scaling"),
        (MIN_MAX, "Min/Max"),
        (LINREG, "Linear regression"),
        (MEAN_STD, "Mean/standard deviation"),
        (BETA_SCALING, "CDF matching with beta distribution fitting"),
    )

    # scale to
    SCALE_TO_REF = "ref"
    SCALE_TO_DATA = "data"

    SCALE_TO_OPTIONS = (
        (SCALE_TO_REF, "Scale to reference"),
        (SCALE_TO_DATA, "Scale to data"),
    )

    # anomalies
    MOVING_AVG_35_D = "moving_avg_35_d"
    CLIMATOLOGY = "climatology"
    NO_ANOM = "none"
    ANOMALIES_METHODS = (
        (NO_ANOM, "Do not calculate"),
        (MOVING_AVG_35_D, "35 day moving average"),
        (CLIMATOLOGY, "Climatology"),
    )

    # upscaling options
    NO_UPSCALE = "none"
    AVERAGE = "average"
    UPSCALING_METHODS = (
        (NO_UPSCALE, "Do not upscale point measurements"),
        (AVERAGE, "Average point measurements"),
    )

    # temporal matching window size:
    TEMP_MATCH_WINDOW = 12

    id: int
    name_tag: str
    total_points: int = 0
    error_points: int = 0
    ok_points: int = 0

    spatial_reference_configuration: Optional[DatasetConfiguration] = None
    temporal_reference_configuration: Optional[DatasetConfiguration] = None
    scaling_ref: Optional[DatasetConfiguration] = None

    scaling_method: str = NO_SCALING
    interval_from: Optional[datetime] = None
    interval_to: Optional[datetime] = None
    anomalies: str = NO_ANOM
    min_lat: Optional[float] = None
    min_lon: Optional[float] = None
    max_lat: Optional[float] = None
    max_lon: Optional[float] = None

    # only applicable if anomalies with climatology is selected
    anomalies_from: Optional[datetime] = None
    anomalies_to: Optional[datetime] = None
    # upscaling of ISMN point measurements
    upscaling_method: str = NO_UPSCALE
    temporal_stability: bool = False

    output_file: Optional[str] = None

    tcol: bool = False
    bootstrap_tcol_cis: bool = False

    temporal_matching: int = TEMP_MATCH_WINDOW

    plots_save_metadata: str = "threshold"  #
    # choices:
    #     ("always", "force creating metadata box plots (e.g. for testing)"),
    #     ("never", "do not create metadata box plots at all"),
    #     (
    #         "threshold",
    #         "create metadata box plots only when the minimum "
    #         "number of required points is available "
    #         "(set in globals of qa4sm-reader",
    #     ),

    intra_annual_metrics: bool = False
    intra_annual_type: Optional[str] = None
    intra_annual_overlap: Optional[int] = None

    stability_metrics: bool = False


class ValidationTaskDoesNotExist(Exception):
    pass


class _ValidationTaskQuerySet:
    def __init__(self, matches: list["ValidationTask"]):
        self._matches = matches

    def exists(self) -> bool:
        return len(self._matches) > 0


class ValidationTaskManager:
    def _resolve_task_id(
        self, task_id: Optional[str] = None, **kwargs
    ) -> Optional[str]:
        return task_id or kwargs.get("task_id")

    def filter(
        self, task_id: Optional[str] = None, **kwargs
    ) -> _ValidationTaskQuerySet:
        resolved_task_id = self._resolve_task_id(task_id=task_id, **kwargs)
        if resolved_task_id is None:
            return _ValidationTaskQuerySet([])
        with ValidationTask._lock:
            task = ValidationTask._store.get(resolved_task_id)
        return _ValidationTaskQuerySet([task] if task else [])

    def get(self, task_id: Optional[str] = None, **kwargs) -> "ValidationTask":
        resolved_task_id = self._resolve_task_id(task_id=task_id, **kwargs)
        if resolved_task_id is None:
            raise ValidationTask.DoesNotExist("task_id is required")
        with ValidationTask._lock:
            task = ValidationTask._store.get(resolved_task_id)
        if task is None:
            raise ValidationTask.DoesNotExist(
                f"ValidationTask with task_id={resolved_task_id} does not exist"
            )
        return task


@dataclass
class ValidationTask:
    validation: Optional[ValidationRun] = None
    task_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    _store: ClassVar[dict[str, "ValidationTask"]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def save(self) -> None:
        if self.task_id is None:
            raise ValueError("ValidationTask.task_id must be set before save()")
        with ValidationTask._lock:
            ValidationTask._store[self.task_id] = self

    def delete(self) -> None:
        if self.task_id is None:
            return
        with ValidationTask._lock:
            ValidationTask._store.pop(self.task_id, None)


ValidationTask.DoesNotExist = ValidationTaskDoesNotExist
ValidationTask.objects = ValidationTaskManager()
