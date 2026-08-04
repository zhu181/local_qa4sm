import ast
import logging
import os
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from typing import Any

import netCDF4
import numpy as np
import pandas as pd
import pytz
from dateutil.tz import tzlocal
from ismn.interface import ISMN_Interface
from pytesmo.validation_framework.adapters import AnomalyAdapter, AnomalyClimAdapter
from pytesmo.validation_framework.data_manager import DataManager
from pytesmo.validation_framework.metric_calculators import (
    PairwiseIntercomparisonMetrics,
    TripleCollocationMetrics,
    get_dataset_names,
)
from pytesmo.validation_framework.metric_calculators_adapters import (
    SubsetsMetricsAdapter,
    TsDistributor,
)
from pytesmo.validation_framework.results_manager import (
    netcdf_results_manager,
)
from pytesmo.validation_framework.temporal_matchers import (
    make_combined_temporal_matcher,
)
from pytesmo.validation_framework.validation import Validation
from pytz import UTC
from qa4sm_reader.intra_annual_temp_windows import (
    TemporalSubWindowsCreator,
    TemporalSubWindowsFactory,
)
from qa4sm_reader.netcdf_transcription import Pytesmo2Qa4smResultsTranscriber

from validator import settings
from validator.adapters import StabilityMetricsAdapter
from validator.batches import create_jobs, create_upscaling_lut
from validator.filters import setup_filtering
from validator.globals import (
    DEFAULT_TSW,
    END_TIME,
    IRREGULAR_GRIDS,
    ISMN,
    METADATA_TEMPLATE,
    OUTPUT_FOLDER,
    START_TIME,
    TEMPORAL_SUB_WINDOW_SEPARATOR,
    TEMPORAL_SUB_WINDOWS,
)
from validator.graphics import generate_all_graphs
from validator.models import DatasetVersion, ValidationRun, ValidationTask
from validator.readers import adapt_timestamp, create_reader
from validator.utils import first_file_in, mkdir_if_not_exists

__logger = logging.getLogger(__name__)


def _get_actual_time_range(val_run: ValidationRun, dataset_version: DatasetVersion):
    try:
        vs_start = dataset_version.time_range_start
        vs_start_time = datetime.strptime(vs_start, "%Y-%m-%d").date()

        vs_end = dataset_version.time_range_end
        vs_end_time = datetime.strptime(vs_end, "%Y-%m-%d").date()

        val_start_time = val_run.interval_from.date()
        val_end_time = val_run.interval_to.date()

        actual_start = (
            val_start_time.strftime("%Y-%m-%d")
            if val_start_time > vs_start_time
            else vs_start_time.strftime("%Y-%m-%d")
        )
        actual_end = (
            val_end_time.strftime("%Y-%m-%d") if val_end_time < vs_end_time else vs_end_time.strftime("%Y-%m-%d")
        )

    except Exception:
        # exception will arise for ISMN, and for that one we can use entire range
        actual_start = START_TIME
        actual_end = END_TIME

    return [actual_start, actual_end]


def _get_spatial_reference_reader(val_run: ValidationRun) -> tuple[Any, str, dict]:
    ref_reader = create_reader(
        val_run.spatial_reference_configuration.dataset,
        val_run.spatial_reference_configuration.version,
    )

    time_adapted_ref_reader = adapt_timestamp(
        ref_reader,
        val_run.spatial_reference_configuration.dataset,
        val_run.spatial_reference_configuration.version,
    )

    # we do the dance with the filtering below because filter may actually
    # change the original reader, see ismn network selection
    filtered_reader, read_name, read_kwargs = setup_filtering(
        reader=time_adapted_ref_reader,
        filters=list(val_run.spatial_reference_configuration.filters),
        param_filters=list(val_run.spatial_reference_configuration.parametrised_filters),
        dataset=val_run.spatial_reference_configuration.dataset,
        variable=val_run.spatial_reference_configuration.variable,
    )

    while hasattr(ref_reader, "cls"):
        ref_reader = ref_reader.cls

    return ref_reader, read_name, read_kwargs


def set_outfile(validation_run: ValidationRun, run_dir: str):
    outfile = first_file_in(run_dir, ".nc")
    if outfile is not None:
        out_norm = os.path.normpath(outfile)
        root_norm = os.path.normpath(OUTPUT_FOLDER)

        if os.path.isabs(out_norm):
            try:
                in_output_root = os.path.commonpath([out_norm, root_norm]) == root_norm
            except ValueError:
                in_output_root = False

            if in_output_root:
                out_norm = os.path.relpath(out_norm, root_norm)
            else:
                out_norm = os.path.basename(out_norm)

        validation_run.output_file = out_norm


def save_validation_config(validation_run: ValidationRun):
    try:
        with netCDF4.Dataset(
            os.path.join(OUTPUT_FOLDER, validation_run.output_file),
            "a",
            format="NETCDF4",
        ) as ds:
            try:
                if hasattr(validation_run, "spatial_reference_configuration"):
                    dataset = validation_run.spatial_reference_configuration.dataset
                    if dataset and hasattr(dataset, "is_scattered_data"):
                        ds.val_is_scattered_data = str(dataset.is_scattered_data)
            except (AttributeError, TypeError):
                pass
            if validation_run.interval_from is None:
                ds.val_interval_from = "N/A"
            else:
                ds.val_interval_from = validation_run.interval_from.strftime("%Y-%m-%d %H:%M")

            if validation_run.interval_to is None:
                ds.val_interval_to = "N/A"
            else:
                ds.val_interval_to = validation_run.interval_to.strftime("%Y-%m-%d %H:%M")

            j = 1
            for dataset_config in validation_run.dataset_configurations:
                filters = None
                if dataset_config.filters:
                    filters = "; ".join([x.description for x in dataset_config.filters])
                if dataset_config.parametrised_filters:
                    if filters:
                        filters += ";"
                    _list_comp = [
                        pf.filter.description + " " + pf.parameters for pf in dataset_config.parametrised_filters
                    ]
                    try:
                        filters += "; ".join(_list_comp)
                    except TypeError as e:
                        __logger.error(f"Error in save_validation_config: {e}. {filters=}{_list_comp=}")
                        filters = "; ".join(_list_comp)

                if not filters:
                    filters = "N/A"

                if validation_run.spatial_reference_configuration and (
                    dataset_config.id == validation_run.spatial_reference_configuration.id
                ):
                    i = 0  # reference is always 0
                else:
                    i = j
                    j += 1

                # there is no error for variables!!, there were some inconsistency with short and pretty names,
                # and it should be like that now
                ds.setncattr("val_dc_dataset" + str(i), dataset_config.dataset.short_name)
                ds.setncattr("val_dc_version" + str(i), dataset_config.version.short_name)
                ds.setncattr("val_dc_variable" + str(i), dataset_config.variable.pretty_name)
                ds.setncattr("val_dc_unit" + str(i), dataset_config.variable.unit)

                ds.setncattr(
                    "val_dc_dataset_pretty_name" + str(i),
                    dataset_config.dataset.pretty_name,
                )
                ds.setncattr(
                    "val_dc_version_pretty_name" + str(i),
                    dataset_config.version.pretty_name,
                )
                ds.setncattr(
                    "val_dc_variable_pretty_name" + str(i),
                    dataset_config.variable.short_name,
                )

                ds.setncattr("val_dc_filters" + str(i), filters)

                actual_interval_from, actual_interval_to = _get_actual_time_range(
                    validation_run, dataset_config.version.id
                )
                ds.setncattr("val_dc_actual_interval_from" + str(i), actual_interval_from)
                ds.setncattr("val_dc_actual_interval_to" + str(i), actual_interval_to)

                if (validation_run.spatial_reference_configuration is not None) and (
                    dataset_config.id == validation_run.spatial_reference_configuration.id
                ):
                    ds.val_ref = "val_dc_dataset" + str(i)

                    try:
                        ds.setncattr(
                            "val_resolution",
                            validation_run.spatial_reference_configuration.dataset.resolution["value"],
                        )
                        ds.setncattr(
                            "val_resolution_unit",
                            validation_run.spatial_reference_configuration.dataset.resolution["unit"],
                        )
                    # ISMN has null resolution attribute, therefore
                    # we write no output resolution
                    # same is true for user datasets
                    except (AttributeError, TypeError):
                        pass

                if (validation_run.scaling_ref is not None) and (dataset_config.id == validation_run.scaling_ref.id):
                    ds.val_scaling_ref = "val_dc_dataset" + str(i)

                if dataset_config.dataset.short_name in IRREGULAR_GRIDS.keys():
                    grid_stepsize = IRREGULAR_GRIDS[dataset_config.dataset.short_name]
                else:
                    grid_stepsize = "nan"
                ds.setncattr("val_dc_dataset" + str(i) + "_grid_stepsize", grid_stepsize)

            ds.val_scaling_method = validation_run.scaling_method

            ds.val_anomalies = validation_run.anomalies
            if validation_run.anomalies == ValidationRun.CLIMATOLOGY:
                ds.val_anomalies_from = validation_run.anomalies_from.strftime("%Y-%m-%d %H:%M")
                ds.val_anomalies_to = validation_run.anomalies_to.strftime("%Y-%m-%d %H:%M")

            if all(
                x is not None
                for x in [
                    validation_run.min_lat,
                    validation_run.min_lon,
                    validation_run.max_lat,
                    validation_run.max_lon,
                ]
            ):
                ds.val_spatial_subset = (
                    f"[{validation_run.min_lat}, {validation_run.min_lon}, "
                    f"{validation_run.max_lat}, {validation_run.max_lon}]"
                )

    except Exception:
        __logger.exception("Validation configuration could not be stored.")


def _apply_anomaly_adapter(reader, validation_run, dataset_config, read_name):
    if validation_run.anomalies == ValidationRun.MOVING_AVG_35_D:
        return AnomalyAdapter(
            reader,
            window_size=35,
            columns=[dataset_config.variable.short_name],
            read_name=read_name,
        )
    if validation_run.anomalies == ValidationRun.CLIMATOLOGY:
        anomalies_baseline = [
            validation_run.anomalies_from.astimezone(tz=pytz.UTC).replace(tzinfo=None),
            validation_run.anomalies_to.astimezone(tz=pytz.UTC).replace(tzinfo=None),
        ]
        return AnomalyClimAdapter(
            reader,
            columns=[dataset_config.variable.short_name],
            timespan=anomalies_baseline,
            read_name=read_name,
        )
    return reader


def _setup_metric_calculators(
    ds_num, ds_names, validation_run, metadata_template, temp_sub_wdws, temp_sub_wdw_instance, spatial_ref_name
):
    _pairwise_metrics = PairwiseIntercomparisonMetrics(
        metadata_template=metadata_template,
        calc_kendall=False,
    )

    if validation_run.intra_annual_metrics and validation_run.stability_metrics:
        raise ValueError("Both intra_annual_metrics and stability_metrics cannot be True at the same time.")

    tsw_metrics = None
    if validation_run.intra_annual_metrics:
        tsw_metrics = "intra_annual"
    elif validation_run.stability_metrics:
        tsw_metrics = "stability"

    if tsw_metrics:
        if isinstance(temp_sub_wdws, dict):
            adapter_cls = StabilityMetricsAdapter if tsw_metrics == "stability" else SubsetsMetricsAdapter
            pairwise_metrics = adapter_cls(
                calculator=_pairwise_metrics,
                subsets=temp_sub_wdw_instance.custom_temporal_sub_windows,
                group_results="join",
            )
    else:
        pairwise_metrics = _pairwise_metrics

    metric_calculators = {(ds_num, 2): pairwise_metrics.calc_metrics}

    if (len(ds_names) >= 3) and (validation_run.tcol is True):
        _tcol_metrics = TripleCollocationMetrics(
            spatial_ref_name,
            metadata_template=metadata_template,
            bootstrap_cis=validation_run.bootstrap_tcol_cis,
        )
        if isinstance(temp_sub_wdws, dict):
            tcol_metrics = SubsetsMetricsAdapter(
                calculator=_tcol_metrics,
                subsets=temp_sub_wdw_instance.custom_temporal_sub_windows,
                group_results="join",
            )
        elif temp_sub_wdws is None:
            tcol_metrics = _tcol_metrics
        metric_calculators.update({(ds_num, 3): tcol_metrics.calc_metrics})

    return metric_calculators


def _setup_upscaling(validation_run, datasets, spatial_ref_name):
    if validation_run.upscaling_method == "none":
        return None
    upscale_parms = {
        "upscaling_method": validation_run.upscaling_method,
        "temporal_stability": validation_run.temporal_stability,
    }
    upscale_parms["upscaling_lut"] = create_upscaling_lut(
        validation_run=validation_run,
        datasets=datasets,
        spatial_ref_name=spatial_ref_name,
    )
    return upscale_parms


def create_pytesmo_validation(validation_run: ValidationRun):
    ds_list = []
    ds_read_names = []
    spatial_ref_name = None
    scaling_ref_name = None
    temporal_ref_name = None
    spatial_ref_short_name = None

    ds_num = 1
    for dataset_config in validation_run.dataset_configurations:
        reader = create_reader(dataset_config.dataset, dataset_config.version)
        time_adapted_reader = adapt_timestamp(reader, dataset_config.dataset, dataset_config.version)
        reader, read_name, read_kwargs = setup_filtering(
            reader=time_adapted_reader,
            filters=list(dataset_config.filters),
            param_filters=list(dataset_config.parametrised_filters),
            dataset=dataset_config.dataset,
            variable=dataset_config.variable,
        )
        reader = _apply_anomaly_adapter(reader, validation_run, dataset_config, read_name)

        is_spatial_ref = (
            validation_run.spatial_reference_configuration
            and dataset_config.id == validation_run.spatial_reference_configuration.id
        )
        dataset_name = (
            f"0-{dataset_config.dataset.short_name}"
            if is_spatial_ref
            else f"{ds_num}-{dataset_config.dataset.short_name}"
        )
        if not is_spatial_ref:
            ds_num += 1

        ds_list.append(
            (
                dataset_name,
                {
                    "class": reader,
                    "columns": [dataset_config.variable.short_name],
                    "kwargs": read_kwargs,
                    "max_dist": dataset_config.dataset.resolution_in_m,
                },
            )
        )
        ds_read_names.append((dataset_name, read_name))

        if is_spatial_ref:
            spatial_ref_name = dataset_name
            spatial_ref_short_name = dataset_config.dataset.short_name
        if validation_run.scaling_ref and dataset_config.id == validation_run.scaling_ref.id:
            scaling_ref_name = dataset_name
        if (
            validation_run.temporal_reference_configuration
            and dataset_config.id == validation_run.temporal_reference_configuration.id
        ):
            temporal_ref_name = dataset_name

    datasets = dict(ds_list)
    ds_num = len(ds_list)
    period = get_period(validation_run)

    upscale_parms = _setup_upscaling(validation_run, datasets, spatial_ref_name)

    datamanager = DataManager(
        datasets,
        ref_name=spatial_ref_name,
        period=period,
        read_ts_names=dict(ds_read_names),
        upscale_parms=upscale_parms,
    )
    ds_names = get_dataset_names(datamanager.reference_name, datamanager.datasets, n=ds_num)

    metadata_template = METADATA_TEMPLATE["ismn_ref" if spatial_ref_short_name == ISMN else "other_ref"]

    tsw_dict = define_tsw_metrics(validation_run, period)
    temp_sub_wdw_instance = tsw_dict["temp_sub_wdw_instance"]
    temp_sub_wdws = tsw_dict["temp_sub_wdws"]

    metric_calculators = _setup_metric_calculators(
        ds_num,
        ds_names,
        validation_run,
        metadata_template,
        temp_sub_wdws,
        temp_sub_wdw_instance,
        spatial_ref_name,
    )

    scaling_method = (
        None if validation_run.scaling_method == validation_run.NO_SCALING else validation_run.scaling_method
    )
    temporalwindow_size = validation_run.temporal_matching

    val = Validation(
        datasets=datamanager,
        temporal_matcher=make_combined_temporal_matcher(pd.Timedelta(temporalwindow_size / 2, "h")),
        temporal_ref=temporal_ref_name,
        spatial_ref=spatial_ref_name,
        scaling=scaling_method,
        scaling_ref=scaling_ref_name,
        metrics_calculators=metric_calculators,
        period=period,
    )
    return val


def num_gpis_from_job(job):
    try:
        num_gpis = len(job[0])
    except Exception:
        num_gpis = 1

    return num_gpis


def execute_job(validation_run, job, task_id=None, max_retries=1, retry_delay_seconds=1):
    if task_id is None:
        task_id = uuid.uuid4().hex
    numgpis = num_gpis_from_job(job)
    for attempt in range(max_retries + 1):
        __logger.debug(
            f"Executing job {task_id} from validation {validation_run.id}, "
            f"# of gpis: {numgpis}, attempt {attempt + 1}/{max_retries + 1}"
        )
        start_time = datetime.now(tzlocal())
        try:
            val = create_pytesmo_validation(validation_run)

            result = val.calc(
                *job,
                rename_cols=False,
                only_with_reference=True,
                handle_errors="ignore",
                use_gpu=settings.USE_GPU,
            )
            end_time = datetime.now(tzlocal())
            duration = end_time - start_time
            duration = (duration.days * 86400) + duration.seconds
            __logger.debug(
                f"Finished job {task_id} from validation {validation_run.id}, "
                f"took {duration} seconds for {numgpis} gpis"
            )
            return result
        except Exception as e:
            # Handle non-retriable KeyError cases (missing DataFrame columns)
            if isinstance(e, KeyError) and str(e).strip("'") in {"gpi", "frm_class", "status"}:
                __logger.warning(
                    f"Job {task_id} from validation {validation_run.id} hit non-retriable key error {e}. "
                    "Marking job as empty result and continuing."
                )
                return {}
            # Handle pytesmo TripleCollocationMetrics bug with empty DataFrames
            if isinstance(e, ValueError) and "list.remove(x): x not in list" in str(e):
                __logger.warning(
                    "Job %s hit pytesmo tcol bug (empty DataFrame for dummy result). Returning empty result.",
                    task_id,
                )
                return {}
            if attempt < max_retries:
                __logger.warning(
                    "Job %s failed on attempt %s/%s (%s: %s). Retrying in %ss.",
                    task_id,
                    attempt + 1,
                    max_retries + 1,
                    type(e).__name__,
                    str(e)[:120],
                    retry_delay_seconds,
                )
                time.sleep(retry_delay_seconds)
                continue
            raise


def check_and_store_results(job_id, results, save_path):
    if len(results) < 1:
        __logger.warning(f"Potentially problematic job: {job_id} - no results")
        return

    try:
        netcdf_results_manager(results, save_path)
    except OSError as exc:
        # A stale/corrupted nc file from a previous interrupted run can break appends.
        msg = str(exc)
        recoverable = "NetCDF: Unknown file format" in msg or "NetCDF: Write to read only" in msg
        if not recoverable:
            raise

        bad_file = None
        for token in msg.split("'"):
            if token.lower().endswith(".nc"):
                bad_file = token
                break

        if bad_file and os.path.exists(bad_file):
            __logger.warning(
                "Detected corrupted netCDF result file for job %s. Removing %s and retrying write once.",
                job_id,
                bad_file,
            )
            os.remove(bad_file)
            netcdf_results_manager(results, save_path)
            return

        raise


def track_validation_task(validation_run, task_id):
    validation_task = ValidationTask()
    validation_task.validation = validation_run
    validation_task.task_id = uuid.UUID(task_id).hex
    validation_task.save()


def validation_task_cancelled(task_id):
    # stop_running_validation deletes the validation task records.
    # If they don't exist anymore, this task is treated as cancelled.
    return not ValidationTask.objects.filter(task_id=task_id).exists()


def untrack_validation_task(task_id):
    try:
        validation_task = ValidationTask.objects.get(task_id=task_id)
        validation_task.delete()
    except ValidationTask.DoesNotExist:
        __logger.debug(f"Task {task_id} already deleted from db.")


def _count_job_status(results: dict, ngpis: int) -> tuple[int, int]:
    result_key = list(results.keys())[0]
    res = results[result_key]
    status_keys = [s for s in res if "status" in s and not s.split("|")[0].isdigit()]
    ok = res[status_keys[0]] == 0
    for sk in status_keys[1:]:
        ok = ok & (res[sk] == 0)
    nok = sum(ok)
    return nok, ngpis - nok


def _process_job_result(task_id, future, job_table, validation_run, run_dir):
    results = future.result()
    if not results:
        validation_run.error_points += num_gpis_from_job(job_table[task_id])
        return None

    results = _pytesmo_to_qa4sm_results(results)
    check_and_store_results(task_id, results, run_dir)

    ngpis = num_gpis_from_job(job_table[task_id])
    ok_pts, error_pts = _count_job_status(results, ngpis)
    validation_run.ok_points += ok_pts
    validation_run.error_points += error_pts
    return results


def _extend_transcriber_datasets(validation_run):
    """
    Extend the qa4sm_reader transcriber's hard-coded DATASETS list with the
    dataset short names actually used in this run.

    Pytesmo2Qa4smResultsTranscriber.is_valid_tcol_metric_name() only accepts
    tcol metric names whose ``{number}-{dataset}`` prefix matches a dataset in
    the static ``qa4sm_reader.globals.DATASETS`` list. Datasets not in that list
    (e.g. SPL3SMPE, NSMCSMC) would have their tcol metrics (beta/snr/err_std)
    silently dropped during transcription. Patching the ``netcdf_transcription``
    module namespace (the one the method reads at call time) fixes that.
    """
    try:
        import qa4sm_reader.netcdf_transcription as qa4sm_transcription
    except ImportError:
        return

    short_names = {validation_run.spatial_reference_configuration.dataset.short_name}
    for cfg in validation_run.dataset_configurations:
        short_names.add(cfg.dataset.short_name)

    new_datasets = [name for name in sorted(short_names) if name not in qa4sm_transcription.DATASETS]
    if new_datasets:
        qa4sm_transcription.DATASETS = qa4sm_transcription.DATASETS + new_datasets
        __logger.info(
            "Extended qa4sm_reader transcriber DATASETS with %s so tcol metrics are kept for all datasets.",
            new_datasets,
        )


def _post_process_run(validation_run, run_dir, results):
    set_outfile(validation_run, run_dir)
    iam_dict = define_tsw_metrics(validation_run, get_period(validation_run))
    temp_sub_wdw_instance = iam_dict["temp_sub_wdw_instance"]
    temp_sub_wdws = iam_dict["temp_sub_wdws"]

    _extend_transcriber_datasets(validation_run)

    transcriber = Pytesmo2Qa4smResultsTranscriber(
        pytesmo_results=os.path.join(OUTPUT_FOLDER, validation_run.output_file),
        intra_annual_slices=temp_sub_wdw_instance,
        keep_pytesmo_ncfile=False,
    )
    if not transcriber.exists:
        return

    _outname, outname_zarr = transcriber.build_outname(run_dir, results.keys())
    transcriber.output_file_name = str(_outname)
    transcriber.output_zarr_name = str(outname_zarr)
    # Initialize transcribed_dataset before writing — write_to_netcdf relies on
    # self.transcribed_dataset which is only populated by get_transcribed_dataset()
    transcriber.get_transcribed_dataset()
    transcriber.write_to_netcdf(transcriber.output_file_name)
    save_validation_config(validation_run)
    transcriber.compress(path=transcriber.output_file_name, compression="zlib", complevel=9)

    temporal_sub_windows_names = [DEFAULT_TSW] if temp_sub_wdws is None else temp_sub_wdw_instance.names
    __logger.info(f"temporal_sub_windows_names: {temporal_sub_windows_names}")

    generate_all_graphs(
        validation_run=validation_run,
        outfolder=run_dir,
        temporal_sub_windows=temporal_sub_windows_names,
        save_metadata=validation_run.plots_save_metadata,
    )


def _clear_stale_outputs(run_dir):
    stale = [os.path.join(run_dir, name) for name in os.listdir(run_dir) if name.endswith(".nc")]
    if not stale:
        return
    for nc_file in stale:
        try:
            os.remove(nc_file)
        except OSError:
            __logger.warning("Could not remove stale netCDF file %s", nc_file)
    __logger.info(
        "Removed %s stale netCDF file(s) from %s before starting validation.",
        len(stale),
        run_dir,
    )


def _determine_max_workers(jobs, ref_reader):
    configured = getattr(settings, "MAX_PARALLEL_WORKERS", os.cpu_count() or 1)
    max_workers = max(1, min(len(jobs), configured))
    if isinstance(ref_reader, ISMN_Interface) and max_workers > 1:
        __logger.warning("ISMN reference reader is not safe for threaded execution. Forcing max_workers=1.")
        max_workers = 1
    return max_workers


def _gpu_dask_requested() -> bool:
    """True when QA4SM_USE_GPU is set AND a working GPU/CuPy is detected."""
    if not getattr(settings, "USE_GPU", False):
        return False
    try:
        from pytesmo.gpu import is_gpu_available

        available = is_gpu_available()
    except ImportError:
        available = False
    if not available:
        __logger.warning(
            "QA4SM_USE_GPU is set but no GPU/CuPy is available. Falling back to the classic threaded path on CPU."
        )
        settings.USE_GPU = False
        return False
    return True


def _dask_memory_limit():
    """Per-worker memory limit for the Dask GPU path.

    Uses ``QA4SM_DASK_MEMORY_LIMIT`` (Dask-compatible string like "16GB") when
    set, otherwise falls back to 60% of total system memory. Dask's default
    "auto" (~40%) is too tight for memory-hungry readers and triggers
    ``KilledWorker``.
    """
    limit = getattr(settings, "DASK_MEMORY_LIMIT", None)
    if limit is not None:
        return limit
    try:
        import psutil

        total = psutil.virtual_memory().total
    except Exception:
        total = 32 * 1024**3
    return int(0.6 * total)


_RUNTIME_FIELDS = {
    "id",
    "name_tag",
    "total_points",
    "error_points",
    "ok_points",
    "output_file",
    "progress",
}


def _config_hash(validation_run) -> str:
    """Stable hash of the validation config for resume-cache keying.

    Excludes run-specific/runtime fields (id, point counters, output file) and
    any callable/back-reference attributes (e.g. bound `save` methods, parent
    run back-refs) so that re-running the *same* config reuses the same Dask
    batch cache while any config change (datasets, interval, bbox, metrics)
    starts a fresh one.
    """
    import hashlib

    data = {
        k: v
        for k, v in vars(validation_run).items()
        if k not in _RUNTIME_FIELDS and not callable(v)
    }
    return hashlib.sha1(repr(data).encode("utf-8")).hexdigest()[:16]


def _dask_batch_cache_path(validation_run) -> str:
    """Per-config Dask batch zarr cache dir (resume source for the GPU path)."""
    cache_root = os.path.join(OUTPUT_FOLDER, ".dask_batch_cache")
    path = os.path.join(cache_root, _config_hash(validation_run))
    os.makedirs(path, exist_ok=True)
    return path


def _format_duration(seconds: float) -> str:
    seconds = int(seconds)
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _format_eta(elapsed_seconds: float, done: int, total: int) -> str:
    if done <= 0 or total <= 0:
        return "n/a"
    rate = done / max(elapsed_seconds, 1e-6)
    remaining = (total - done) / rate
    return _format_duration(remaining)


def _format_progress(done: int, total: int, elapsed_seconds: float, unit: str = "gpis") -> str:
    pct = 100.0 * done / total if total else 0.0
    return (
        f"{done:,}/{total:,} {unit} ({pct:.1f}%) "
        f"elapsed={_format_duration(elapsed_seconds)} ETA≈{_format_eta(elapsed_seconds, done, total)}"
    )


def _make_gpu_progress_callback(validation_run, log_interval: float = 15.0):
    """Build a throttled progress callback for the Dask-parallel GPU path.

    In addition to progress logging, this callback periodically queries
    worker memory statistics via the Dask scheduler and emits a WARNING
    through the client's logging system when unmanaged memory is high.
    This ensures the diagnostic reaches the log file (the worker-side
    ``distributed.worker.memory`` warning only goes to the worker's stderr).
    """
    milestones = {0.25, 0.50, 0.75, 1.0}
    logged_pct = set()
    last_log = {"t": -float("inf")}
    last_mem_check = {"t": -float("inf")}
    started = time.monotonic()
    mem_check_interval = 60.0  # seconds between worker memory queries

    def _cb(done: int, total: int) -> None:
        now = time.monotonic()
        elapsed = now - started
        pct = 100.0 * done / total if total else 0.0
        throttled = (now - last_log["t"]) >= log_interval
        milestone = False
        for m in milestones:
            if pct >= m * 100 and m not in logged_pct:
                logged_pct.add(m)
                milestone = True
                break
        if throttled or milestone or done >= total:
            last_log["t"] = now
            __logger.info(
                "GPU/Dask progress: %s (validation %s)",
                _format_progress(done, total, elapsed),
                validation_run.id,
            )

        # Periodically check worker memory from the client side so that
        # unmanaged-memory warnings appear in the log file.
        if (now - last_mem_check["t"]) >= mem_check_interval:
            last_mem_check["t"] = now
            try:
                from dask.distributed import get_client

                client = get_client()
                info = client.scheduler_info()
                for addr, winfo in info.get("workers", {}).items():
                    mem = winfo.get("metrics", {}).get("memory", {})
                    process_mem = mem.get("process", 0)
                    managed = mem.get("managed", 0)
                    unmanaged = process_mem - managed if process_mem > managed else 0
                    mem_limit = winfo.get("nbytes", 0)
                    if mem_limit and process_mem:
                        frac = process_mem / mem_limit
                        if frac > 0.7:
                            __logger.warning(
                                "Dask worker %s memory usage high: %.0f%% "
                                "(process=%s managed=%s unmanaged=%s limit=%s)",
                                addr,
                                frac * 100,
                                _fmt_bytes(process_mem),
                                _fmt_bytes(managed),
                                _fmt_bytes(unmanaged),
                                _fmt_bytes(mem_limit),
                            )
            except Exception:
                # Non-fatal: memory monitoring is diagnostic, not essential.
                pass

    return _cb


def _fmt_bytes(n: int) -> str:
    """Format byte count as human-readable string (e.g. '1.5 GiB')."""
    if n <= 0:
        return "0 B"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PiB"


def _run_gpu_dask_validation(validation_run, val, jobs, run_dir):
    """
    Run the whole validation through a single pytesmo ``Validation.calc`` call
    using the Dask-parallel GPU path.

    All jobs are concatenated into one flat gpi/lon/lat (+ metadata) set and
    processed by pytesmo's DaskParallelExecutor (1 GPU per worker), replacing
    the classic ThreadPoolExecutor job loop.
    """
    all_gpis = np.concatenate([job[0] for job in jobs])
    all_lons = np.concatenate([job[1] for job in jobs])
    all_lats = np.concatenate([job[2] for job in jobs])

    # ISMN jobs carry a 4th element: per-gpi metadata dicts; gridded jobs don't.
    meta_list = []
    for job in jobs:
        if len(job) > 3:
            meta_list.extend(job[3])
        else:
            meta_list.extend([{} for _ in range(len(job[0]))])

    n_workers = getattr(settings, "MAX_PARALLEL_WORKERS", os.cpu_count() or 1)
    __logger.info(
        "Running validation %s with Dask-parallel GPU path (%s gpis, %s workers).",
        validation_run.id,
        len(all_gpis),
        n_workers,
    )

    # Streaming accumulator: each completed batch is converted and appended to
    # the run netCDF immediately, so neither the client nor the Dask worker ever
    # holds the full result set in memory.
    state = {"ok": 0, "error": 0, "key": None, "seen": False}

    def _batch_cb(compact_results, n_gpis):
        state["seen"] = True
        if not compact_results:
            state["error"] += n_gpis
            return
        converted = _pytesmo_to_qa4sm_results(compact_results)
        check_and_store_results(str(validation_run.id), converted, run_dir)
        ok_pts, error_pts = _count_job_status(converted, n_gpis)
        state["ok"] += ok_pts
        state["error"] += error_pts
        if state["key"] is None and converted:
            state["key"] = next(iter(converted))

    progress_cb = _make_gpu_progress_callback(validation_run)
    try:
        summary = val.calc(
            all_gpis,
            all_lons,
            all_lats,
            meta_list,
            rename_cols=False,
            only_with_reference=True,
            handle_errors="ignore",
            use_gpu=True,
            parallel="dask",
            n_workers=n_workers,
            batch_size=100,
            output_format="zarr",
            output_path=_dask_batch_cache_path(validation_run),
            progress=True,
            progress_callback=progress_cb,
            batch_callback=_batch_cb,
            parallel_kwargs={
                "dashboard": False,
                "memory_limit": _dask_memory_limit(),
                "memory_target_fraction": 0.6,
                "memory_spill_fraction": 0.55,
            },
        )
    except Exception as e:
        # A failure after every gpi has already been computed and streamed to the
        # output netCDF (e.g. a Dask cluster-teardown timeout) must not trigger
        # the classic-path fallback, which would discard the finished results.
        processed = state["ok"] + state["error"]
        if state["seen"] and processed == len(all_gpis):
            __logger.warning(
                "GPU/Dask validation %s already processed all %d gpis before %s during "
                "teardown; treating the run as successful.",
                validation_run.id,
                processed,
                type(e).__name__,
            )
            summary = {"n_gpis": processed, "batches": 0, "keys": []}
        else:
            raise
    if not summary or summary.get("n_gpis", 0) == 0:
        __logger.warning(f"GPU/Dask validation {validation_run.id} produced no results.")
        raise RuntimeError("GPU/Dask path produced no results; falling back to the classic path.")

    validation_run.ok_points += state["ok"]
    validation_run.error_points += state["error"]
    validation_run.progress = round(
        (validation_run.ok_points + validation_run.error_points) / validation_run.total_points * 100
    )
    # _post_process_run only needs the result keys; hand it a dict whose keys are
    # the qa4sm combination key (mirrors what the classic path passes).
    _post_process_run(validation_run, run_dir, {state["key"]: None})


def _run_classic_validation(validation_run, jobs, run_dir, max_workers):
    """Run the classic ThreadPoolExecutor job loop (per-job validation)."""
    validation_aborted = False
    __logger.info(f"Running validation {validation_run.id} with {max_workers} parallel workers.")
    total_jobs = len(jobs)
    run_started_at = time.monotonic()
    last_heartbeat_at = run_started_at
    heartbeat_interval = getattr(settings, "HEARTBEAT_INTERVAL_SECONDS", 60)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task_id = {}
        job_table = {}
        last_results = None

        for j in jobs:
            task_id = uuid.uuid4().hex
            future = executor.submit(execute_job, validation_run, j, task_id)
            future_to_task_id[future] = task_id
            job_table[task_id] = j
            track_validation_task(validation_run, task_id)

        pending = set(future_to_task_id.keys())
        while pending:
            done, pending = wait(pending, timeout=10, return_when=FIRST_COMPLETED)

            now = time.monotonic()
            if now - last_heartbeat_at >= heartbeat_interval:
                completed_jobs = total_jobs - len(pending)
                elapsed_seconds = now - run_started_at
                __logger.info(
                    "Heartbeat validation %s: %s jobs ok=%s error=%s",
                    validation_run.id,
                    _format_progress(
                        completed_jobs,
                        total_jobs,
                        elapsed_seconds,
                        unit="jobs",
                    ),
                    validation_run.ok_points,
                    validation_run.error_points,
                )
                last_heartbeat_at = now

            if not done:
                if any(validation_task_cancelled(future_to_task_id[f]) for f in pending):
                    validation_aborted = True
                    __logger.debug(f"Validation {validation_run.id} got cancelled while waiting.")
                continue

            for future in done:
                task_id = future_to_task_id[future]
                try:
                    if validation_task_cancelled(task_id):
                        validation_aborted = True

                    if validation_aborted:
                        validation_run.error_points += num_gpis_from_job(job_table[task_id])
                    else:
                        res = _process_job_result(task_id, future, job_table, validation_run, run_dir)
                        if res is not None:
                            last_results = res

                except Exception as e:
                    validation_run.error_points += num_gpis_from_job(job_table[task_id])
                    __logger.error(
                        "Job %s failed after all retries (%s: %s).",
                        task_id,
                        type(e).__name__,
                        str(e)[:120],
                    )
                    if validation_task_cancelled(task_id):
                        validation_aborted = True
                finally:
                    if not validation_task_cancelled(task_id):
                        untrack_validation_task(task_id)

                if not validation_aborted:
                    validation_run.progress = round(
                        (validation_run.ok_points + validation_run.error_points) / validation_run.total_points * 100
                    )
                else:
                    validation_run.progress = -1

    if not validation_aborted and last_results is not None:
        _post_process_run(validation_run, run_dir, last_results)


def run_validation(validation_run: ValidationRun):
    __logger.info(f"Starting validation: {validation_run.id}")

    try:
        run_dir = os.path.join(OUTPUT_FOLDER, str(validation_run.id))
        mkdir_if_not_exists(run_dir)
        _clear_stale_outputs(run_dir)

        ref_reader, read_name, read_kwargs = _get_spatial_reference_reader(validation_run)
        total_points, jobs = create_jobs(
            validation_run=validation_run,
            reader=ref_reader,
            dataset_config=validation_run.spatial_reference_configuration,
        )
        validation_run.total_points = total_points
        __logger.debug(f"Jobs to run: {[job[:-1] for job in jobs]}")

        if _gpu_dask_requested():
            val = create_pytesmo_validation(validation_run)
            try:
                _run_gpu_dask_validation(validation_run, val, jobs, run_dir)
            except Exception as e:
                __logger.warning(
                    "GPU/Dask path failed (%s: %s). Falling back to the classic threaded path with GPU metrics.",
                    type(e).__name__,
                    str(e)[:120],
                )
                validation_run.ok_points = 0
                validation_run.error_points = 0
                _clear_stale_outputs(run_dir)
                max_workers = _determine_max_workers(jobs, ref_reader)
                _run_classic_validation(validation_run, jobs, run_dir, max_workers)
        else:
            max_workers = _determine_max_workers(jobs, ref_reader)
            _run_classic_validation(validation_run, jobs, run_dir, max_workers)

    except Exception:
        __logger.exception(f"Unexpected exception during validation {validation_run}:")

    finally:
        validation_run.end_time = datetime.now(tzlocal())
        __logger.info(
            f"Validation finished: {validation_run}. "
            f"Jobs: {validation_run.total_points}, "
            f"Errors: {validation_run.error_points}, "
            f"OK: {validation_run.ok_points}, "
            f"End time: {validation_run.end_time}"
        )

    return validation_run


def _pytesmo_to_qa4sm_results(results: dict) -> dict:
    """
    Converts the new pytesmo results dictionary format to the old format that
    is still used by QA4SM.

    Parameters
    ----------
    results : dict
        Each key in the dictionary is a tuple of ``((ds1, col1), (d2, col2))``,
        and the values contain the respective results for this combination of
        datasets/columns.

    Returns
    -------
    qa4sm_results : dict
        Dictionary in the format required by QA4SM. This involves merging the
        different dictionary entries from `results` to a single dictionary and
        renaming the metrics to avoid name clashes, using the naming convention
        from the old metric calculators.
    """
    # each key is a tuple of ((ds1, col1), (ds2, col2))
    # this adds all tuples to a single list, and then only
    # keeps unique entries
    qa4sm_key = tuple(sorted(set(sum(map(list, results.keys()), []))))

    qa4sm_res = {qa4sm_key: {}}
    for key in results:
        for metric in results[key]:
            if TEMPORAL_SUB_WINDOW_SEPARATOR in metric:
                prefix = metric.split(TEMPORAL_SUB_WINDOW_SEPARATOR)[0]
                metric = metric.split(TEMPORAL_SUB_WINDOW_SEPARATOR)[1]
            else:
                prefix = None
            # static 'metrics' (e.g. metadata, geoinfo) are not related to datasets
            statics = ["gpi", "lat", "lon"]
            statics.extend(METADATA_TEMPLATE["ismn_ref"])
            if metric in statics:
                new_key = metric
            else:
                datasets = list(map(lambda t: t[0], key))
                if metric[0] == "(" and metric[-1] == ")":
                    metric = ast.literal_eval(metric)  # casts the string representing a tuple to a real tuple
                if isinstance(metric, tuple):
                    new_metric = "_".join(metric)
                else:
                    new_metric = metric
                if prefix:
                    new_metric = f"{prefix}{TEMPORAL_SUB_WINDOW_SEPARATOR}{new_metric}"
                new_key = f"{new_metric}_between_{'_and_'.join(datasets)}"
            if prefix:
                metric = f"{prefix}{TEMPORAL_SUB_WINDOW_SEPARATOR}{metric}"
            qa4sm_res[qa4sm_key][new_key] = results[key][metric]
    return qa4sm_res


def get_period(val_run: ValidationRun) -> None | list[str]:
    """
    Extract the validation period from the validation run object.

    Parameters
    ----------
    val_run : ValidationRun
        The validation run object

    Returns
    -------
    Union[None, List[str]]
        The validation period as a list of two strings, the start and end date,
        respectively. If no period is defined, None is returned.
    """
    if val_run.interval_from is not None and val_run.interval_to is not None:
        # while pytesmo can't deal with timezones, normalise the validation
        # period to utc; can be removed once pytesmo can do timezones
        startdate = val_run.interval_from.astimezone(UTC).replace(tzinfo=None)
        enddate = val_run.interval_to.astimezone(UTC).replace(tzinfo=None)
        return [startdate, enddate]
    return None


def define_tsw_metrics(
    val_run: ValidationRun, period: list
) -> dict[str, TemporalSubWindowsCreator | dict[str, TsDistributor] | None]:
    """
    Extract the temporal sub-window metrics settings from the validation run
    and instantiate the corresponding objects.

    Parameters
    ----------
    val_run : ValidationRun
        The validation run object
    period : List
        The period of a validation run

    Returns
    -------
    Dict[str, Union[TemporalSubWindowsCreator, Dict[str, TsDistributor], None]]
        A dictionary containing the temporal sub-window instance and the custom
        temporal sub-windows, if applicable. Otherwise, filled with None.
    """
    temp_sub_wdw_instance = None

    # Handle intra-annual metrics
    if val_run.intra_annual_metrics:
        intra_annual_metric_lut = {
            "Seasonal": "seasons",
            "Monthly": "months",
        }  # TODO implement properly in qa4sm_reader.globals
        temp_sub_wdw_instance = TemporalSubWindowsFactory.create(
            temporal_sub_window_type=intra_annual_metric_lut[val_run.intra_annual_type],
            overlap=int(val_run.intra_annual_overlap),
            period=period,
        )

    # Handle stability metrics
    elif val_run.stability_metrics:
        temp_sub_wdw_instance = TemporalSubWindowsFactory.create(
            temporal_sub_window_type="stability",
            overlap=0,  # Adjust overlap for stability metrics
            period=period,
            custom_subwindows=TEMPORAL_SUB_WINDOWS.get("custom", None),
        )

    temp_sub_wdws = temp_sub_wdw_instance.custom_temporal_sub_windows if temp_sub_wdw_instance else None

    __logger.debug(f"{temp_sub_wdw_instance=}")
    __logger.debug(f"{temp_sub_wdws=}")
    return {
        "temp_sub_wdw_instance": temp_sub_wdw_instance,
        "temp_sub_wdws": temp_sub_wdws,
    }
