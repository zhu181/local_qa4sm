import ast
import logging
import os
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from typing import Dict, List, Tuple, Union

import netCDF4
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


def _get_spatial_reference_reader(val_run: ValidationRun) -> Tuple["Reader", str, dict]:
    ref_reader = create_reader(
        val_run.spatial_reference_configuration.dataset,
        val_run.spatial_reference_configuration.version,
    )

    time_adapted_ref_reader = adapt_timestamp(
        ref_reader,
        val_run.spatial_reference_configuration.dataset,
        val_run.spatial_reference_configuration.version,
    )

    # we do the dance with the filtering below because filter may actually change the original reader, see ismn network selection
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
                ds.val_spatial_subset = f"[{validation_run.min_lat}, {validation_run.min_lon}, {validation_run.max_lat}, {validation_run.max_lon}]"

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
    ds_num, ds_names, validation_run, metadata_template, temp_sub_wdws, temp_sub_wdw_instance
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
            validation_run.spatial_reference_configuration.dataset.short_name,
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
            "0-{}".format(dataset_config.dataset.short_name)
            if is_spatial_ref
            else "{}-{}".format(ds_num, dataset_config.dataset.short_name)
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
    except:
        num_gpis = 1

    return num_gpis


def execute_job(validation_run, job, task_id=None, max_retries=1, retry_delay_seconds=1):
    if task_id is None:
        task_id = uuid.uuid4().hex
    numgpis = num_gpis_from_job(job)
    for attempt in range(max_retries + 1):
        __logger.debug(
            "Executing job {} from validation {}, # of gpis: {}, attempt {}/{}".format(
                task_id, validation_run.id, numgpis, attempt + 1, max_retries + 1
            )
        )
        start_time = datetime.now(tzlocal())
        try:
            val = create_pytesmo_validation(validation_run)

            result = val.calc(
                *job,
                rename_cols=False,
                only_with_reference=True,
                handle_errors="ignore",
            )
            end_time = datetime.now(tzlocal())
            duration = end_time - start_time
            duration = (duration.days * 86400) + duration.seconds
            __logger.debug(
                "Finished job {} from validation {}, took {} seconds for {} gpis".format(
                    task_id, validation_run.id, duration, numgpis
                )
            )
            return result
        except Exception as e:
            # Handle non-retriable KeyError cases (missing DataFrame columns)
            if isinstance(e, KeyError) and str(e).strip("'") in {"gpi", "frm_class", "status"}:
                __logger.warning(
                    "Job {} from validation {} hit non-retriable key error {}. "
                    "Marking job as empty result and continuing.".format(task_id, validation_run.id, e)
                )
                return {}
            # Handle pytesmo TripleCollocationMetrics bug with empty DataFrames
            if isinstance(e, ValueError) and "list.remove(x): x not in list" in str(e):
                __logger.warning(
                    "Job %s hit pytesmo tcol bug (empty DataFrame for dummy result). "
                    "Returning empty result.",
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
        __logger.warning("Potentially problematic job: {} - no results".format(job_id))
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
        __logger.debug("Task {} already deleted from db.".format(task_id))


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


def _post_process_run(validation_run, run_dir, results):
    set_outfile(validation_run, run_dir)
    iam_dict = define_tsw_metrics(validation_run, get_period(validation_run))
    temp_sub_wdw_instance = iam_dict["temp_sub_wdw_instance"]
    temp_sub_wdws = iam_dict["temp_sub_wdws"]

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


def run_validation(validation_run: ValidationRun):
    __logger.info("Starting validation: {}".format(validation_run.id))
    validation_aborted = False

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
        __logger.debug("Jobs to run: {}".format([job[:-1] for job in jobs]))

        max_workers = _determine_max_workers(jobs, ref_reader)
        __logger.info("Running validation {} with {} parallel workers.".format(validation_run.id, max_workers))
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
                    elapsed_seconds = int(now - run_started_at)
                    __logger.info(
                        "Heartbeat validation %s: elapsed=%ss progress=%s%% jobs_completed=%s/%s pending=%s",
                        validation_run.id,
                        elapsed_seconds,
                        validation_run.progress,
                        completed_jobs,
                        total_jobs,
                        len(pending),
                    )
                    last_heartbeat_at = now

                if not done:
                    if any(validation_task_cancelled(future_to_task_id[f]) for f in pending):
                        validation_aborted = True
                        __logger.debug("Validation {} got cancelled while waiting.".format(validation_run.id))
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

    except Exception:
        __logger.exception("Unexpected exception during validation {}:".format(validation_run))

    finally:
        validation_run.end_time = datetime.now(tzlocal())
        __logger.info(
            "Validation finished: {}. Jobs: {}, Errors: {}, OK: {}, End time: {} ".format(
                validation_run,
                validation_run.total_points,
                validation_run.error_points,
                validation_run.ok_points,
                validation_run.end_time,
            )
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


def get_period(val_run: ValidationRun) -> Union[None, List[str]]:
    """
    Extract the validation period from the validation run object.

    Parameters
    ----------
    val_run : ValidationRun
        The validation run object

    Returns
    -------
    Union[None, List[str]]
        The validation period as a list of two strings, the start and end date, respectively. If no period is defined, None is returned.
    """
    if val_run.interval_from is not None and val_run.interval_to is not None:
        # while pytesmo can't deal with timezones, normalise the validation period to utc; can be removed once pytesmo can do timezones
        startdate = val_run.interval_from.astimezone(UTC).replace(tzinfo=None)
        enddate = val_run.interval_to.astimezone(UTC).replace(tzinfo=None)
        return [startdate, enddate]
    return None


def define_tsw_metrics(
    val_run: ValidationRun, period: List
) -> Dict[str, Union[TemporalSubWindowsCreator, Dict[str, TsDistributor], None]]:
    """
    Extract the temporal sub-window metrics settings from the validation run and instantiate the corresponding objects.

    Parameters
    ----------
    val_run : ValidationRun
        The validation run object
    period : List
        The period of a validation run

    Returns
    -------
    Dict[str, Union[TemporalSubWindowsCreator, Dict[str, TsDistributor], None]]
        A dictionary containing the temporal sub-window instance and the custom temporal sub-windows, if applicable. Otherwise, filled with None.
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
