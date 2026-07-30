import argparse
import json
import logging
import signal
import warnings
from datetime import datetime
from pathlib import Path

from validator.orchestrator import (
    parse_validation_run_config,
)

LOGGER = logging.getLogger(__name__)
_ORIGINAL_SHOWWARNING = warnings.showwarning


def _install_termination_signal_handlers() -> None:
    """Convert termination signals into a logged KeyboardInterrupt path."""

    def _handler(signum, _frame):
        try:
            signame = signal.Signals(signum).name
        except Exception:
            signame = f"signal-{signum}"
        LOGGER.warning(
            "Received %s. Stopping validation run gracefully...",
            signame,
        )
        raise KeyboardInterrupt()

    for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
        if sig is None:
            continue
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            # Signal registration may fail in non-main threads or constrained runtimes.
            continue


def _install_warning_suppression() -> None:
    def _showwarning(message, category, filename, lineno, file=None, line=None):
        msg = str(message)
        if (
            "Not enough observations to calculate metrics." in msg
            or "One or more sample arguments is too small; all returned values will be NaN." in msg
            or "An input array is constant; the correlation coefficient is not defined." in msg
            or "No data for dataset" in msg
        ):
            return
        return _ORIGINAL_SHOWWARNING(message, category, filename, lineno, file=file, line=line)

    warnings.showwarning = _showwarning


class _ValidationNoiseFilter(logging.Filter):
    """Suppress highly repetitive non-fatal records while keeping first samples."""

    def __init__(self, keep_first: int = 5):
        super().__init__()
        self.keep_first = keep_first
        self._counts = {
            "no_data_for_gpi": 0,
            "no_temporal_match": 0,
            "warn_not_enough_obs": 0,
            "warn_small_sample": 0,
            "warn_constant_input": 0,
            "warn_no_data_dataset": 0,
            "pytesmo_tcol_bug": 0,
        }

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "No data for gpi" in msg:
            self._counts["no_data_for_gpi"] += 1
            return self._counts["no_data_for_gpi"] <= self.keep_first
        if "No temporally matched data" in msg:
            self._counts["no_temporal_match"] += 1
            return self._counts["no_temporal_match"] <= self.keep_first
        if "Not enough observations to calculate metrics." in msg:
            self._counts["warn_not_enough_obs"] += 1
            return self._counts["warn_not_enough_obs"] <= self.keep_first
        if "SmallSampleWarning" in msg:
            self._counts["warn_small_sample"] += 1
            return self._counts["warn_small_sample"] <= self.keep_first
        if "ConstantInputWarning" in msg:
            self._counts["warn_constant_input"] += 1
            return self._counts["warn_constant_input"] <= self.keep_first
        if "No data for dataset" in msg:
            self._counts["warn_no_data_dataset"] += 1
            return self._counts["warn_no_data_dataset"] <= self.keep_first
        if "list.remove(x): x not in list" in msg:
            self._counts["pytesmo_tcol_bug"] += 1
            return self._counts["pytesmo_tcol_bug"] <= self.keep_first
        return True

    def emit_summary(self) -> None:
        suppressed_gpi = max(0, self._counts["no_data_for_gpi"] - self.keep_first)
        suppressed_temporal = max(0, self._counts["no_temporal_match"] - self.keep_first)
        if suppressed_gpi > 0:
            LOGGER.info(
                "Suppressed %s repetitive 'No data for gpi' records.",
                suppressed_gpi,
            )
        if suppressed_temporal > 0:
            LOGGER.info(
                "Suppressed %s repetitive 'No temporally matched data' records.",
                suppressed_temporal,
            )

        for key, label in [
            ("warn_not_enough_obs", "Not enough observations warnings"),
            ("warn_small_sample", "Small sample warnings"),
            ("warn_constant_input", "Constant input warnings"),
            ("warn_no_data_dataset", "No data for dataset warnings"),
            ("pytesmo_tcol_bug", "Pytesmo tcol dummy result errors"),
        ]:
            suppressed = max(0, self._counts[key] - self.keep_first)
            if suppressed > 0:
                LOGGER.info("Suppressed %s repetitive %s.", suppressed, label)


def _default_config_path() -> Path:
    return Path(__file__).with_name("validation_run.template.json")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run QA4SM validation from a JSON configuration file.")
    parser.add_argument(
        "config",
        nargs="?",
        default=str(_default_config_path()),
        help="Path to JSON validation config (default: validator/validation_run.template.json).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only parse and validate the JSON structure without executing run_validation.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO).",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional path to a log file. If set, logs are written to both console and file.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Optional override for parallel worker count used by validation.",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration and use CPU-only execution.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override grid points per job batch (default: 500 for GPU, 100 for CPU).",
    )
    parser.add_argument(
        "--cache-size-mb",
        type=int,
        default=None,
        help="Override time series cache size in MB (default: 512).",
    )
    parser.add_argument(
        "--gpu-device",
        type=int,
        default=None,
        help="Select GPU device index (default: 0).",
    )
    return parser


def _configure_logging(log_level: str, log_file: str | None = None) -> _ValidationNoiseFilter:
    level = getattr(logging, log_level.upper(), logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    noise_filter = _ValidationNoiseFilter()

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.filters.clear()
    root_logger.addFilter(noise_filter)
    logging.captureWarnings(True)
    _install_warning_suppression()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(noise_filter)
    root_logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(noise_filter)
        root_logger.addHandler(file_handler)

    # Suppress verbose pytesmo internal logging (e.g. full GPI metadata dumps).
    logging.getLogger("pytesmo").setLevel(logging.CRITICAL)

    # Keep warnings enabled, but limit duplicates via logging filter above.
    warnings.simplefilter("default")

    return noise_filter


def _print_dry_run_summary(config_path: Path) -> int:
    LOGGER.info("Running dry-run parse for config: %s", config_path)
    with config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    val_run = parse_validation_run_config(payload)
    print("Dry run succeeded.")
    print(f"Validation id: {val_run.id}")
    print(f"Name tag: {val_run.name_tag}")
    print(f"Dataset configurations: {len(val_run.dataset_configurations)}")
    if val_run.spatial_reference_configuration is not None:
        print(f"Spatial reference config id: {val_run.spatial_reference_configuration.id}")
    LOGGER.info(
        "Dry-run parsed validation id=%s with %s dataset configurations",
        val_run.id,
        len(val_run.dataset_configurations),
    )
    return 0


def _validate_dataset_inputs(val_run) -> None:
    for cfg in val_run.dataset_configurations:
        dataset = cfg.dataset
        storage_path = Path(dataset.storage_path)
        if not storage_path.exists():
            raise FileNotFoundError(
                f"Dataset storage_path does not exist: {storage_path} (dataset: {dataset.short_name})"
            )
        if not storage_path.is_dir():
            raise NotADirectoryError(
                f"Dataset storage_path is not a directory: {storage_path} (dataset: {dataset.short_name})"
            )

        # Reader-specific preflight checks for earlier and clearer failures.
        if getattr(dataset, "reader", None) == "SMAPTs":
            grid_file = storage_path / "grid.nc"
            if not grid_file.exists():
                raise FileNotFoundError(f"SMAPTs requires grid.nc but it was not found: {grid_file}")
            try:
                with grid_file.open("rb") as f:
                    f.read(16)
            except OSError as e:
                raise OSError(
                    f"SMAPTs grid.nc is not readable: {grid_file}. "
                    "This often indicates disk/share I/O issues (e.g., mapped drive problems or file corruption)."
                ) from e


def _default_log_file(config_path: Path) -> Path:
    from validator.settings import MEDIA_ROOT

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_stem = config_path.stem
    log_dir = Path(MEDIA_ROOT)
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"qa4sm_{timestamp}_{config_stem}.log"


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        parser.error(f"Config file not found: {config_path}")

    log_file = args.log_file
    if log_file is None:
        log_file = str(_default_log_file(config_path))

    noise_filter = _configure_logging(args.log_level, log_file)
    _install_termination_signal_handlers()

    LOGGER.info("CLI started. config=%s dry_run=%s log_file=%s", config_path, args.dry_run, log_file)

    try:
        if args.dry_run:
            return _print_dry_run_summary(config_path)

        if args.max_workers is not None:
            if args.max_workers < 1:
                raise ValueError("--max-workers must be >= 1")
            from validator import settings

            settings.MAX_PARALLEL_WORKERS = args.max_workers
            LOGGER.info("Overriding MAX_PARALLEL_WORKERS to %s", settings.MAX_PARALLEL_WORKERS)

        if args.no_gpu:
            from validator import settings

            settings.GPU_ENABLED = False
            LOGGER.info("GPU acceleration disabled via --no-gpu")

        if args.batch_size is not None:
            if args.batch_size < 1:
                raise ValueError("--batch-size must be >= 1")
            from validator import settings

            settings.GPU_BATCH_SIZE = args.batch_size
            LOGGER.info("Overriding GPU_BATCH_SIZE to %s", settings.GPU_BATCH_SIZE)

        if args.cache_size_mb is not None:
            if args.cache_size_mb < 0:
                raise ValueError("--cache-size-mb must be >= 0")
            from validator import settings

            settings.TS_CACHE_SIZE_MB = args.cache_size_mb
            LOGGER.info("Overriding TS_CACHE_SIZE_MB to %s", settings.TS_CACHE_SIZE_MB)

        if args.gpu_device is not None:
            from validator import settings

            settings.GPU_DEVICE_ID = args.gpu_device
            LOGGER.info("Overriding GPU_DEVICE_ID to %s", settings.GPU_DEVICE_ID)

        with config_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        val_run = parse_validation_run_config(payload)
        _validate_dataset_inputs(val_run)

        LOGGER.info("Starting validation run from %s", config_path)
        from validator.validation import run_validation

        result = run_validation(val_run, val_run_payload=payload)
        LOGGER.info(
            "Validation finished. id=%s config_id=%s total=%s ok=%s error=%s",
            result.id,
            getattr(result, "config_id", "N/A"),
            result.total_points,
            result.ok_points,
            result.error_points,
        )
        return 0
    except KeyboardInterrupt:
        LOGGER.warning("Validation CLI interrupted before completion.")
        return 130
    except Exception:
        LOGGER.exception("Validation CLI failed")
        return 1
    finally:
        noise_filter.emit_summary()


if __name__ == "__main__":
    raise SystemExit(main())
