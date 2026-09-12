import argparse
import json
import logging
import signal
import warnings
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
    """Suppress the highly-repetitive per-gpi warnings emitted by pytesmo.

    The pytesmo validation framework emits one warning per gpi for cases like
    "Not enough observations to calculate metrics." or "No data for dataset".
    On a 9km SMAP×NSMCSMC bbox run this produces hundreds of thousands of
    lines that drown the actual progress information.

    Two layers are used: a ``warnings.filterwarnings`` rule to silence the
    warning categories at the source, and a ``showwarning`` override so any
    warning that does get through the filters is still dropped instead of
    printing to stderr (bypassing the logging pipeline).
    """
    # Silenced warning-message prefixes. Match is substring-based.
    silenced_prefixes = (
        "Not enough observations to calculate metrics.",
        "One or more sample arguments is too small",
        "An input array is constant; the correlation coefficient is not defined.",
        "No data for dataset",
        "Sending large graph of size",  # dask
    )

    def _showwarning(message, category, filename, lineno, file=None, line=None):
        msg = str(message)
        if any(p in msg for p in silenced_prefixes):
            return
        return _ORIGINAL_SHOWWARNING(message, category, filename, lineno, file=file, line=line)

    warnings.showwarning = _showwarning

    # Belt-and-braces: install filter rules at the source so pytesmo's
    # warnings.warn(...) calls never reach the showwarning hook on most
    # Python builds (some packages re-enter the warning pipeline).
    for prefix in silenced_prefixes:
        warnings.filterwarnings("ignore", message=f".*{prefix}.*")


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

    # Keep Dask/distributed at WARNING so their INFO chatter doesn't drown out
    # the validator's own progress/error lines in log files. This only affects
    # the client process; scheduler/worker subprocesses need the dask config
    # below, which Dask serializes into every new process it spawns.
    for noisy in ("dask", "distributed"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
        for name in (noisy, f"distributed.{noisy}", f"{noisy}.distributed"):
            logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger("distributed.client").setLevel(logging.WARNING)
    logging.getLogger("distributed.worker").setLevel(logging.WARNING)

    try:
        import dask.config

        dask.config.set(
            {
                "logging": {
                    "distributed": "warning",
                    "distributed.core": "warning",
                    "distributed.scheduler": "warning",
                    "distributed.nanny": "warning",
                    "distributed.nanny.memory": "warning",
                    "distributed.worker": "warning",
                    "distributed.client": "warning",
                    "distributed.utils": "warning",
                }
            }
        )
    except ImportError:
        pass

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


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    noise_filter = _configure_logging(args.log_level, args.log_file)
    _install_termination_signal_handlers()

    config_path = Path(args.config)
    if not config_path.exists():
        parser.error(f"Config file not found: {config_path}")

    LOGGER.info("CLI started. config=%s dry_run=%s", config_path, args.dry_run)

    try:
        if args.dry_run:
            return _print_dry_run_summary(config_path)

        if args.max_workers is not None:
            if args.max_workers < 1:
                raise ValueError("--max-workers must be >= 1")
            from validator import settings

            settings.MAX_PARALLEL_WORKERS = args.max_workers
            LOGGER.info("Overriding MAX_PARALLEL_WORKERS to %s", settings.MAX_PARALLEL_WORKERS)

        with config_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        val_run = parse_validation_run_config(payload)
        _validate_dataset_inputs(val_run)

        LOGGER.info("Starting validation run from %s", config_path)
        from validator.validation import run_validation

        result = run_validation(val_run)
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
