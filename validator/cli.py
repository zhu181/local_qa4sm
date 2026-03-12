import argparse
import json
import logging
from pathlib import Path

from validator.orchestrator import (
	parse_validation_run_config,
	run_validation_from_json_file,
)


LOGGER = logging.getLogger(__name__)


def _default_config_path() -> Path:
	return Path(__file__).with_name("validation_run.template.json")


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Run QA4SM validation from a JSON configuration file."
	)
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


def _configure_logging(log_level: str, log_file: str | None = None) -> None:
	level = getattr(logging, log_level.upper(), logging.INFO)
	formatter = logging.Formatter(
		"%(asctime)s | %(levelname)s | %(name)s | %(message)s"
	)

	root_logger = logging.getLogger()
	root_logger.setLevel(level)
	root_logger.handlers.clear()

	console_handler = logging.StreamHandler()
	console_handler.setLevel(level)
	console_handler.setFormatter(formatter)
	root_logger.addHandler(console_handler)

	if log_file:
		log_path = Path(log_file)
		log_path.parent.mkdir(parents=True, exist_ok=True)
		file_handler = logging.FileHandler(log_path, encoding="utf-8")
		file_handler.setLevel(level)
		file_handler.setFormatter(formatter)
		root_logger.addHandler(file_handler)


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
		print(
			"Spatial reference config id: "
			f"{val_run.spatial_reference_configuration.id}"
		)
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
				raise FileNotFoundError(
					f"SMAPTs requires grid.nc but it was not found: {grid_file}"
				)
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
	_configure_logging(args.log_level, args.log_file)

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
			LOGGER.info(
				"Overriding MAX_PARALLEL_WORKERS to %s", settings.MAX_PARALLEL_WORKERS
			)

		with config_path.open("r", encoding="utf-8") as f:
			payload = json.load(f)
		val_run = parse_validation_run_config(payload)
		_validate_dataset_inputs(val_run)

		LOGGER.info("Starting validation run from %s", config_path)
		result = run_validation_from_json_file(config_path)
		LOGGER.info(
			"Validation finished. id=%s total=%s ok=%s error=%s",
			result.id,
			result.total_points,
			result.ok_points,
			result.error_points,
		)
		return 0
	except Exception:
		LOGGER.exception("Validation CLI failed")
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
