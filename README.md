# QA4SM Local Validation

Local command-line tool for validating soil moisture products against reference datasets. Part of the [QA4SM](https://qa4sm.eu) ecosystem.

## Quick Start

```bash
uv run qa4sm-validate validator/validation_run.template.json --dry-run
```

This parses the template configuration and prints a summary without running any validation.

## Configuration

Validation runs are defined in a JSON file. See `validator/validation_run.template.json` for the full schema.

Key fields:

| Field | Description |
|-------|-------------|
| `validation_run.dataset_configurations[].dataset.reader` | Reader class name (e.g. `"SMAPL3_V9Reader"`, `"ISMN_Interface"`, `"GriddedNcOrthoMultiTs"`) |
| `storage_path` | Path to local data directory |
| `interval_from` / `interval_to` | Validation time range |
| `scaling_method` | `"none"`, `"min_max"`, `"linreg"`, `"mean_std"`, `"cdf_beta_match"` |

Readers are selected by the `reader` string in the config, not by dataset `short_name`.

## CLI

```
qa4sm-validate [config] [--dry-run] [--log-level LEVEL] [--log-file PATH] [--max-workers N]
```

- `--dry-run`: parse and validate config without executing
- `--log-level`: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` (default: `INFO`)
- `--log-file`: write logs to file in addition to console
- `--max-workers`: override parallel worker count (env: `QA4SM_MAX_PARALLEL_WORKERS`, default: CPU count)

## Project Structure

```
validator/              Main validation package (CLI, engine, models)
  cli.py                Entry point: argument parsing, logging, orchestration
  orchestrator.py       JSON config → ValidationRun objects
  validation.py         Core validation loop (pytesmo-based)
  models.py             In-memory dataclass models (mimic Django ORM)
  readers.py            Dataset reader factory
  filters.py            Data filtering and masking
  batches.py            Job creation and upscaling LUT
  settings.py           Configuration (output dir, parallel workers)
  graphics.py           Plot generation (via qa4sm-reader)
tests/                  Unit tests (no external data required)
```

### Sub-projects

This repo embeds two standalone packages with their own git history:

- **[qa4sm-reader](./qa4sm-reader/)** — NetCDF result reading and plotting
- **[qa4sm-preprocessing](./qa4sm-preprocessing/)** — Data preprocessing readers

They are available on PyPI as `qa4sm_reader` and `qa4sm_preprocessing`.

## Development

```bash
# Install with dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Lint and format
uv run ruff check .
uv run ruff format .

# Type check
uv run mypy validator
```

Output directory: `outputs/` (controlled by `validator/settings.py:MEDIA_ROOT`).
Stale `.nc` files from previous runs are auto-cleaned on re-run.

## License

See `LICENSE.txt` in each sub-project.
