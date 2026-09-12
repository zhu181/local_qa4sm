import json
import tempfile
from pathlib import Path

from validator.cli import _build_parser, main


def test_parser_defaults():
    parser = _build_parser()
    args = parser.parse_args([])
    assert args.dry_run is False
    assert args.log_level == "INFO"
    assert args.max_workers is None
    assert args.log_file is None


def test_parser_custom_config():
    parser = _build_parser()
    args = parser.parse_args(["my_config.json"])
    assert args.config == "my_config.json"


def test_parser_dry_run():
    parser = _build_parser()
    args = parser.parse_args(["--dry-run"])
    assert args.dry_run is True


def test_parser_log_level():
    parser = _build_parser()
    args = parser.parse_args(["--log-level", "DEBUG"])
    assert args.log_level == "DEBUG"


def test_parser_max_workers():
    parser = _build_parser()
    args = parser.parse_args(["--max-workers", "4"])
    assert args.max_workers == 4


def test_parser_log_file():
    parser = _build_parser()
    args = parser.parse_args(["--log-file", "/tmp/test.log"])
    assert args.log_file == "/tmp/test.log"


def test_dry_run_with_template():
    template = Path(__file__).parent.parent / "validator" / "validation_run.template.json"
    exit_code = main([str(template), "--dry-run"])
    assert exit_code == 0


def test_dry_run_invalid_config():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(json.dumps({"validation_run": {"id": 1, "name_tag": "bad"}}))
        f.flush()
        path = f.name

    exit_code = main([path, "--dry-run"])
    assert exit_code == 0


def test_dry_run_broken_json():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("{invalid json}")
        f.flush()
        path = f.name

    exit_code = main([path, "--dry-run"])
    assert exit_code == 1


def test_nonexistent_config():
    import pytest

    with pytest.raises(SystemExit) as exc:
        main(["/nonexistent/path.json", "--dry-run"])
    assert exc.value.code == 2


def test_negative_max_workers():
    parser = _build_parser()
    args = parser.parse_args(["--max-workers", "0"])
    assert args.max_workers == 0
