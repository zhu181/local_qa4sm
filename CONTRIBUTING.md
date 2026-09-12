# Contributing

Thanks for your interest in `local_qa4sm`. This guide covers the
mechanics of contributing; the architectural picture is in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and the agent memory
file is [`AGENTS.md`](AGENTS.md).

## Ground rules

- Be respectful. We're a small project; signal clearly, ask early.
- No CLA. The MIT license ([LICENSE](LICENSE)) covers all
  contributions.
- Keep PRs focused. One logical change per PR; multiple commits are
  fine when they tell a story, but the diff itself should stay
  reviewable.

## Branch & commit conventions

### Branch names

- `feat/<short-topic>` — new feature
- `fix/<short-topic>` — bug fix
- `chore/<short-topic>` — tooling / dependencies / non-feature work
- `docs/<short-topic>` — documentation only
- `refactor/<short-topic>` — internal restructure with no behaviour change

### Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<optional-scope>): <short summary>

<optional body — explain the why, not the what>
<optional footer — refs, breaking-change notes>
```

Types used in this repo (matches the git history):

| Type | Use for |
|---|---|
| `feat` | New user-facing functionality |
| `fix` | Bug fix |
| `chore` | Tooling, dependencies, non-feature work |
| `docs` | Documentation only |
| `style` | Formatting / lint fixes that don't change semantics |
| `test` | Test-only changes |
| `refactor` | Internal restructure, no behaviour change |

Examples from the repo:

```
feat: Dask streaming path as default with classic escape hatch
fix: classic-path netCDF4 write coalescing + corruption reopen
chore: bump pytesmo submodule to 385277e
docs: rewrite README.md for public viewers
```

## Pre-PR checklist

Run these locally and confirm they're green before opening a PR:

```bash
# Format check (CI fails on this)
uv run ruff format --check .

# Lint (auto-fix with `uv run ruff check . --fix` if you want)
uv run ruff check .

# Tests
uv run pytest

# (Optional) pytesmo submodule tests
cd pytesmo && uv run pytest \
    --ignore=tests/test_docs/test_examples.py \
    --ignore=tests/test_validation_framework/test_adapters.py \
    --ignore=tests/test_validation_framework/test_validation.py \
    --ignore=tests/test_io_formats.py \
    --ignore=tests/test_timedate/test_julian.py
```

If your change touches environment variables, update
[`AGENTS.md`](AGENTS.md) in the same commit (it is the canonical
reference for what each `QA4SM_*` env does).

If your change touches user-visible behaviour, add a one-line entry
to the `## [Unreleased]` section of [`CHANGELOG.md`](CHANGELOG.md).

## Pytesmo submodule workflow

`pytesmo/` is a git submodule pinned to the `zhu181/pytesmo` fork.
Edits to pytesmo go through a separate workflow — **do not edit it
from the parent**:

```bash
cd pytesmo
git checkout -b feat/<short-topic>
# ... edit / test ...
git commit -m "feat(parallel): ..."
# (optionally push: git push origin feat/<short-topic>)

# Back in the parent, bump the submodule pointer
cd ..
git add pytesmo
git commit -m "chore: bump pytesmo submodule to <new-sha>"
```

The submodule has its own `AGENTS.md` (read it before editing) and
its own test suite. Run its tests with `uv run pytest` from inside
`pytesmo/`.

The five `--ignore` flags listed in the pre-PR checklist above
deselect modules that depend on optional reader packages
(`ascat`, `ismn`) or notebook libraries (`nbformat`, `zarr`,
`pytz`). On a fresh checkout these are expected to be missing.

## Code conventions

- Python ≥ 3.13, managed with `uv`.
- Linter: `ruff` (rules `E,F,I,N,W,UP`); line-length 120; double
  quotes. Configured in `pyproject.toml`.
- Tests live in `tests/`. The pytesmo submodule has its own
  `tests/`.
- Models in `validator/models.py` are in-memory dataclasses
  mimicking Django ORM (`objects.filter()`, `.get()`, `.save()`,
  `.delete()`). No DB.
- Output directory: `outputs/` (controlled by
  `validator/settings.py:MEDIA_ROOT`). Stale `.nc` files are
  auto-cleaned on re-run.
- `outputs/`, `presets/`, `data/`, `logs/`, `dist/` are gitignored
  (runtime / local data dirs).

## Reporting issues

Open an issue with:

- A minimal config (the smallest `validation_run.*.json` that
  reproduces).
- The exact `qa4sm-validate` command (or `run_validation.ps1`
  flags) and the full log file (`logs/validation_run_*.log`).
- The expected vs actual result (counts of `ok_points` /
  `error_points`, exit code).
- Output of `uv run --frozen --with pytesmo --with dask
  --with cupy-cuda13x python -c "import sys; print(sys.version)"`
  for environment context.

## Pull request flow

1. Branch from `main` (or `feat/gpu-pytesmo-integrate` for changes
   in that lineage — coordinate in the issue first).
2. Push to your fork and open a PR against `main`.
3. CI runs ruff + pytest. PRs are merged once CI is green and at
   least one reviewer approves.
4. Squash-merge is the default; the PR title becomes the commit
   summary.

## License

By submitting a contribution, you agree your work will be released
under the MIT license at the root of this repository. The vendored
`pytesmo/` submodule keeps its own BSD-3-Clause terms; do not copy
upstream pytesmo code into the parent without honouring the
upstream license.