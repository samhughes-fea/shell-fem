# Releasing

The rules are the shared standard's
[releasing](docs/conventions/standard/releasing.md) and
[versioning](docs/conventions/standard/versioning.md): a release is a pull request
titled `chore(release): prepare X.Y.Z`, and the synced `release.yml` tags and
releases its squash commit when it merges. Never push or move a tag by hand.

This repository's specifics:

- `__version__` lives in `src/shell_fem/__init__.py`; `pyproject.toml` reads it
  from there.
- The release PR moves the `## [Unreleased]` notes in `CHANGELOG.md` under
  `## [X.Y.Z] - YYYY-MM-DD`.
- shell-fem depends on implicit-geometry-kernel and laminate-constitutive-model by
  tag, so release those first and bump the pins in `tests.yml` (and the minimum versions in `pyproject.toml`).
- PyPI: `pypi: false` in `.github/repo.yml`. To publish, add a PyPI trusted
  publisher for `shell-fem` (workflow `release.yml`, environment `pypi`), create
  the `pypi` environment in the repository settings, and set `pypi: true`.
