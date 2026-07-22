"""Package-layout contract: full section_shell_model tree, not a lean spine."""

from __future__ import annotations

from pathlib import Path

import pytest

_PKG_ROOT = Path(__file__).resolve().parents[1]

_REQUIRED_TOP_LEVEL_DIRS = (
    "pre_processing",
    "processing",
    "simulation_runner",
    "calculix_ccx",
    "examples",
    "benchmarks",
    "interface",
    "io",
    "validation",
    "tests",
)


@pytest.mark.parametrize("dirname", _REQUIRED_TOP_LEVEL_DIRS)
def test_full_package_top_level_dirs_exist(dirname: str) -> None:
    assert (_PKG_ROOT / dirname).is_dir(), f"missing required package dir: {dirname}"


def test_production_entrypoints_importable() -> None:
    from blade_structure_utilities.section_shell_model import (
        run_section_linear_static_from_resultants,
        run_station_hashin,
    )

    assert callable(run_station_hashin)
    assert callable(run_section_linear_static_from_resultants)


def test_no_legacy_foundation_import_paths() -> None:
    """Code under the package must not import old fastblade foundation paths."""
    banned = (
        "blade_precompute.section_shell_model",
        "blade_utilities.mesh",
        "blade_utilities.sdf",
        "blade_utilities.point_to_surftraction",
    )
    self_path = Path(__file__).resolve()
    offenders: list[str] = []
    for path in _PKG_ROOT.rglob("*.py"):
        if path.resolve() == self_path:
            continue
        text = path.read_text(encoding="utf-8")
        for token in banned:
            if token in text:
                offenders.append(f"{path.relative_to(_PKG_ROOT)}: {token}")
    assert not offenders, "legacy import paths remain:\n" + "\n".join(offenders)
