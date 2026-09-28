"""WP9: alias expiry ledger — retain adapter homes; gate hard removals."""

from __future__ import annotations

from datetime import date
from pathlib import Path

_FE_ROOT = Path(__file__).resolve().parents[1]
_EXPIRY = date(2026, 11, 30)
_LEDGER = _FE_ROOT / "docs" / "ALIAS_EXPIRY_LEDGER.md"

_ADAPTER_HOMES = (
    "pre_processing/mesh",
    "pre_processing/loading",
    "pre_processing/reference_thin_wall",
)

_EXPIRING_SHIMS = (
    "pre_processing/topology/__init__.py",
    "pre_processing/loading/thin_wall/__init__.py",
)


def test_alias_expiry_ledger_exists_and_classifies_homes() -> None:
    text = _LEDGER.read_text(encoding="utf-8")
    assert "internal adapter home" in text
    assert "expiring alias" in text
    assert "2026-11-30" in text
    for home in ("mesh/", "loading/", "reference_thin_wall/"):
        assert home in text


def test_three_internal_adapter_homes_remain() -> None:
    for rel in _ADAPTER_HOMES:
        root = _FE_ROOT / rel
        assert root.is_dir(), f"missing adapter home: {rel}"
        assert (root / "README.md").is_file(), f"missing README for {rel}"
        assert (root / "__init__.py").is_file()


def test_mesh_naca_profile_retained() -> None:
    assert (_FE_ROOT / "pre_processing" / "mesh" / "naca_profile.py").is_file()


def test_expiring_shims_present_before_expiry() -> None:
    """Hard deletion is date-gated; before expiry the warning aliases must remain."""
    if date.today() < _EXPIRY:
        for rel in _EXPIRING_SHIMS:
            path = _FE_ROOT / rel
            assert path.is_file(), f"pre-expiry shim missing: {rel}"
            text = path.read_text(encoding="utf-8")
            assert "DeprecationWarning" in text
    else:
        for rel in _EXPIRING_SHIMS:
            path = _FE_ROOT / rel
            assert not path.exists(), (
                f"post-expiry shim still present (run WP9 retirement): {rel}"
            )


def test_bsu_owners_remain_importable() -> None:
    from blade_structure_utilities.mesh_utilities.section_mesh import SectionStripMesh
    from blade_structure_utilities.load_utilities.section_loads import SectionForceVector
    from blade_structure_utilities.geometry_utilities.section_thin_wall_model import (
        omega_increment,
        sectorial_warping,
    )

    assert SectionStripMesh is not None
    assert SectionForceVector is not None
    assert callable(omega_increment)
    assert callable(sectorial_warping)
