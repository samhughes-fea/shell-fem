"""Checkout-depth-independent Cotide data discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.grid_independence_study import (
    discover_cotide_checkout_root,
)


def test_discover_cotide_checkout_root_is_independent_of_depth(tmp_path: Path):
    (tmp_path / "cotide_data_interfaces").mkdir()
    nested = tmp_path / "a" / "b" / "c" / "module.py"
    nested.parent.mkdir(parents=True)
    nested.write_text("# placeholder\n", encoding="utf-8")
    assert discover_cotide_checkout_root(nested) == tmp_path
    assert discover_cotide_checkout_root(nested.parent) == tmp_path


def test_discover_cotide_checkout_root_errors_when_data_absent(tmp_path: Path):
    start = tmp_path / "lonely" / "module.py"
    start.parent.mkdir()
    start.write_text("# placeholder\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="cotide_data_interfaces"):
        discover_cotide_checkout_root(start)


def test_discover_cotide_checkout_root_from_this_checkout():
    root = discover_cotide_checkout_root()
    assert (root / "cotide_data_interfaces" / "new_3.2m_rotor").is_dir()
    src = Path(discover_cotide_checkout_root.__code__.co_filename).read_text(encoding="utf-8")
    assert "parents[" not in src
    assert "discover_cotide_checkout_root" in src
