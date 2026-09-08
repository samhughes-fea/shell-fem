"""Quarantine contract for FE demo-bind named leak."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest


def test_shell_panel_topology_requires_zones_or_explicit_demo_bind():
    from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.api.shell_panel_topology import (
        shell_panel_topology_from_shell_inputs,
    )

    shell = SimpleNamespace()  # unused if we raise before bind
    with pytest.raises(ValueError, match="allow_demo_bind=True"):
        shell_panel_topology_from_shell_inputs(shell)

    # Explicit demo path still warns (named leak), then may fail later on bad shell;
    # the contract under test is that allow_demo_bind=False is rejected first.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            shell_panel_topology_from_shell_inputs(shell, allow_demo_bind=True)
        except Exception:
            pass
        assert any("demo-bind" in str(w.message) for w in caught)
