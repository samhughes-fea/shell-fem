"""Production per-station dispatch + the single shell-theory capability seam.

Covers ``engine/station_theory_dispatch.py`` (theory routing + error path) and the
capability table in ``core/types.py`` that every dispatch seam now consults.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from blade_precompute.section_shell_model.pre_processing.containers import (
    types as core_types,
)
from blade_precompute.section_shell_model.pre_processing.containers.types import (
    SHELL_THEORY_CAPABILITIES,
    ShellTheory,
    require_shell_theory_supports,
    resolve_shell_theory_capability,
    shell_theories_supporting,
)
from blade_precompute.section_shell_model.engine import station_theory_dispatch as std
from blade_precompute.section_shell_model.engine.station_theory_dispatch import (
    StationTheorySolveRequest,
    dispatch_station_theory,
)


def _request(theory: ShellTheory) -> StationTheorySolveRequest:
    return StationTheorySolveRequest(
        theory=theory,
        section_view=None,
        shell_inputs=None,
        airfoil=np.zeros((0, 2), dtype=float),
        spars_m=[],
        station_dir=Path("."),
        station_tag="station-0",
        N=0.0,
        Vy=0.0,
        Vz=0.0,
        My=0.0,
        Mz=0.0,
        T=0.0,
        n_elements_per_panel=None,
        target_element_length_m=1.0,
        min_elements_per_panel=2,
    )


# --- capability table (single dispatch seam) -------------------------------


def test_capability_table_covers_all_theories() -> None:
    assert set(SHELL_THEORY_CAPABILITIES) == set(ShellTheory)


def test_only_fsdt_has_native_strip_kernel() -> None:
    assert shell_theories_supporting("native_strip_kernel") == (ShellTheory.FSDT,)


def test_all_theories_support_periodic_and_station_dispatch() -> None:
    assert set(shell_theories_supporting("periodic")) == set(ShellTheory)
    assert set(shell_theories_supporting("station_dispatch")) == set(ShellTheory)


def test_only_ccx_requires_external_work_dir() -> None:
    reqs = {t for t, c in SHELL_THEORY_CAPABILITIES.items() if c.requires_external_work_dir}
    assert reqs == {ShellTheory.CALCULIX_CCX}


def test_require_shell_theory_supports_raises_for_missing_feature() -> None:
    with pytest.raises(ValueError, match="does not support 'native_strip_kernel'"):
        require_shell_theory_supports(ShellTheory.CALCULIX_CCX, feature="native_strip_kernel")


def test_require_shell_theory_supports_returns_capability() -> None:
    cap = require_shell_theory_supports(ShellTheory.FSDT, feature="native_strip_kernel")
    assert cap is resolve_shell_theory_capability(ShellTheory.FSDT)


# --- station dispatch routing -------------------------------------------


def test_dispatch_routes_fsdt_to_periodic_hashin(monkeypatch) -> None:
    monkeypatch.setattr(std, "fsdt_periodic_station_hashin", lambda *a, **k: {"solver": "fsdt"})
    res = dispatch_station_theory(_request(ShellTheory.FSDT))
    assert res.fsdt_periodic_extra == {"solver": "fsdt"}
    assert res.backend_extra is None


def test_dispatch_routes_ccx_to_thin_wall(monkeypatch) -> None:
    """CCX is quarantined (Phase 6): dispatch only works after opt-in registration."""
    from blade_precompute.section_shell_model.calculix_ccx.register import (
        register_calculix_ccx_backend,
    )
    from blade_precompute.section_shell_model.calculix_ccx import station_dispatch_adapter

    register_calculix_ccx_backend()
    monkeypatch.setattr(station_dispatch_adapter, "ccx_station_solve_from_thin_wall", lambda req: {"solver": "ccx"})
    res = dispatch_station_theory(_request(ShellTheory.CALCULIX_CCX))
    assert res.backend_extra == {"solver": "ccx"}
    assert res.fsdt_periodic_extra is None


def test_dispatch_rejects_theory_without_station_dispatch_support(monkeypatch) -> None:
    # Monkeypatch FSDT capability rather than CCX: avoids needing CCX registration
    # just to exercise the supports_station_dispatch gate.
    cap = SHELL_THEORY_CAPABILITIES[ShellTheory.FSDT]
    monkeypatch.setitem(
        core_types.SHELL_THEORY_CAPABILITIES,
        ShellTheory.FSDT,
        replace(cap, supports_station_dispatch=False),
    )
    with pytest.raises(ValueError, match="does not support 'station_dispatch'"):
        dispatch_station_theory(_request(ShellTheory.FSDT))
