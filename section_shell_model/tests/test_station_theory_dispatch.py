"""Production per-station dispatch + the single shell-theory capability seam.

Covers ``pre_processing/dispatch/station_theory_dispatch.py`` (theory routing + error path)
and the capability table that every dispatch seam now consults.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from blade_structure_utilities.section_shell_model.pre_processing.containers import (
    types as core_types,
)
from blade_structure_utilities.section_shell_model.pre_processing.containers.types import (
    SHELL_THEORY_CAPABILITIES,
    ShellTheory,
    require_shell_theory_supports,
    resolve_shell_theory_capability,
    shell_theories_supporting,
)
from blade_structure_utilities.section_shell_model.pre_processing.dispatch import station_theory_dispatch as std
from blade_structure_utilities.section_shell_model.pre_processing.dispatch.station_theory_dispatch import (
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


def test_fsdt_traction_routes_to_linear_static_not_periodic(monkeypatch) -> None:
    """Usable load-field traction must use linear-static, never fi_track=periodic."""
    from blade_structure_utilities.section_shell_model.pre_processing.loading.containers.types import (
        DistributedStationLoadSpec,
        IncidentAeroSpec,
        LoadFieldProvenance,
        SectionStationLoadField,
    )
    from blade_structure_utilities.point_to_surftraction.tractions import TractionField

    captured: dict[str, Any] = {}

    def _fake_hashin(pb, **kwargs):
        captured.update(kwargs)
        return {
            "max_fi_hashin": 0.1,
            "needs_review": False,
            "fi_by_panel": [0.1],
            "governing": {"fi": 0.1},
            "used_traction": True,
            "track": "linear_static",
        }

    monkeypatch.setattr(
        "blade_structure_utilities.section_shell_model.simulation_runner.api.station_recovery.run_station_hashin",
        _fake_hashin,
    )
    monkeypatch.setattr(
        "blade_structure_utilities.section_shell_model.pre_processing.mesh.api.pipeline_b.pipeline_b_section_from_shell_inputs",
        lambda *_a, **_k: object(),
    )

    n_panels = 4
    outline = np.column_stack(
        [np.linspace(0.0, 1.0, n_panels + 1), np.zeros(n_panels + 1)]
    )
    forces = np.zeros((n_panels, 2), dtype=float)
    forces[:, 0] = 10.0
    lf = SectionStationLoadField(
        station_id="s0",
        z_station=0.5,
        chord_m=1.0,
        twist_rad=0.0,
        outline_yz=outline,
        traction_field=TractionField(
            pressure_forces=forces.copy(),
            shear_forces=np.zeros_like(forces),
            total_forces=forces,
        ),
        distributed_source=DistributedStationLoadSpec(
            z_station=0.5, q_y_Npm=40.0, q_z_Npm=0.0, m_x_Nmpm=1.0,
        ),
        incident=IncidentAeroSpec(
            u_rel_m_s=10.0,
            reynolds_number=1e6,
            mach_number=0.1,
            alpha_deg=0.0,
            omega_rad_s=0.0,
            u_tan_m_s=0.0,
        ),
        provenance=LoadFieldProvenance(
            derivation="test",
            xfoil_used=False,
            synthetic_aero=True,
            direct_scalar_override=True,
        ),
    )

    out = std.fsdt_periodic_station_hashin(
        None,
        None,
        N=0.0,
        Vy=0.0,
        Vz=0.0,
        My=0.0,
        Mz=0.0,
        T=1.0,
        n_elements_per_panel=4,
        target_element_length_m=None,
        min_elements_per_panel=2,
        load_field=lf,
    )
    assert out["solver"] == "fsdt_linear_static_cascade"
    assert out["used_traction"] is True
    assert captured.get("fi_track") == "linear_static"
    assert captured.get("traction_field") is not None
    assert captured.get("traction_points_yz") is not None


def test_dispatch_routes_ccx_to_thin_wall(monkeypatch) -> None:
    """CCX is quarantined (Phase 6): dispatch only works after opt-in registration."""
    from blade_structure_utilities.section_shell_model.calculix_ccx.register import (
        register_calculix_ccx_backend,
    )
    from blade_structure_utilities.section_shell_model.calculix_ccx import station_dispatch_adapter

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
