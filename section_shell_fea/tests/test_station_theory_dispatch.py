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

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers import (
    types as core_types,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import (
    SHELL_THEORY_CAPABILITIES,
    ShellTheory,
    require_shell_theory_supports,
    resolve_shell_theory_capability,
    shell_theories_supporting,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.dispatch import station_theory_dispatch as std
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.dispatch.station_theory_dispatch import (
    StationTheorySolveRequest,
    dispatch_station_theory,
)


def _request(
    theory: ShellTheory,
    *,
    with_prebuilt: bool = False,
) -> StationTheorySolveRequest:
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
        section_strip_mesh=object() if with_prebuilt else None,
        panel_topology=object() if with_prebuilt else None,
        section_force_vector=object() if with_prebuilt else None,
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
    res = dispatch_station_theory(_request(ShellTheory.FSDT, with_prebuilt=True))
    assert res.fsdt_periodic_extra == {"solver": "fsdt"}
    assert res.backend_extra is None


def test_dispatch_fsdt_requires_prebuilt_products() -> None:
    with pytest.raises(ValueError, match="prebuilt"):
        dispatch_station_theory(_request(ShellTheory.FSDT, with_prebuilt=False))


def test_dispatch_fsdt_demo_convenience_warns_and_builds(monkeypatch) -> None:
    monkeypatch.setattr(
        "blade_structure_utilities.mesh_utilities.section_mesh.strip.strip_mesh.build_section_strip_mesh",
        lambda *a, **k: object(),
    )
    monkeypatch.setattr(
        "blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.api."
        "shell_panel_topology.shell_panel_topology_from_shell_inputs",
        lambda *a, **k: object(),
    )
    monkeypatch.setattr(
        "blade_structure_utilities.load_utilities.section_loads.shape_section_force_vector",
        lambda *a, **k: object(),
    )
    monkeypatch.setattr(std, "fsdt_periodic_station_hashin", lambda *a, **k: {"solver": "fsdt"})
    req = _request(ShellTheory.FSDT, with_prebuilt=False)
    # Convenience path needs shell_inputs for SFV shaping when load_field is None.
    from dataclasses import replace

    req = replace(req, shell_inputs=object())
    with pytest.warns(UserWarning, match="dispatch_fsdt_demo_convenience"):
        res = std.dispatch_fsdt_demo_convenience(req)
    assert res.fsdt_periodic_extra == {"solver": "fsdt"}


def test_fsdt_traction_requires_sfv_no_overlay(monkeypatch) -> None:
    """Usable load-field traction selects Channel B; SFV required; no overlay kwargs."""
    from blade_structure_utilities.load_utilities.section_loads.surface_traction.station_load import (
        DistributedStationLoadSpec,
        IncidentAeroSpec,
        LoadFieldProvenance,
        SectionStationLoadField,
    )
    from blade_structure_utilities.load_utilities.section_loads import TractionField

    captured: dict[str, Any] = {}

    class _FakeResults:
        quaternary_results = type(
            "Q",
            (),
            {
                "governing": {"fi": 0.1},
                "needs_review": None,
                "hashin_fi": None,
            },
        )()
        tertiary_results = None

    class _FakeAssembly:
        panel_maps = [object()]

    class _FakeRunner:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.load_channel_used = str(kwargs.get("load_channel", "thinwall"))
            self.assembly = _FakeAssembly()

        def execute(self):
            return self

        def get_results(self):
            return _FakeResults()

    import blade_structure_utilities.fem_utilities.section_shell_fea.simulation_runner.section_linear_static_runner as slsr
    import blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.api.shell_panel_topology as spt

    monkeypatch.setattr(slsr, "SectionLinearStaticRunner", _FakeRunner)
    monkeypatch.setattr(spt, "shell_panel_topology_from_shell_inputs", lambda *_a, **_k: object())

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

    with pytest.raises(ValueError, match="section_force_vector is required"):
        std.fsdt_periodic_station_hashin(
            None,
            None,
            N=0.0,
            Vy=0.0,
            Vz=0.0,
            My=0.0,
            Mz=0.0,
            T=1.0,
            section_strip_mesh=object(),
            panel_topology=object(),
            load_field=lf,
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
        section_strip_mesh=object(),
        panel_topology=object(),
        load_field=lf,
        section_force_vector=object(),
    )
    assert out["solver"] == "fsdt_linear_static_cascade"
    assert out["load_channel"] == "surface_traction"
    assert out["used_traction"] is False
    assert "traction_field" not in captured
    assert "traction_points_yz" not in captured
    assert captured.get("load_channel") == "surface_traction"
    assert captured.get("section_force_vector") is not None


def test_dispatch_routes_ccx_to_thin_wall(monkeypatch) -> None:
    """CCX station dispatch is not registered; only the global validation backend is."""
    from blade_structure_utilities.fem_utilities.section_shell_fea.calculix_ccx.register import (
        register_calculix_ccx_backend,
    )

    register_calculix_ccx_backend()
    with pytest.raises(ValueError, match="No station-dispatch handler"):
        dispatch_station_theory(_request(ShellTheory.CALCULIX_CCX))


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
