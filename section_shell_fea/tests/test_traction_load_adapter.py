"""Tests for Channel A / B load-provenance in ``traction_load_adapter``."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from blade_structure_utilities.geometry_utilities.section_contract.beam_resultant_vector import (
    BEAM_LOAD_AXIS_YZ,
    FRAME_SECTION_YZ,
    BeamResultantVector,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.interface.traction_load_adapter import (
    load_field_for_dispatch,
    resolve_shell_station_resultants,
    section_resultants_from_load_field,
    traction_points_from_load_field,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.simulation_runner.api.linear_static import (
    LinearStaticCascadeResult,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.simulation_runner.section_linear_static_runner import (
    SectionLinearStaticRunner,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.loading.containers.types import (
    DistributedStationLoadSpec,
    IncidentAeroSpec,
    LoadFieldProvenance,
    SectionStationLoadField,
)
from blade_structure_utilities.load_utilities.section_loads import TractionField


def _incident() -> IncidentAeroSpec:
    return IncidentAeroSpec(
        u_rel_m_s=10.0,
        reynolds_number=1.0e6,
        mach_number=0.1,
        alpha_deg=5.0,
        omega_rad_s=0.0,
        u_tan_m_s=0.0,
    )


def _load_field_with_traction(
    *,
    q_y: float = 1000.0,
    q_z: float = 50.0,
    m_x: float = 10.0,
    n_panels: int = 4,
) -> SectionStationLoadField:
    """Synthetic load field whose traction panel count matches outline midpoints."""
    outline = np.column_stack(
        [
            np.linspace(0.0, 1.0, n_panels + 1),
            np.zeros(n_panels + 1),
        ]
    )
    forces = np.zeros((n_panels, 2), dtype=float)
    forces[:, 0] = q_y / n_panels
    forces[:, 1] = q_z / n_panels
    tf = TractionField(
        pressure_forces=forces.copy(),
        shear_forces=np.zeros_like(forces),
        total_forces=forces,
    )
    return SectionStationLoadField(
        station_id="s0",
        z_station=0.5,
        chord_m=1.0,
        twist_rad=0.0,
        outline_yz=outline,
        traction_field=tf,
        distributed_source=DistributedStationLoadSpec(
            z_station=0.5,
            q_y_Npm=q_y,
            q_z_Npm=q_z,
            m_x_Nmpm=m_x,
        ),
        incident=_incident(),
        provenance=LoadFieldProvenance(
            derivation="test",
            xfoil_used=False,
            synthetic_aero=True,
            direct_scalar_override=True,
        ),
    )


def _load_field_without_usable_traction(
    *,
    q_y: float = 1000.0,
    q_z: float = 50.0,
    m_x: float = 10.0,
) -> SectionStationLoadField:
    """Traction row count mismatches outline panels → no Channel B overlay."""
    outline = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)  # 1 panel
    forces = np.zeros((3, 2), dtype=float)  # mismatch
    tf = TractionField(
        pressure_forces=forces.copy(),
        shear_forces=np.zeros_like(forces),
        total_forces=forces,
    )
    return SectionStationLoadField(
        station_id="s0",
        z_station=0.5,
        chord_m=1.0,
        twist_rad=0.0,
        outline_yz=outline,
        traction_field=tf,
        distributed_source=DistributedStationLoadSpec(
            z_station=0.5,
            q_y_Npm=q_y,
            q_z_Npm=q_z,
            m_x_Nmpm=m_x,
        ),
        incident=_incident(),
        provenance=LoadFieldProvenance(
            derivation="test",
            xfoil_used=False,
            synthetic_aero=True,
            direct_scalar_override=True,
        ),
    )


def test_traction_points_align_with_total_forces() -> None:
    lf = _load_field_with_traction(n_panels=4)
    pts = traction_points_from_load_field(lf)
    assert pts is not None
    assert pts.shape == (4, 2)


def test_section_resultants_zero_shears_when_traction_usable() -> None:
    """Double-count invariant: same q cannot enter Channel A shears when B is active."""
    lf = _load_field_with_traction(q_y=1234.0, q_z=56.0, m_x=7.0)
    N, Vy, Vz, My, Mz, T = section_resultants_from_load_field(lf)
    assert N == 0.0 and My == 0.0 and Mz == 0.0
    assert Vy == 0.0 and Vz == 0.0
    assert T == 7.0


def test_section_resultants_use_distributed_shears_without_usable_traction() -> None:
    lf = _load_field_without_usable_traction(q_y=100.0, q_z=20.0, m_x=3.0)
    assert traction_points_from_load_field(lf) is None
    N, Vy, Vz, My, Mz, T = section_resultants_from_load_field(lf)
    assert (N, Vy, Vz, My, Mz, T) == (0.0, 100.0, 20.0, 0.0, 0.0, 3.0)


def test_resolve_prefers_explicit_station_resultants() -> None:
    lf = _load_field_with_traction()
    explicit = {0: (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = resolve_shell_station_resultants(
            0,
            station_resultants=explicit,
            station_load_fields={0: lf},
        )
    assert out == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_resolve_falls_back_to_load_field_when_no_resultants() -> None:
    lf = _load_field_with_traction(m_x=9.0)
    out = resolve_shell_station_resultants(
        0,
        station_resultants=None,
        station_load_fields={0: lf},
    )
    assert out == (0.0, 0.0, 0.0, 0.0, 0.0, 9.0)


def test_resolve_missing_load_data_raises() -> None:
    with pytest.raises(ValueError, match="no load data"):
        resolve_shell_station_resultants(
            3,
            station_resultants=None,
            station_load_fields=None,
        )


def _load_field_aligned_to_skin_mesh(*, m_x: float = 12.0) -> tuple[Any, SectionStationLoadField]:
    """Build a shell panel topology section + load field whose traction points match skin pairs."""
    from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.loading.traction_face_load import (
        _skin_node_pairs,
    )
    from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.api.shell_panel_topology import (
        build_shell_panel_topology,
    )
    from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.global_shell_assembly import (
        assemble_global_shell,
    )
    from blade_structure_utilities.mesh_utilities.section_mesh import build_section_strip_mesh

    panel_topology = build_shell_panel_topology("2412", [0.35], chord_m=1.0)
    zones = list(panel_topology.zones)
    zeros = [np.zeros(len(np.asarray(p.s, dtype=float))) for p in zones]
    mesh = build_section_strip_mesh(panel_topology.shell_inputs, n_elements_per_panel=6)
    _K, _f, node_meta, _pm, *_rest = assemble_global_shell(
        zones, zeros, zeros, section_strip_mesh=mesh,
    )
    locs, _tangents, _ = _skin_node_pairs(node_meta)
    n = int(locs.shape[0])
    forces = np.zeros((n, 2), dtype=float)
    forces[:, 0] = 100.0 / max(n, 1)
    outline = np.zeros((n + 1, 2), dtype=float)
    outline[0] = locs[0]
    for i in range(n):
        outline[i + 1] = 2.0 * locs[i] - outline[i]
    tf = TractionField(
        pressure_forces=forces.copy(),
        shear_forces=np.zeros_like(forces),
        total_forces=forces,
    )
    lf = SectionStationLoadField(
        station_id="s0",
        z_station=0.0,
        chord_m=1.0,
        twist_rad=0.0,
        outline_yz=outline,
        traction_field=tf,
        distributed_source=DistributedStationLoadSpec(
            z_station=0.0,
            q_y_Npm=500.0,
            q_z_Npm=40.0,
            m_x_Nmpm=m_x,
        ),
        incident=_incident(),
        provenance=LoadFieldProvenance(
            derivation="test",
            xfoil_used=False,
            synthetic_aero=True,
            direct_scalar_override=True,
        ),
    )
    return panel_topology, lf


def test_run_section_linear_static_from_load_field_exclusive_and_fi() -> None:
    """Production load-field path: zero shears + quaternary FI on linear cascade."""
    from blade_structure_utilities.mesh_utilities.section_mesh.engine.strip_mesh import (
        build_section_strip_mesh,
    )

    panel_topology, lf = _load_field_aligned_to_skin_mesh(m_x=12.0)
    N, Vy, Vz, My, Mz, T = section_resultants_from_load_field(lf)
    assert Vy == 0.0 and Vz == 0.0 and T == 12.0

    traction_points = traction_points_from_load_field(lf)
    traction_field = lf.traction_field if traction_points is not None else None
    resultants = BeamResultantVector.from_named(
        N=N, Vy=Vy, Vz=Vz, My=My, Mz=Mz, T=T,
        frame=FRAME_SECTION_YZ,
        about_yz=BEAM_LOAD_AXIS_YZ,
    )
    cascade = LinearStaticCascadeResult.from_executed_runner(
        SectionLinearStaticRunner(
            panel_topology=panel_topology,
            resultants=resultants,
            traction_field=traction_field,
            traction_points_yz=traction_points,
            section_strip_mesh=build_section_strip_mesh(
                panel_topology.shell_inputs,
                n_elements_per_panel=6,
                min_elements_per_panel=2,
            ),
            load_channel="equipollent",
        ).execute()
    )
    assert cascade.used_traction is True
    assert cascade.quaternary is not None
    fi = float((cascade.quaternary.governing or {}).get("fi", 0.0) or 0.0)
    assert np.isfinite(fi)


def test_load_field_for_dispatch_suppresses_when_explicit_resultants_win() -> None:
    lf = _load_field_with_traction()
    explicit = {0: (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)}
    assert load_field_for_dispatch(
        0,
        station_resultants=explicit,
        station_load_fields={0: lf},
    ) is None
    assert load_field_for_dispatch(
        0,
        station_resultants=None,
        station_load_fields={0: lf},
    ) is lf
