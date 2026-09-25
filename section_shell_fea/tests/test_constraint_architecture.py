"""WP0 numerical baselines for constraints, assembly, and solve sequence."""

from __future__ import annotations

import numpy as np
import pytest

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.constraints.compose import (
    build_section_constraint_set,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.constraints.types import (
    constraint_equations_to_solver_dict,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.assemble_global import (
    AssembleGlobalShellSystem,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.simulation_runner.section_linear_static_runner import (
    SectionLinearStaticRunner,
)
from blade_structure_utilities.geometry_utilities.section_contract.beam_resultant_vector import (
    BEAM_LOAD_AXIS_YZ,
    FRAME_SECTION_YZ,
    BeamResultantVector,
)
from blade_structure_utilities.load_utilities.section_loads import (
    shape_section_force_vector,
)
from blade_structure_utilities.mesh_utilities.section_mesh import (
    build_section_strip_mesh,
)
from blade_structure_utilities.mesh_utilities.section_mesh.handoff._section_to_shell_inputs import (
    shell_mesh_inputs_from_section,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.api.shell_panel_topology import (
    shell_panel_topology_from_shell_inputs,
)


pytestmark = pytest.mark.unit


def _5df_topology_and_mesh():
    from blade_structure_utilities.geometry_utilities.section_geometry.api.factory import (
        build_section_view,
        resolve_system_type,
    )
    from blade_structure_utilities.geometry_utilities.section_geometry.sdf.airfoil import (
        AirfoilSDF,
    )

    twist = float(np.deg2rad(34.1))
    airfoil = AirfoilSDF.from_naca_series(
        6, 63.0, 4.0, 15.0, n_points=200, chord=0.241, closed_te=True
    )
    section = build_section_view(
        airfoil,
        resolve_system_type("5D-F"),
        twist_angle_rad=twist,
        skin_thickness=0.001,
        web_thickness=0.0015,
        cap_height=0.004,
    )
    shell = shell_mesh_inputs_from_section(
        section, twist_rad=twist, layout_key="5D-F", n_web_samples=20, n_cap_samples=80
    )
    mesh = build_section_strip_mesh(shell, target_element_length_frac=0.024, endpoint_tol=1e-6)
    topology = shell_panel_topology_from_shell_inputs(shell, allow_demo_bind=True)
    return topology, mesh, shell


def test_production_execute_scatters_once(monkeypatch):
    calls = {"n": 0}
    original = AssembleGlobalShellSystem.run

    def counted(self, cache, n_gdof):
        calls["n"] += 1
        return original(self, cache, n_gdof)

    monkeypatch.setattr(AssembleGlobalShellSystem, "run", counted)
    topology, mesh, shell = _5df_topology_and_mesh()
    resultants = BeamResultantVector.from_named(
        T=5.0e3, Vy=2.0e4, Mz=1.0e4,
        frame=FRAME_SECTION_YZ, about_xy=BEAM_LOAD_AXIS_YZ,
    )
    sfv = shape_section_force_vector(resultants, shell.midlines, cell_wall_loops=shell.cell_wall_loops)
    runner = SectionLinearStaticRunner(
        panel_topology=topology,
        section_strip_mesh=mesh,
        resultants=resultants,
        section_force_vector=sfv,
        load_channel="thinwall",
        cap_tie_mode="skin_offset",
        interface_constraint_mode="shared",
    ).execute()
    assert calls["n"] == 0, "production path must reuse assembly K, not scatter-add again"
    assert runner.assembly is not None
    assert runner.assembly.n_scatter_add == 1
    assert runner.constraint_set is not None
    assert runner.f_element is not None
    assert runner.f_station is not None
    np.testing.assert_allclose(
        np.asarray(runner.global_sys.f),
        np.asarray(runner.f_element) + np.asarray(runner.f_station),
        rtol=0.0,
        atol=1e-12,
    )
    results = runner.get_results()
    assert results.quaternary_results is not None
    fi = float((results.quaternary_results.governing or {}).get("fi", 0.0) or 0.0)
    assert np.isfinite(fi)
    meta = results.diagnostics.constraint_metadata
    assert meta["interface_constraint_mode"] == "shared"
    assert meta["cap_tie_mode"] == "skin_offset"
    assert int(meta["n_cap_links"]) > 0
    assert "equations" not in meta


def test_shared_none_and_tb_skin_offset_compose(assembled_5df):
    shared_none = build_section_constraint_set(
        assembled_5df, interface_constraint_mode="shared", cap_tie_mode="none"
    )
    shared_tie = build_section_constraint_set(
        assembled_5df, interface_constraint_mode="shared", cap_tie_mode="skin_offset"
    )
    assert len(shared_tie.equations) > len(shared_none.equations)
    d_none = constraint_equations_to_solver_dict(shared_none.equations)
    d_tie = constraint_equations_to_solver_dict(shared_tie.equations)
    assert set(d_none).issubset(set(d_tie))
