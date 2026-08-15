"""Station linear-static facade: execute → get_results → quaternary FI."""

from __future__ import annotations

import numpy as np
import pytest

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.api.shell_panel_topology import (
    build_shell_panel_topology,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.simulation_runner.api.station_recovery import (
    build_fsdt_station_analysis,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.simulation_runner.section_linear_static_runner import (
    SectionLinearStaticRunner,
)
from blade_structure_utilities.mesh_utilities.section_mesh.strip.strip_mesh import (
    build_section_strip_mesh,
)


def _pb():
    return build_shell_panel_topology("2412", [0.35], chord_m=1.0)


def _strip_mesh(panel_topology, *, n_elements_per_panel: int = 4):
    return build_section_strip_mesh(
        panel_topology.shell_inputs,
        n_elements_per_panel=n_elements_per_panel,
    )


def test_section_linear_static_runner_execute_get_results() -> None:
    from blade_structure_utilities.geometry_utilities.section_contract.beam_resultant_vector import (
        BEAM_LOAD_AXIS_YZ,
        FRAME_SECTION_YZ,
        BeamResultantVector,
    )
    from blade_structure_utilities.load_utilities.section_loads import (
        shape_section_force_vector_from_panel_topology,
    )

    panel_topology = _pb()
    resultants = BeamResultantVector.from_named(
        N=1.0e5,
        My=3.0e4,
        Vz=2.0e4,
        frame=FRAME_SECTION_YZ,
        about_xy=BEAM_LOAD_AXIS_YZ,
    )
    sfv = shape_section_force_vector_from_panel_topology(resultants, panel_topology)
    static = SectionLinearStaticRunner(
        panel_topology=panel_topology,
        resultants=resultants,
        section_strip_mesh=_strip_mesh(panel_topology, n_elements_per_panel=4),
        load_channel="thinwall",
        section_force_vector=sfv,
    ).execute()
    results = static.get_results()
    quaternary = results.quaternary_results
    assert quaternary is not None
    assert quaternary.hashin_fi is not None
    fi_field = quaternary.hashin_fi.in_plane
    assert fi_field is not None
    assert fi_field.ndim >= 1
    gov = quaternary.governing or {}
    max_fi = float(gov.get("fi", 0.0) or 0.0)
    assert np.isfinite(max_fi) and max_fi >= 0.0
    assert static.assembly is not None
    # Idempotent execute
    assert static.execute() is static


def test_get_results_requires_execute() -> None:
    with pytest.raises(RuntimeError, match="execute"):
        SectionLinearStaticRunner(panel_topology=_pb(), My=1.0e3).get_results()


def test_execute_requires_panel_topology() -> None:
    with pytest.raises(RuntimeError, match="panel_topology"):
        SectionLinearStaticRunner(My=1.0e3).execute()


def test_execute_requires_section_strip_mesh() -> None:
    with pytest.raises(ValueError, match="section_strip_mesh"):
        SectionLinearStaticRunner(panel_topology=_pb(), My=1.0e3).execute()


def test_build_fsdt_station_analysis_unknown_fi_track_raises() -> None:
    panel_topology = _pb()
    mesh = _strip_mesh(panel_topology, n_elements_per_panel=4)
    with pytest.raises(ValueError, match="unknown fi_track|removed"):
        build_fsdt_station_analysis(
            panel_topology.section,
            panel_topology.shell_inputs,
            z_station=0.35,
            N=1.0e5,
            My=3.0e4,
            section_strip_mesh=mesh,
            fi_track="not_a_track",  # type: ignore[arg-type]
        )


def test_build_fsdt_station_analysis_periodic_track_removed() -> None:
    panel_topology = _pb()
    mesh = _strip_mesh(panel_topology, n_elements_per_panel=4)
    with pytest.raises(ValueError, match="removed"):
        build_fsdt_station_analysis(
            panel_topology.section,
            panel_topology.shell_inputs,
            z_station=0.35,
            N=1.0e5,
            My=3.0e4,
            section_strip_mesh=mesh,
            fi_track="periodic",  # type: ignore[arg-type]
        )


def test_build_fsdt_station_analysis_omits_deprecated_k6() -> None:
    panel_topology = _pb()
    mesh = _strip_mesh(panel_topology, n_elements_per_panel=3)
    analysis = build_fsdt_station_analysis(
        panel_topology.section,
        panel_topology.shell_inputs,
        z_station=0.35,
        My=1.0e3,
        section_strip_mesh=mesh,
        station_id="linear-static-no-k6",
        enable_buckling=False,
    )
    assert analysis.buckling is None
    assert analysis.constitutive.K6 is None
