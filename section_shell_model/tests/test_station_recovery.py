"""Station linear-static facade: execute → get_results → quaternary FI."""

from __future__ import annotations

import numpy as np
import pytest

from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.mesh.api.shell_panel_topology import (
    build_shell_panel_topology,
)
from blade_structure_utilities.fem_utilities.section_shell_model.simulation_runner.api.station_recovery import (
    build_fsdt_station_analysis,
    run_station_hashin,
)
from blade_structure_utilities.fem_utilities.section_shell_model.simulation_runner.section_linear_static_runner import (
    SectionLinearStaticRunner,
)


def _pb():
    return build_shell_panel_topology("2412", [0.35], chord_m=1.0)


def test_section_linear_static_runner_execute_get_results() -> None:
    panel_topology = _pb()
    static = SectionLinearStaticRunner(
        panel_topology=panel_topology,
        N=1.0e5,
        My=3.0e4,
        Vz=2.0e4,
        n_elements_per_panel=4,
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


def test_deprecated_run_station_hashin_shim_still_works() -> None:
    panel_topology = _pb()
    out = run_station_hashin(panel_topology, N=1.0e5, My=3.0e4, n_elements_per_panel=4)
    assert out["track"] == "linear_static"
    max_fi = float(out["max_fi_hashin"])
    assert np.isfinite(max_fi) and max_fi >= 0.0
    fi = np.asarray(out["fi_by_panel"], dtype=float)
    assert fi.ndim == 1 and fi.size > 0


def test_run_station_hashin_unknown_fi_track_raises() -> None:
    panel_topology = _pb()
    with pytest.raises(ValueError, match="unknown fi_track|removed"):
        run_station_hashin(panel_topology, N=1.0e5, My=3.0e4, n_elements_per_panel=4, fi_track="not_a_track")  # type: ignore[arg-type]


def test_run_station_hashin_periodic_track_removed() -> None:
    panel_topology = _pb()
    with pytest.raises(ValueError, match="removed"):
        run_station_hashin(panel_topology, N=1.0e5, My=3.0e4, n_elements_per_panel=4, fi_track="periodic")  # type: ignore[arg-type]


def test_build_fsdt_station_analysis_omits_deprecated_k6() -> None:
    panel_topology = _pb()
    analysis = build_fsdt_station_analysis(
        panel_topology.section,
        panel_topology.shell_inputs,
        z_station=0.35,
        My=1.0e3,
        n_elements_per_panel=3,
        min_elements_per_panel=2,
        station_id="linear-static-no-k6",
        enable_buckling=False,
    )
    assert analysis.buckling is None
    assert analysis.constitutive.K6 is None
