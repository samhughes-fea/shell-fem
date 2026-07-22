"""Production linear-static station Hashin FI workflow (``simulation_runner/api``).

``run_station_hashin`` / ``run_station_linear_static_hashin`` feed orchestration
job JSON; this locks the output contract on a real section.
"""

from __future__ import annotations

import numpy as np
import pytest

from blade_structure_utilities.section_shell_model.pre_processing.mesh.api.pipeline_b import (
    build_pipeline_b_section,
)
from blade_structure_utilities.section_shell_model.simulation_runner.api.station_recovery import (
    build_fsdt_station_analysis,
    run_station_hashin,
    run_station_linear_static_hashin,
)


def _pb():
    return build_pipeline_b_section("2412", [0.35], chord_m=1.0)


def test_run_station_linear_static_hashin_output_contract() -> None:
    pb = _pb()
    out = run_station_linear_static_hashin(
        pb, N=1.0e5, My=3.0e4, Vz=2.0e4, n_elements_per_panel=4,
    )

    assert set(out) >= {
        "cascade",
        "quaternary",
        "max_fi_hashin",
        "needs_review",
        "fi_by_panel",
        "governing",
        "track",
    }
    assert out["track"] == "linear_static"
    assert isinstance(out["needs_review"], bool)
    assert out["quaternary"] is not None


def test_run_station_hashin_defaults_to_linear_static() -> None:
    pb = _pb()
    out = run_station_hashin(pb, N=1.0e5, My=3.0e4, n_elements_per_panel=4)
    assert out["track"] == "linear_static"
    max_fi = float(out["max_fi_hashin"])
    assert np.isfinite(max_fi) and max_fi >= 0.0

    fi = np.asarray(out["fi_by_panel"], dtype=float)
    assert fi.ndim == 1 and fi.size > 0
    assert np.all(np.isfinite(fi)) and np.all(fi >= 0.0)
    assert max_fi + 1e-9 >= float(np.nanmax(fi))


def test_run_station_linear_static_hashin_governing_present() -> None:
    pb = _pb()
    out = run_station_linear_static_hashin(pb, N=1.0e5, My=3.0e4, n_elements_per_panel=4)
    gov = out["governing"]
    assert isinstance(gov, dict)
    assert "fi" in gov or float(out["max_fi_hashin"]) >= 0.0


def test_run_station_hashin_unknown_fi_track_raises() -> None:
    pb = _pb()
    with pytest.raises(ValueError, match="unknown fi_track|removed"):
        run_station_hashin(pb, N=1.0e5, My=3.0e4, n_elements_per_panel=4, fi_track="not_a_track")  # type: ignore[arg-type]


def test_run_station_hashin_periodic_track_removed() -> None:
    pb = _pb()
    with pytest.raises(ValueError, match="removed"):
        run_station_hashin(pb, N=1.0e5, My=3.0e4, n_elements_per_panel=4, fi_track="periodic")  # type: ignore[arg-type]


def test_build_fsdt_station_analysis_omits_deprecated_k6() -> None:
    pb = _pb()
    analysis = build_fsdt_station_analysis(
        pb.section,
        pb.shell_inputs,
        z_station=0.35,
        My=1.0e3,
        n_elements_per_panel=3,
        min_elements_per_panel=2,
        station_id="linear-static-no-k6",
        enable_buckling=False,
    )
    assert analysis.buckling is None
    assert analysis.constitutive.K6 is None
