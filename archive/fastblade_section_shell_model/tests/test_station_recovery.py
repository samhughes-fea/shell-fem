"""Production periodic-solve + Hashin FI workflow (``api/station_recovery.py``).

``run_station_periodic_hashin`` is the v2 orchestration entry point feeding job JSON;
this locks its output contract and provenance auditing on a real section.
"""

from __future__ import annotations

import numpy as np

from blade_precompute.section_shell_model.api.station_recovery import (
    run_station_periodic_hashin,
)

from blade_precompute.section_shell_model.api.pipeline_b import build_pipeline_b_section
from blade_precompute.section_shell_model.pre_processing.containers.types import SectionShellRecoveryBundle


def _pb():
    return build_pipeline_b_section("2412", [0.35], chord_m=1.0)


def test_run_station_periodic_hashin_output_contract() -> None:
    pb = _pb()
    out = run_station_periodic_hashin(
        pb, N=1.0e5, My=3.0e4, Vz=2.0e4, n_elements_per_panel=4,
    )

    assert set(out) >= {
        "bundle",
        "max_fi_hashin",
        "needs_review",
        "fi_by_panel",
        "provenance_breakdown",
        "governing_element",
    }
    assert isinstance(out["bundle"], SectionShellRecoveryBundle)
    assert isinstance(out["needs_review"], bool)


def test_run_station_periodic_hashin_values_are_physical() -> None:
    pb = _pb()
    out = run_station_periodic_hashin(pb, N=1.0e5, My=3.0e4, n_elements_per_panel=4)

    max_fi = float(out["max_fi_hashin"])
    assert np.isfinite(max_fi) and max_fi >= 0.0

    fi = np.asarray(out["fi_by_panel"], dtype=float)
    assert fi.shape[0] == len(out["bundle"].panels)
    assert np.all(np.isfinite(fi)) and np.all(fi >= 0.0)
    # The governing FI must be at least the per-panel maximum.
    assert max_fi + 1e-9 >= float(fi.max())


def test_run_station_periodic_hashin_provenance_is_trusted() -> None:
    pb = _pb()
    out = run_station_periodic_hashin(pb, N=1.0e5, My=3.0e4, n_elements_per_panel=4)

    breakdown = out["provenance_breakdown"]
    assert breakdown, "provenance breakdown must not be empty for a real solve"
    # FSDT periodic recovery must be tagged as the trusted MITC4 kernel, never a placeholder.
    assert "mitc4" in breakdown
    assert "placeholder" not in breakdown
