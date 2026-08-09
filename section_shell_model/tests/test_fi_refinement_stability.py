"""Hashin FI must be stable under mesh refinement (docs/bug_log.md BL-003).

Nothing anywhere asserted that FI settles as the mesh is refined, which is why an
unconstrained spar cap could report a factor-of-700 swing in cap FI across three
refinement levels and still look like a normal result. FI is the *only* active
sizing gate at every station on the reference rotor, so an FI that moves with the
mesh moves the design.

The discriminator is the untied case: with ``cap_tie_mode="none"`` the caps have
no load path into the section, so their in-plane response is set by their own
strip stiffness rather than by equilibrium, and refining makes it worse instead
of better. Skin and web are the control — they are connected under both settings
and must converge either way.

Measured on this fixture (5D-F, chord 0.241 m, twist 34.1 deg, torsion-dominated
load), FI drift as max/min across the three refinements:

    cap_tie_mode   cap        skin     web
    none           764.343x   1.146x   1.550x
    skin_offset      1.165x   1.173x   1.174x

Tied, the cap tracks the control to within 1%. The thresholds below sit well
clear of both numbers so the tests discriminate rather than merely record.
"""

from __future__ import annotations

import numpy as np
import pytest

from blade_structure_utilities.geometry_utilities.section_contract.beam_resultant_vector import (
    BeamResultantVector,
)
from blade_structure_utilities.geometry_utilities.section_geometry.api.factory import (
    build_section_view,
    resolve_system_type,
)
from blade_structure_utilities.geometry_utilities.section_geometry.sdf.airfoil import (
    AirfoilSDF,
)
from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.mesh.api.shell_panel_topology import (
    shell_panel_topology_from_shell_inputs,
)
from blade_structure_utilities.fem_utilities.section_shell_model.simulation_runner.section_linear_static_runner import (
    SectionLinearStaticRunner,
)
from blade_structure_utilities.load_utilities.beam_to_shell_loads.shape_external import (
    shape_section_force_vector,
)
from blade_structure_utilities.mesh_utilities.section_mesh import (
    build_section_strip_mesh,
    build_shell_mesh_inputs,
)

LAYOUT = "5D-F"
CHORD = 0.241
TWIST_DEG = 34.1
THICKNESS = {"skin": 0.001, "spar": 0.004, "web": 0.0015}
# Torsion-dominated so in-plane shear governs, which is where an untied cap is
# worst behaved -- the response BL-003 is about.
LOAD = {"T": 5.0e3, "Vy": 2.0e4, "Mz": 1.0e4}
# Element length as a fraction of chord. Three levels is enough to separate
# 1.2x from 764x and keeps the whole module around six seconds.
REFINEMENTS = (0.024, 0.016, 0.012)

# Tied caps must land on the same scale as the connected control components.
MAX_TIED_CAP_DRIFT = 2.0
# ... and not far outside what skin and web do on the same meshes.
MAX_CAP_DRIFT_VS_CONTROL = 3.0
# Untied caps must be dramatically worse, or this fixture has stopped
# discriminating and the tied result above means nothing.
MIN_UNTIED_CAP_DRIFT = 10.0


def _fi_by_kind(frac: float, cap_tie_mode: str) -> dict[str, float]:
    """Peak in-plane Hashin FI per component kind at one refinement level."""
    twist = float(np.deg2rad(TWIST_DEG))
    airfoil = AirfoilSDF.from_naca_series(
        6, 63.0, 4.0, 15.0, n_points=200, chord=CHORD, closed_te=True
    )
    section = build_section_view(
        airfoil,
        resolve_system_type(LAYOUT),
        twist_angle_rad=twist,
        skin_thickness=THICKNESS["skin"],
        web_thickness=THICKNESS["web"],
        cap_height=THICKNESS["spar"],
    )
    shell = build_shell_mesh_inputs(
        section, twist_rad=twist, layout_key=LAYOUT, n_web_samples=20, n_cap_samples=80
    )
    mesh = build_section_strip_mesh(
        shell, target_element_length_frac=frac, endpoint_tol=1e-6
    )
    resultants = BeamResultantVector.from_named(**LOAD)
    runner = SectionLinearStaticRunner(
        panel_topology=shell_panel_topology_from_shell_inputs(shell),
        section_strip_mesh=mesh,
        resultants=resultants,
        # Only the thinwall channel loads the caps; equipollent drops every strip
        # whose kind is not skin/web, which would hide the defect entirely.
        section_force_vector=shape_section_force_vector(resultants, shell.midlines),
        load_channel="thinwall",
        cap_tie_mode=cap_tie_mode,
    )
    runner.execute()

    results = runner.get_results()
    quaternary = results.quaternary_results
    hashin = None if quaternary is None else quaternary.hashin_fi
    if hashin is None or hashin.in_plane is None:
        raise AssertionError("no in-plane Hashin FI in the quaternary results")
    in_plane = np.asarray(hashin.in_plane, dtype=float)
    panel_index = np.asarray(results.tertiary_results.panel_index)
    labels = list(getattr(runner.assembly, "lab_p", None) or [])

    by_kind: dict[str, float] = {}
    for pi, label in enumerate(labels):
        text = str(label)
        kind = text.split(":", 1)[0] if ":" in text else "panel"
        rows = np.where(panel_index == pi)[0]
        if rows.size:
            by_kind[kind] = max(by_kind.get(kind, 0.0), float(np.nanmax(in_plane[rows])))
    return by_kind


def _drift(values: list[float]) -> float:
    """max/min across refinements; 1.0 means perfectly converged."""
    finite = [v for v in values if np.isfinite(v) and v > 0.0]
    assert len(finite) == len(values), f"non-finite or zero FI in sweep: {values}"
    return max(finite) / min(finite)


@pytest.fixture(scope="module")
def tied_sweep() -> dict[str, list[float]]:
    return _sweep("skin_offset")


@pytest.fixture(scope="module")
def untied_sweep() -> dict[str, list[float]]:
    return _sweep("none")


def _sweep(cap_tie_mode: str) -> dict[str, list[float]]:
    series: dict[str, list[float]] = {}
    for frac in REFINEMENTS:
        fi = _fi_by_kind(frac, cap_tie_mode)
        for kind in ("cap", "skin", "web"):
            series.setdefault(kind, []).append(fi.get(kind, float("nan")))
    return series


def test_tied_cap_fi_converges_under_refinement(tied_sweep):
    """With ties on, cap FI settles instead of running away."""
    drift = _drift(tied_sweep["cap"])
    assert drift <= MAX_TIED_CAP_DRIFT, (
        f"cap FI drifts {drift:.3f}x across {REFINEMENTS} with ties on; "
        f"values {tied_sweep['cap']}"
    )


def test_tied_cap_fi_tracks_the_connected_control(tied_sweep):
    """Cap drift must be on the same scale as skin and web, not merely bounded.

    Skin and web are connected under either setting, so they measure how much of
    the drift is ordinary discretisation error. A cap that converges but an order
    of magnitude more slowly is still not behaving like a connected component.
    """
    cap = _drift(tied_sweep["cap"])
    control = max(_drift(tied_sweep["skin"]), _drift(tied_sweep["web"]))
    assert cap <= MAX_CAP_DRIFT_VS_CONTROL * control, (
        f"cap FI drift {cap:.3f}x against a skin/web control of {control:.3f}x"
    )


def test_untied_cap_fi_diverges_so_the_test_discriminates(untied_sweep):
    """The guard on the guard: without ties this must fail loudly.

    If this ever stops holding, the fixture no longer exercises the defect and
    the two tests above prove nothing.
    """
    cap = _drift(untied_sweep["cap"])
    control = max(_drift(untied_sweep["skin"]), _drift(untied_sweep["web"]))
    assert cap >= MIN_UNTIED_CAP_DRIFT, (
        f"untied cap FI drift is only {cap:.3f}x; the fixture no longer "
        f"discriminates, so the tied assertions are vacuous"
    )
    assert cap > control, "untied cap must be worse than the connected control"


def test_connected_components_converge_regardless_of_cap_ties(tied_sweep, untied_sweep):
    """Skin and web are connected either way, so both settings must converge.

    Pins the control itself: if skin or web started diverging, the comparison
    above would be measuring something other than the cap's load path.
    """
    for label, sweep in (("tied", tied_sweep), ("untied", untied_sweep)):
        for kind in ("skin", "web"):
            drift = _drift(sweep[kind])
            assert drift <= MAX_TIED_CAP_DRIFT, (
                f"{label} {kind} FI drifts {drift:.3f}x; the control is not "
                f"converging, so the cap comparison is not meaningful"
            )
