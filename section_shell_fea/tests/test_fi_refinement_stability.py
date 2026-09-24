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
    none           16.17x     1.100x   1.282x   (2026-09-20, after BL-015)
    skin_offset    0 (slave)  1.109x   1.033x

Historical 2026-08-15 numbers (wrong shear/torque pairing): untied cap 4.94x,
tied web 10.06x. Those are superseded; see ``docs/bug_log.md`` BL-014 / BL-015.

Implementation: thin Q-only wrapper over
``cascade_mesh_convergence.sweep_cascade_convergence``.
"""

from __future__ import annotations

import pytest

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.cascade_mesh_convergence import (
    series_by_kind,
    sweep_cascade_convergence,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.element_library.registry import (
    clear_formulation_registry_for_tests,
    default_formulation_id,
)

REFINEMENTS = (0.024, 0.016, 0.012)


@pytest.fixture(scope="module", autouse=True)
def _isolate_formulation_registry():
    clear_formulation_registry_for_tests()
    yield
    clear_formulation_registry_for_tests()

# Two separate situations here, and they need opposite treatment.
#
# The tied *cap* peak is `0.0` at every refinement **by design**: under
# `cap_tie_mode="skin_offset"` the cap is a tie slave and its Hashin is owned
# through the tie, so the continuum peak on the cap strip is legitimately zero.
# `test_cascade_mesh_convergence.py::test_tied_cap_hashin_slave_owned` asserts
# exactly that and passes. The two tied-cap tests below were therefore measuring
# drift of a definitionally-zero quantity — they are re-scoped, not xfailed.
#
# Tied caps must land on the same scale as the connected control components.
MAX_TIED_CAP_DRIFT = 2.0
# Untied caps remain worse than the connected control. Measured 2026-09-20 after
# the BL-015 channel-map correction: 16.17x (tied web is 1.033x on the same
# ladder). Threshold 10 still discriminates.
MIN_UNTIED_CAP_DRIFT = 10.0


def _drift(values: list[float]) -> float:
    """max/min across refinements; 1.0 means perfectly converged."""
    import numpy as np

    finite = [v for v in values if np.isfinite(v) and v > 0.0]
    assert len(finite) == len(values), f"non-finite or zero FI in sweep: {values}"
    return max(finite) / min(finite)


@pytest.fixture(scope="module")
def tied_sweep() -> dict[str, list[float]]:
    rows = sweep_cascade_convergence(REFINEMENTS, cap_tie_mode="skin_offset")
    return series_by_kind(rows, "peak_hashin")


@pytest.fixture(scope="module")
def untied_sweep() -> dict[str, list[float]]:
    rows = sweep_cascade_convergence(REFINEMENTS, cap_tie_mode="none")
    return series_by_kind(rows, "peak_hashin")


def test_tied_cap_hashin_is_tie_slave_owned(tied_sweep):
    """Under `skin_offset` the cap's continuum peak is zero, and that is correct.

    This replaces two tests that asserted the tied cap's FI *converged under
    refinement*. It cannot: the cap is a tie slave in this mode, so its continuum
    peak is definitionally `0.0` and a drift ratio over it is `0/0`, not a
    measurement. Asserting the invariant directly is the honest version — and it
    still fails loudly if the cap ever starts carrying a continuum peak, which is
    the change that would actually matter.

    The connected control lives in
    `test_connected_components_converge_regardless_of_cap_ties`; cap FI ownership
    under ties is pinned by
    `test_cascade_mesh_convergence.py::test_tied_cap_hashin_slave_owned`.
    """
    assert all(float(v) == 0.0 for v in tied_sweep["cap"]), (
        f"tied cap continuum peak is no longer zero: {tied_sweep['cap']} — "
        f"either the tie-slave ownership changed or the cap is being double-counted"
    )
    assert all(float(v) > 0.0 for v in tied_sweep["skin"])
    assert all(float(v) > 0.0 for v in tied_sweep["web"])


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


def test_untied_connected_components_converge(untied_sweep):
    """Skin and web stay connected without cap ties, so they must converge."""
    for kind in ("skin", "web"):
        drift = _drift(untied_sweep[kind])
        assert drift <= MAX_TIED_CAP_DRIFT, (
            f"untied {kind} FI drifts {drift:.3f}x; the control is not converging"
        )


def test_tied_skin_converges(tied_sweep):
    drift = _drift(tied_sweep["skin"])
    assert drift <= MAX_TIED_CAP_DRIFT, f"tied skin FI drifts {drift:.3f}x"


def test_tied_web_hashin_refinement_stability(tied_sweep, untied_sweep):
    """Tied-web Hashin must converge; the historical 10x BL-014 drift is gone.

    Measured 2026-09-20 after the BL-015 product-channel map:
    tied web 1.033x vs untied web 1.282x on ``REFINEMENTS``.
    """
    tied = _drift(tied_sweep["web"])
    untied = _drift(untied_sweep["web"])
    assert tied <= MAX_TIED_CAP_DRIFT, (
        f"tied web FI drifts {tied:.3f}x vs untied {untied:.3f}x"
    )
    assert untied <= MAX_TIED_CAP_DRIFT, f"untied web FI drifts {untied:.3f}x"


def test_refinement_gates_are_order_independent():
    """Default FSDT dispatch must not depend on leftover registry entries."""
    clear_formulation_registry_for_tests()
    assert default_formulation_id("FSDT") == "MITC4-FSDT"
    from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.element_library.contracts import (
        mitc4_fsdt_descriptor,
    )
    from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.element_library.registry import (
        ShellFormulationRegistration,
        register_formulation,
        resolve_formulation,
    )

    desc = mitc4_fsdt_descriptor()
    register_formulation(
        ShellFormulationRegistration(
            descriptor=desc.__class__(
                **{
                    **{f.name: getattr(desc, f.name) for f in desc.__dataclass_fields__.values()},
                    "formulation_id": "TEST-REFINEMENT-FSDT",
                }
            ),
            factory_cls=resolve_formulation("MITC4-FSDT").factory_cls,
            kernel=resolve_formulation("MITC4-FSDT").kernel,
            projector=resolve_formulation("MITC4-FSDT").projector,
        )
    )
    assert default_formulation_id("FSDT") == "MITC4-FSDT"
    clear_formulation_registry_for_tests()
    assert default_formulation_id("FSDT") == "MITC4-FSDT"
