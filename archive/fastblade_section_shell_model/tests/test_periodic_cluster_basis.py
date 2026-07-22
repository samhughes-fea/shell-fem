"""Periodic homogeniser × cluster-basis (transformed_basis) MPC integration.

Regression target for the singularity documented in
``docs/PERIODIC_CLUSTER_BASIS_SINGULARITY.md``: the resultant-driven periodic
solve (``compute_k6_periodic_bc``) used to raise ``Factor is exactly singular``
with ``interface_constraint_mode="transformed_basis"`` because the cluster-basis
MPCs (and their virtual reference nodes) were dropped by the periodic path.

The fix composes the cluster MPC reduction with periodic pairing and gives the
virtual cluster nodes geometry so they periodic-pair. These tests lock in:

  1. non-singularity on the production MITC4 (ndof=6) path,
  2. mesh-layer stability under refinement (pairing/geometry artifacts stable
     across ne for the same topology),
  3. axial/bending K6 equivalence to ``shared`` (these modes are insensitive to
     the rotated-basis junction treatment).

NOTE: torsion/transverse-shear K6 at the sharp ~90deg skin/web junction differs
substantially from ``shared`` (which over-constrains there). Validating that
those modes are *correct* (near-collinear equivalence + equilibrium recovery) is
deferred — see the doc's Step 3. Do not assert it here.
"""

from __future__ import annotations

import numpy as np
import pytest

from blade_precompute.section_shell_model.api.pipeline_b import build_pipeline_b_section
from blade_precompute.section_shell_model.engine.global_shell_assembly import (
    assemble_for_periodic_k6,
)
from blade_precompute.section_shell_model.engine.homogenisation import (
    compute_elastic_centroid_from_panels,
    compute_k6_periodic_bc,
)

NE = 6


def _section():
    # NACA 2412 + single spar web -> non-collinear skin/web/cap junctions.
    return build_pipeline_b_section("2412", [0.35], chord_m=1.0)


def _zeros(panels):
    return [np.zeros(len(np.asarray(p.s, dtype=float))) for p in panels]


def _k6_for(panels, mode: str, ne: int = NE):
    z = _zeros(panels)
    ye, ze = compute_elastic_centroid_from_panels(panels)
    K, _f, nm, pm, *_rest = assemble_for_periodic_k6(
        panels, z, z, n_elements_per_panel=ne,
        interface_constraint_mode=mode,
    )
    K6, _ = compute_k6_periodic_bc(K, nm, pm, panels, ye, ze)
    return K6, nm


def _k6(mode: str, ne: int = NE):
    return _k6_for(_section().panels, mode, ne=ne)


def _ngon_panels(n_sides: int, R: float = 0.5):
    """Closed regular N-gon tube: N straight panels, junction angle = 360/N deg.

    All junctions share the same (near-collinear for large N) angle, so as
    n_sides grows every junction -> collinear and `transformed_basis` K6 MUST
    converge to `shared` K6 for ALL modes (the cluster rotation -> identity).
    Closed loop => the periodic K_rr is well-conditioned (an open chain is
    under-constrained in the periodic BC and would be singular).
    """
    from blade_precompute.section_shell_model.pre_processing.constitutive.default_laminates import (
        WallLaminate,
    )
    from blade_precompute.section_shell_model.pre_processing.containers.panel_types import Panel

    lam = WallLaminate(E=45e9, t=1e-3, name="iso")
    th = np.linspace(0.0, 2 * np.pi, n_sides, endpoint=False)
    verts = np.column_stack([R * np.cos(th), R * np.sin(th)])  # built once (exact shared endpoints)
    panels = []
    for i in range(n_sides):
        a, b = verts[i], verts[(i + 1) % n_sides]
        nodes = np.array([a, 0.5 * (a + b), b])  # midpoint interior node => well-defined end tangent
        panels.append(Panel(nodes=nodes, lam=lam, label=f"poly{i}"))
    return panels


def _worst_mode_reldiff(panels) -> float:
    """Worst per-mode relative difference of K6 diagonal, transformed vs shared."""
    K6s, _ = _k6_for(panels, "shared")
    K6t, _ = _k6_for(panels, "transformed_basis")
    ds, dt = np.diag(K6s), np.diag(K6t)
    floor = 1e-30 * abs(np.trace(K6s))
    return float((np.abs(dt - ds) / np.maximum(np.abs(ds), floor)).max())


def test_periodic_transformed_basis_not_singular() -> None:
    """Previously raised 'Factor is exactly singular' on the MITC4 path."""
    K6, _ = _k6("transformed_basis")
    assert np.all(np.isfinite(K6))
    # K6 must be a genuine (non-degenerate) stiffness for the energetic modes.
    assert np.linalg.cond(K6) < 1e16


def test_cluster_mpcs_attached_to_node_meta() -> None:
    _, nm = _k6("transformed_basis")
    assert nm.constraints, "cluster-basis MPCs must be attached for the periodic compose"
    # Virtual cluster reference nodes received geometry + a periodic pair group.
    assert np.any((nm.pair_group >= 0) & ~np.isnan(nm.yz[:, 0]))


def test_mesh_layer_stable_under_refinement() -> None:
    """Pairing topology and nodal geometry must remain coherent under mesh refinement.

    Finer meshes add interior nodes but preserve the same pair_group structure for
    virtual cluster refs and panel endpoints; geometry entries stay finite.
    """
    _, nm_coarse = _k6("transformed_basis", ne=4)
    _, nm_fine = _k6("transformed_basis", ne=8)
    assert nm_fine.yz.shape[0] > nm_coarse.yz.shape[0]
    assert np.all(np.isfinite(nm_coarse.yz) | np.isnan(nm_coarse.yz))
    assert np.all(np.isfinite(nm_fine.yz) | np.isnan(nm_fine.yz))
    # Virtual cluster refs keep a periodic pair group at both resolutions.
    assert np.any((nm_coarse.pair_group >= 0) & ~np.isnan(nm_coarse.yz[:, 0]))
    assert np.any((nm_fine.pair_group >= 0) & ~np.isnan(nm_fine.yz[:, 0]))
    # Endpoint / junction pair IDs present in the coarse mesh remain in the fine mesh.
    coarse_pairs = set(int(g) for g in nm_coarse.pair_group if g >= 0)
    fine_pairs = set(int(g) for g in nm_fine.pair_group if g >= 0)
    assert coarse_pairs.issubset(fine_pairs)


def test_axial_bending_k6_matches_shared() -> None:
    """Axial (eps0) and bending (kappa_y, kappa_z) K6 are insensitive to the
    rotated-basis junction treatment, so transformed_basis must match shared."""
    K6s, _ = _k6("shared")
    K6t, _ = _k6("transformed_basis")
    for i in (0, 1, 2):  # eps0, kappa_y, kappa_z
        assert K6t[i, i] == pytest.approx(K6s[i, i], rel=0.05), (
            f"mode {i}: transformed={K6t[i, i]:.4e} shared={K6s[i, i]:.4e}"
        )


# --- Discriminator (doc Step 3.1): near-collinear equivalence -----------------
# DECISION RESULT (2026-06): this is the regression target for an OPEN BUG in the
# cluster-basis periodic compose, currently xfail(strict). Evidence (closed N-gon
# tube, MITC4): `shared` K6 reproduces the analytic beam EA exactly (A11*perimeter)
# and is mesh/N-stable; `transformed_basis` is WRONG and does NOT converge to it as
# the junction angle -> 0:
#   - axial K6 is a fixed 0.877x of EA (12% soft), mesh- AND N-independent, with
#     large spurious in-plane "breathing" of the cross-section (|u_s|~0.12, |w|~0.21
#     under a pure-u_x axial mode where `shared` gives ~0);
#   - torsion K6 DIVERGES (grows ~ N^2 and ~ n_elem) instead of converging.
# Ruled out: not a singular/near-mechanism (clamping up to 153 cols changes nothing);
# not the rotated-basis MPC (bending K6 is exact); not the slave-jump tangent
# hypothesis (axial is tangent-independent yet wrong). The fault is in how the
# virtual cluster reference nodes participate in the periodic jump/pairing in
# `_build_periodic_transform_clustered` (a spurious junction freedom that breaks the
# rigid-cross-section beam kinematics `shared` enforces). Remove the xfail when fixed.
#
# Spike 1A (2026-07-14) reconfirmed on N=24/48/96 (ne=4): axial ratio exactly
# 0.8775; torsion ratios ~259 / 918 / 3830; under unit axial, transformed_basis
# has |u_s|~0.19 / |w|~0.35 while shared has ~1e-15. Constraint-manifold issue.


def test_spike_1a_ngon_wrong_k6_signature() -> None:
    """Lock the documented wrong-K6 signature (formulation bug, not coding bug).

    This is not the pass oracle (see xfail discriminators); it ensures the failure
    mode stays the known shape if someone changes the periodic compose.
    """
    panels = _ngon_panels(48)
    K6s, _ = _k6_for(panels, "shared", ne=4)
    K6t, nm = _k6_for(panels, "transformed_basis", ne=4)
    # Axial soft by the fixed ~12% factor; bending matches.
    assert K6t[0, 0] / K6s[0, 0] == pytest.approx(0.8775, rel=1e-3)
    assert K6t[1, 1] == pytest.approx(K6s[1, 1], rel=1e-4)
    assert K6t[2, 2] == pytest.approx(K6s[2, 2], rel=1e-4)
    # Torsion over-stiff and grows with N (check ratio >> 1 on N=48).
    assert K6t[3, 3] / K6s[3, 3] > 100.0
    # Spurious in-plane breathing under axial macro (shared is ~machine zero).
    from blade_precompute.section_shell_model.engine.homogenisation import (
        compute_elastic_centroid_from_panels,
        compute_k6_periodic_bc,
    )
    z = _zeros(panels)
    ye, ze = compute_elastic_centroid_from_panels(panels)
    K, _f, nm2, pm, *_ = assemble_for_periodic_k6(
        panels, z, z, n_elements_per_panel=4,
        interface_constraint_mode="transformed_basis",
    )
    _K6, u_modes = compute_k6_periodic_bc(K, nm2, pm, panels, ye, ze)
    ndof = 6
    u_ax = u_modes[:, 0]
    assert float(np.max(np.abs(u_ax[1::ndof]))) > 0.05  # US breathing
    assert float(np.max(np.abs(u_ax[2::ndof]))) > 0.05  # W breathing
    assert nm.constraints  # manifold includes cluster MPCs
@pytest.mark.xfail(strict=True, reason="open bug: transformed_basis periodic K6 does not "
                   "converge to shared as junction angle -> 0 (axial 12% soft, torsion diverges)")
def test_collinear_ngon_limit_transformed_equals_shared() -> None:
    # High-N polygon (junction ~3.75 deg): cluster rotation ~ identity, so all 6
    # K6 modes must match shared. Modes 3-5 (torsion/shear) are the discriminator.
    panels = _ngon_panels(96)
    K6s, _ = _k6_for(panels, "shared")
    K6t, _ = _k6_for(panels, "transformed_basis")
    ds, dt = np.diag(K6s), np.diag(K6t)
    floor = 1e-30 * abs(np.trace(K6s))
    r = np.abs(dt - ds) / np.maximum(np.abs(ds), floor)
    assert r.max() < 0.05, f"per-mode rel err {np.round(r, 4)} (modes 3-5 are the discriminator)"


@pytest.mark.xfail(strict=True, reason="open bug: transformed_basis periodic K6 diverges "
                   "from shared as junction angle -> 0 instead of converging")
def test_transformed_converges_to_shared_as_collinear() -> None:
    # As junction angle -> 0 the worst-mode error must shrink toward 0.
    worst = [_worst_mode_reldiff(_ngon_panels(N)) for N in (24, 48, 96)]
    assert worst[2] < 0.5 * worst[0], f"no convergence: worst-mode err sweep (N=24,48,96) = {worst}"
