"""Tier 3 (CLT recovery) orchestrator: parity with the pre-split fused path.

Covers PROCESSING_ARCHITECTURE_PROPOSAL.md Phase 2 acceptance: the batched ABD
solve and ply-stress recovery extracted into ``compute_tertiary`` reproduce the
same intermediate values the fused ``api/fi_recovery.hashin_fi_per_ply_from_rows_batch``
used before the split, and the facade's ``(fi_out, needs_review)`` are unaffected.
"""

from __future__ import annotations

import numpy as np
import pytest

from blade_precompute.materials.laminate import LaminateDefinition
from blade_precompute.materials.clpt_recovery import ply_mid_stresses_laminate
from blade_precompute.materials.ply import OrthotropicPly
from blade_precompute.section_shell_model.pre_processing.containers.formulation_objects import ShellFormulationCache
from blade_precompute.section_shell_model.pre_processing.containers.types import ShellPanelResultants
from blade_precompute.section_shell_model.api.fi_recovery import (
    hashin_fi_per_ply_from_rows_batch,
    clpt_inputs_trusted as api_clpt_inputs_trusted,
)
from blade_precompute.section_shell_model.processing.results.compute_tertiary import (
    ShellTertiaryResultsOrchestrator,
    batched_clt_strains,
    clpt_inputs_trusted,
)
from blade_precompute.section_shell_model.processing.results.containers.container_hopper import (
    ShellSecondaryResultSet,
)
from blade_precompute.section_shell_model.processing.results.containers.elemental_results import (
    ShellElementalResults,
)


def _ply() -> OrthotropicPly:
    return OrthotropicPly(
        name="gfrp", E1=30e9, E2=8e9, G12=3e9, nu12=0.3, rho=1.5e3, t_ply=2e-4,
        Xt=500e6, Xc=300e6, Yt=50e6, Yc=200e6, S12=40e6, Zt=1e6, S13=1e6, S23=1e6,
    )


def _lam(angle_deg: float = 0.0) -> LaminateDefinition:
    p = _ply()
    return LaminateDefinition(plies=[(p, angle_deg), (p, -angle_deg)])


def _rows(n: int, *, panel_index: int = 0) -> list[ShellPanelResultants]:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        rows.append(
            ShellPanelResultants(
                Nx=float(rng.uniform(-1e4, 1e4)),
                Ny=float(rng.uniform(-1e3, 1e3)),
                Nxy=float(rng.uniform(-1e3, 1e3)),
                Mx=float(rng.uniform(-10, 10)),
                My=float(rng.uniform(-50, 50)),
                Mxy=float(rng.uniform(-5, 5)),
                Qx=1.0,
                Qy=2.0,
                thickness_m=8e-4,
                panel_index=panel_index,
                station_index=i,
            )
        )
    return rows


def test_provenance_gating_moved_and_reexported_identically():
    """clpt_inputs_trusted behaves the same via tier 3 and the api re-export."""
    row = _rows(1)[0]
    assert clpt_inputs_trusted(row) == api_clpt_inputs_trusted(row)
    assert clpt_inputs_trusted(row) is True


def test_batched_clt_strains_matches_direct_abd_solve():
    lam = _lam(0.0)
    rows = _rows(5)
    eps0, kappa, trusted = batched_clt_strains(rows, lam)
    assert np.all(trusted)

    abd = lam.stiffness_bundle().ABD
    for i, row in enumerate(rows):
        nm = np.concatenate([row.to_N_vec()[:3], row.to_M_vec()[:3]])
        sol = np.linalg.solve(abd, nm)
        np.testing.assert_allclose(eps0[i], sol[:3])
        np.testing.assert_allclose(kappa[i], sol[3:6])


def test_batched_clt_strains_respects_provenance_gate():
    from blade_precompute.section_shell_model.pre_processing.containers.types import (
        FieldProvenance,
        ProvenanceKind,
    )

    lam = _lam(0.0)
    rows = _rows(3)
    rows[1].provenance["My"] = FieldProvenance(ProvenanceKind.PLACEHOLDER, "stub")
    eps0, kappa, trusted = batched_clt_strains(rows, lam)
    assert trusted[0] and trusted[2]
    assert not trusted[1]
    assert np.all(eps0[1] == 0.0)
    assert np.all(kappa[1] == 0.0)


def test_tertiary_orchestrator_ply_stresses_match_fused_path_for_zero_deg_plies():
    """0deg plies: laminate frame == material frame, so sig_ply_mat matches the
    unrotated intermediate the fused hashin_fi_per_ply_from_rows_batch computes."""
    lam = _lam(0.0)
    rows = _rows(4, panel_index=0)
    cache = ShellFormulationCache(stiffness_objects=[], force_objects=[], panel_laminates=[lam])
    secondary = ShellSecondaryResultSet(elemental_results=ShellElementalResults(resultant_rows=rows))

    tertiary = ShellTertiaryResultsOrchestrator(secondary=secondary, cache=cache).compute()
    assert tertiary.ply_results is not None
    assert np.all(tertiary.ply_results.trusted_mask)

    for i, row in enumerate(rows):
        eps0 = tertiary.ply_results.eps0[i]
        kappa = tertiary.ply_results.kappa[i]
        sig_lam_expected = np.asarray(ply_mid_stresses_laminate(lam, eps0, kappa), dtype=float)
        np.testing.assert_allclose(
            tertiary.ply_results.sig_ply_mat[i, : len(lam.plies)],
            sig_lam_expected,
            atol=1e-8,
        )

    # Pass-through fields for tier 4 (avoids reaching back to tier 2).
    np.testing.assert_allclose(tertiary.transverse_shear[:, 0], 1.0)
    np.testing.assert_allclose(tertiary.transverse_shear[:, 1], 2.0)
    np.testing.assert_allclose(tertiary.thickness_m, 8e-4)


def test_tertiary_orchestrator_groups_rows_by_panel_index():
    """Rows from two panels with different laminates are solved independently."""
    lam_a = _lam(0.0)
    lam_b = LaminateDefinition(plies=[(_ply(), 0.0)])
    rows = _rows(2, panel_index=0) + _rows(2, panel_index=1)
    cache = ShellFormulationCache(
        stiffness_objects=[], force_objects=[], panel_laminates=[lam_a, lam_b],
    )
    secondary = ShellSecondaryResultSet(elemental_results=ShellElementalResults(resultant_rows=rows))

    tertiary = ShellTertiaryResultsOrchestrator(secondary=secondary, cache=cache).compute()
    assert np.all(tertiary.ply_results.trusted_mask)
    # n_ply_max reflects the larger laminate (lam_a has 2 plies, lam_b has 1).
    assert tertiary.ply_results.sig_ply_mat.shape[1] == 2
    # lam_b rows (panel 1) only populate the first ply slot; the second is padding.
    np.testing.assert_allclose(tertiary.ply_results.sig_ply_mat[2:, 1], 0.0)


def test_hashin_fi_per_ply_from_rows_batch_unchanged_after_clt_extraction():
    """Facade regression: fi_out/needs_review reproduce a from-scratch reference
    computation (independent of the refactored implementation) for off-axis plies."""
    from blade_precompute.materials.failure_criteria import hashin_fi

    lam = _lam(30.0)
    plies = []
    rows = _rows(6)
    fi_out, needs_review = hashin_fi_per_ply_from_rows_batch(rows, lam, plies)

    abd = lam.stiffness_bundle().ABD
    Xt = np.array([p.Xt for p, _ in lam.plies])
    Xc = np.array([p.Xc for p, _ in lam.plies])
    Yt = np.array([p.Yt for p, _ in lam.plies])
    Yc = np.array([p.Yc for p, _ in lam.plies])
    S12 = np.array([p.S12 for p, _ in lam.plies])

    for ri, row in enumerate(rows):
        assert not needs_review[ri]
        nm = np.concatenate([row.to_N_vec()[:3], row.to_M_vec()[:3]])
        sol = np.linalg.solve(abd, nm)
        sig_lam = ply_mid_stresses_laminate(lam, sol[:3], sol[3:6])
        for pi, sig in enumerate(sig_lam):
            sig3 = np.asarray(sig, dtype=float).ravel()[:3]
            expected = float(
                hashin_fi(
                    sig3.reshape(1, 3),
                    Xt[pi : pi + 1], Xc[pi : pi + 1], Yt[pi : pi + 1], Yc[pi : pi + 1], S12[pi : pi + 1],
                )[0]
            )
            assert fi_out[ri, pi] == pytest.approx(expected)
