"""Tier 3 (CLT recovery) orchestrator: parity with the pre-split fused path.

Covers PROCESSING_ARCHITECTURE_PROPOSAL.md Phase 2 acceptance: the batched ABD
solve and ply-stress recovery extracted into ``compute_tertiary`` reproduce the
same intermediate values the fused ``api/fi_recovery.hashin_fi_per_ply_from_rows_batch``
used before the split, and the facade's ``(fi_out, needs_review)`` are unaffected.
"""

from __future__ import annotations

import numpy as np
import pytest

from blade_structure_utilities.material_utilities.laminate import (
    FACE_MID,
    LaminateDefinition,
    ply_face_stresses_laminate,
    ply_mid_stresses_laminate,
)
from blade_structure_utilities.material_utilities.orthotropic import OrthotropicPly
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.formulation_objects import ShellFormulationCache
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import ShellPanelResultants
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.api.fi_recovery import (
    hashin_fi_per_ply_from_rows_batch,
    clpt_inputs_trusted as api_clpt_inputs_trusted,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.compute_tertiary import (
    ShellTertiaryResultsOrchestrator,
    batched_clt_strains,
    clpt_inputs_trusted,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.containers.container_hopper import (
    ShellSecondaryResultSet,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.containers.elemental_results import (
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
                N11=float(rng.uniform(-1e4, 1e4)),
                N22=float(rng.uniform(-1e3, 1e3)),
                N12=float(rng.uniform(-1e3, 1e3)),
                M11=float(rng.uniform(-10, 10)),
                M22=float(rng.uniform(-50, 50)),
                M12=float(rng.uniform(-5, 5)),
                Q1=1.0,
                Q2=2.0,
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
    from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import (
        FieldProvenance,
        ProvenanceKind,
    )

    lam = _lam(0.0)
    rows = _rows(3)
    rows[1].provenance["M22"] = FieldProvenance(ProvenanceKind.PLACEHOLDER, "stub")
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
        sig_faces = np.asarray(ply_face_stresses_laminate(lam, eps0, kappa), dtype=float)
        assert tertiary.ply_results.sig_ply_mat.shape[-2:] == (3, 3)
        np.testing.assert_allclose(
            tertiary.ply_results.sig_ply_mat[i, : len(lam.plies)],
            sig_faces,
            atol=1e-8,
        )
        sig_mid_expected = np.asarray(ply_mid_stresses_laminate(lam, eps0, kappa), dtype=float)
        np.testing.assert_allclose(
            tertiary.ply_results.sig_ply_mat[i, : len(lam.plies), FACE_MID, :],
            sig_mid_expected,
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
    assert tertiary.ply_results.sig_ply_mat.shape[1:] == (2, 3, 3)
    # lam_b rows (panel 1) only populate the first ply slot; the second is padding.
    np.testing.assert_allclose(tertiary.ply_results.sig_ply_mat[2:, 1], 0.0)


def test_hashin_fi_per_ply_from_rows_batch_evaluates_in_material_frame():
    """Facade regression: fi_out/needs_review reproduce a from-scratch reference
    computation (independent of the refactored implementation) for off-axis plies.

    The reference rotates E→M with the canonical ``T_sigma`` before applying the
    criterion, because Hashin is only valid in **M**. At 30° plies the E-frame
    value is several times too low, so this also pins the rotation itself.
    """
    from blade_structure_utilities.geometry_utilities.coordinate_frames import (
        rotate_ply_stress_E_to_M,
    )
    from blade_structure_utilities.material_utilities.failure import hashin_fi

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
            sig3 = rotate_ply_stress_E_to_M(
                np.asarray(sig, dtype=float).ravel()[:3], lam.plies[pi][1]
            )
            expected = float(
                hashin_fi(
                    sig3.reshape(1, 3),
                    Xt[pi : pi + 1], Xc[pi : pi + 1], Yt[pi : pi + 1], Yc[pi : pi + 1], S12[pi : pi + 1],
                    np.array([p.S23 for p, _ in lam.plies])[pi : pi + 1],
                )[0]
            )
            assert fi_out[ri, pi] == pytest.approx(expected)


def test_cache_formulation_id_survives_tertiary_and_quaternary():
    """A registered non-default formulation id is copied cache → T → Q unchanged."""
    from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.element_library.contracts import (
        mitc4_fsdt_descriptor,
    )
    from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.element_library.registry import (
        ShellFormulationRegistration,
        clear_formulation_registry_for_tests,
        default_formulation_id,
        register_formulation,
        resolve_formulation,
    )
    from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.compute_quaternary import (
        ShellQuaternaryResultsOrchestrator,
    )

    alt_id = "TEST-CACHE-FSDT"
    desc = mitc4_fsdt_descriptor()
    register_formulation(
        ShellFormulationRegistration(
            descriptor=desc.__class__(
                **{
                    **{f.name: getattr(desc, f.name) for f in desc.__dataclass_fields__.values()},
                    "formulation_id": alt_id,
                }
            ),
            factory_cls=resolve_formulation("MITC4-FSDT").factory_cls,
            kernel=resolve_formulation("MITC4-FSDT").kernel,
            projector=resolve_formulation("MITC4-FSDT").projector,
        )
    )
    try:
        assert default_formulation_id("FSDT") == "MITC4-FSDT"
        lam = _lam(0.0)
        rows = _rows(2)
        cache = ShellFormulationCache(
            stiffness_objects=[],
            force_objects=[],
            panel_laminates=[lam],
            formulation_id=alt_id,
            analysis_dof_layout_id="test_layout",
        )
        secondary = ShellSecondaryResultSet(
            elemental_results=ShellElementalResults(resultant_rows=rows)
        )
        tertiary = ShellTertiaryResultsOrchestrator(secondary=secondary, cache=cache).compute()
        assert tertiary.meta["formulation_id"] == alt_id
        assert tertiary.meta["analysis_dof_layout_id"] == "test_layout"
        assert set(tertiary.ply_results.formulation_id) == {alt_id}

        quaternary = ShellQuaternaryResultsOrchestrator(tertiary=tertiary, cache=cache).compute()
        assert quaternary.formulation_id == alt_id
        assert quaternary.meta["formulation_id"] == alt_id
    finally:
        clear_formulation_registry_for_tests()
        assert default_formulation_id("FSDT") == "MITC4-FSDT"
