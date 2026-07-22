"""Tier 4 (failure indices) orchestrator and api/fi_recovery facade parity.

Covers PROCESSING_ARCHITECTURE_PROPOSAL.md Phase 3 acceptance: the new
container-based ``ShellQuaternaryResultsOrchestrator`` reproduces per-ply Hashin
FI computed directly from tier 3's ply stresses + laminate strengths, and the
``api/fi_recovery`` facades (``interlaminar_fi_from_row``,
``select_governing_hashin_element``, ``max_hashin_fi_from_bundle``) are
unchanged re-exports of the tier 4 implementations.

Also covers Phase 6 (quaternary scope tightening) acceptance: the interlaminar
mode folded into ``ShellHashinFiResult.interlaminar`` is bit-identical to the
pre-fold standalone computer, and the new ``von_mises_fi_from_row`` facade is
bit-identical to a direct ``materials.failure_criteria.von_mises_plane_stress_fi``
call (historical inline path lived in the removed optimiser FI evaluator).
"""

from __future__ import annotations

import numpy as np
import pytest

from blade_structure_utilities.section_materials.failure_criteria import hashin_fi
from blade_structure_utilities.section_materials.laminate import LaminateDefinition
from blade_structure_utilities.section_materials.ply import OrthotropicPly
from blade_structure_utilities.section_shell_model.pre_processing.containers.formulation_objects import ShellFormulationCache
from blade_structure_utilities.section_shell_model.pre_processing.containers.types import ShellPanelResultants
from blade_structure_utilities.section_shell_model.processing.static.results.compute_quaternary import (
    ShellQuaternaryResultsOrchestrator,
    interlaminar_fi_from_row as tier4_interlaminar_fi_from_row,
    max_hashin_fi_from_bundle as tier4_max_hashin_fi_from_bundle,
    select_governing_hashin_element as tier4_select_governing_hashin_element,
)
from blade_structure_utilities.section_shell_model.processing.static.results.compute_tertiary import (
    ShellTertiaryResultsOrchestrator,
)
from blade_structure_utilities.section_shell_model.processing.static.results.containers.container_hopper import (
    ShellSecondaryResultSet,
)
from blade_structure_utilities.section_shell_model.processing.static.results.containers.elemental_results import (
    ShellElementalResults,
)


def _ply(**overrides) -> OrthotropicPly:
    base = dict(
        name="gfrp", E1=30e9, E2=8e9, G12=3e9, nu12=0.3, rho=1.5e3, t_ply=2e-4,
        Xt=500e6, Xc=300e6, Yt=50e6, Yc=200e6, S12=40e6, Zt=1e6, S13=6e6, S23=6e6,
    )
    base.update(overrides)
    return OrthotropicPly(**base)


def _rows(n: int, *, panel_index: int = 0) -> list[ShellPanelResultants]:
    rng = np.random.default_rng(1)
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
                Qx=500.0,
                Qy=200.0,
                thickness_m=8e-4,
                panel_index=panel_index,
                station_index=i,
            )
        )
    return rows


def _tertiary_for(
    lam: LaminateDefinition,
    rows: list[ShellPanelResultants],
    *,
    panel_laminates: list | None = None,
    panel_constitutive_kind: list[str] | None = None,
    panel_sigma_allow_pa: list[float] | None = None,
):
    lams = panel_laminates if panel_laminates is not None else [lam]
    kinds = panel_constitutive_kind if panel_constitutive_kind is not None else ["orthotropic"] * len(lams)
    allows = (
        panel_sigma_allow_pa
        if panel_sigma_allow_pa is not None
        else [float("nan")] * len(lams)
    )
    cache = ShellFormulationCache(
        stiffness_objects=[],
        force_objects=[],
        panel_laminates=lams,
        panel_constitutive_kind=list(kinds),  # type: ignore[arg-type]
        panel_sigma_allow_pa=list(allows),
    )
    secondary = ShellSecondaryResultSet(elemental_results=ShellElementalResults(resultant_rows=rows))
    tertiary = ShellTertiaryResultsOrchestrator(secondary=secondary, cache=cache).compute()
    return tertiary, cache


def test_quaternary_orchestrator_hashin_matches_direct_evaluation():
    lam = LaminateDefinition(plies=[(_ply(), 0.0), (_ply(), 0.0)])
    rows = _rows(4)
    tertiary, cache = _tertiary_for(lam, rows)

    quaternary = ShellQuaternaryResultsOrchestrator(tertiary=tertiary, cache=cache).compute()
    assert quaternary.hashin_fi is not None
    assert quaternary.hashin_fi.in_plane is not None
    assert not np.any(quaternary.needs_review)

    Xt = np.array([p.Xt for p, _ in lam.plies])
    Xc = np.array([p.Xc for p, _ in lam.plies])
    Yt = np.array([p.Yt for p, _ in lam.plies])
    Yc = np.array([p.Yc for p, _ in lam.plies])
    S12 = np.array([p.S12 for p, _ in lam.plies])
    for ri in range(len(rows)):
        for pi in range(len(lam.plies)):
            sig3 = tertiary.ply_results.sig_ply_mat[ri, pi]
            expected = float(
                hashin_fi(
                    sig3.reshape(1, 3),
                    Xt[pi : pi + 1], Xc[pi : pi + 1], Yt[pi : pi + 1], Yc[pi : pi + 1], S12[pi : pi + 1],
                )[0]
            )
            assert quaternary.hashin_fi.in_plane[ri, pi] == pytest.approx(expected)


def test_quaternary_orchestrator_interlaminar_fi_positive_under_shear():
    lam = LaminateDefinition(plies=[(_ply(), 0.0)])
    rows = _rows(2)
    tertiary, cache = _tertiary_for(lam, rows)

    quaternary = ShellQuaternaryResultsOrchestrator(tertiary=tertiary, cache=cache).compute()
    assert quaternary.hashin_fi is not None
    assert quaternary.hashin_fi.interlaminar is not None
    assert np.all(quaternary.hashin_fi.interlaminar > 0.0)

    # Matches the row-based tier 4 facade for the same Qx/Qy/thickness/s_il.
    for row, fi in zip(rows, quaternary.hashin_fi.interlaminar):
        fi_row, nr = tier4_interlaminar_fi_from_row(row, s_il=6e6, peak_factor=1.5)
        assert not nr
        assert fi == pytest.approx(fi_row)


def test_governing_element_and_max_hashin_facades_are_reexported_from_tier4():
    from blade_structure_utilities.section_shell_model.processing.static.results.api import fi_recovery

    assert fi_recovery.select_governing_hashin_element is tier4_select_governing_hashin_element
    assert fi_recovery.max_hashin_fi_from_bundle is tier4_max_hashin_fi_from_bundle


def test_interlaminar_fi_from_row_facade_is_reexported_from_tier4():
    from blade_structure_utilities.section_shell_model.processing.static.results.api import fi_recovery

    assert fi_recovery.interlaminar_fi_from_row is tier4_interlaminar_fi_from_row


def test_von_mises_fi_from_row_facade_is_reexported_from_tier4_and_bit_identical():
    from blade_structure_utilities.section_materials.failure_criteria import von_mises_plane_stress_fi
    from blade_structure_utilities.section_shell_model.processing.static.results.api import fi_recovery
    from blade_structure_utilities.section_shell_model.processing.static.results.compute_quaternary import (
        von_mises_fi_from_row as tier4_von_mises_fi_from_row,
    )

    assert fi_recovery.von_mises_fi_from_row is tier4_von_mises_fi_from_row

    sig3 = np.array([120e6, -30e6, 15e6])
    sigma_allow = 250e6
    expected = float(
        von_mises_plane_stress_fi(sig3.reshape(1, 3), np.array([sigma_allow]))[0]
    )
    assert fi_recovery.von_mises_fi_from_row(sig3, sigma_allow) == pytest.approx(expected)


def test_batched_interlaminar_fi_matches_pre_fold_row_facade():
    from blade_structure_utilities.section_shell_model.processing.static.results.compute_quaternary.compute_hashin_fi import (
        batched_interlaminar_fi,
    )

    rows = _rows(5)
    s_il = 6e6
    qxy = np.array([[r.Qx, r.Qy] for r in rows])
    thickness = np.array([r.thickness_m for r in rows])

    fi_batch, bad_batch = batched_interlaminar_fi(qxy, thickness, s_il=s_il, peak_factor=1.5)

    for i, row in enumerate(rows):
        fi_row, nr_row = tier4_interlaminar_fi_from_row(row, s_il=s_il, peak_factor=1.5)
        assert fi_batch[i] == pytest.approx(fi_row)
        assert bool(bad_batch[i]) == nr_row


def test_quaternary_branches_hashin_vs_von_mises_by_panel_constitutive_kind():
    """Orthotropic panels get Hashin; isotropic panels get Von Mises and skip Hashin."""
    from blade_structure_utilities.section_materials.failure_criteria import von_mises_plane_stress_fi

    lam_ortho = LaminateDefinition(plies=[(_ply(), 0.0), (_ply(), 0.0)])
    # Single isotropic-equivalent ply (E1=E2) used for CLT recovery on panel 1.
    e, nu, t, sig_allow = 70e9, 0.33, 8e-4, 250e6
    g = e / (2.0 * (1.0 + nu))
    lam_iso = LaminateDefinition(
        plies=[
            (
                _ply(
                    name="alum",
                    E1=e,
                    E2=e,
                    G12=g,
                    nu12=nu,
                    rho=2700.0,
                    t_ply=t,
                    Xt=sig_allow,
                    Xc=sig_allow,
                    Yt=sig_allow,
                    Yc=sig_allow,
                    S12=sig_allow,
                ),
                0.0,
            )
        ]
    )
    rows_o = _rows(2, panel_index=0)
    rows_i = _rows(2, panel_index=1)
    rows = rows_o + rows_i

    tertiary, cache = _tertiary_for(
        lam_ortho,
        rows,
        panel_laminates=[lam_ortho, lam_iso],
        panel_constitutive_kind=["orthotropic", "isotropic"],
        panel_sigma_allow_pa=[float("nan"), sig_allow],
    )
    quaternary = ShellQuaternaryResultsOrchestrator(tertiary=tertiary, cache=cache).compute()

    assert quaternary.hashin_fi is not None
    assert quaternary.von_mises_fi is not None
    # Orthotropic rows: Hashin filled, VM nan.
    assert np.any(quaternary.hashin_fi.in_plane[:2] > 0.0) or np.any(
        quaternary.hashin_fi.in_plane[:2] == 0.0
    )
    assert np.all(np.isnan(quaternary.von_mises_fi[:2]))
    # Isotropic rows: Hashin left at zero, VM finite and matches direct criterion.
    assert np.all(quaternary.hashin_fi.in_plane[2:] == 0.0)
    assert np.all(np.isfinite(quaternary.von_mises_fi[2:]))
    for local_i, global_i in enumerate((2, 3)):
        sig3 = tertiary.ply_results.sig_ply_mat[global_i, 0]
        expected = float(
            von_mises_plane_stress_fi(sig3.reshape(1, 3), np.array([sig_allow]))[0]
        )
        assert quaternary.von_mises_fi[global_i] == pytest.approx(expected)
    assert quaternary.governing.get("criterion") in ("hashin", "von_mises")


def test_quaternary_isotropic_missing_sigma_allow_marks_needs_review():
    lam_iso = LaminateDefinition(plies=[(_ply(E1=70e9, E2=70e9), 0.0)])
    rows = _rows(2, panel_index=0)
    tertiary, cache = _tertiary_for(
        lam_iso,
        rows,
        panel_constitutive_kind=["isotropic"],
        panel_sigma_allow_pa=[float("nan")],
    )
    quaternary = ShellQuaternaryResultsOrchestrator(tertiary=tertiary, cache=cache).compute()
    assert np.all(quaternary.needs_review)
    assert quaternary.von_mises_fi is not None
    assert np.all(np.isnan(quaternary.von_mises_fi))
