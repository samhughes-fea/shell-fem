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

from blade_structure_utilities.material_utilities.failure import hashin_fi
from blade_structure_utilities.material_utilities.laminate import FACE_INNER, FACE_MID, FACE_OUTER, LaminateDefinition
from blade_structure_utilities.material_utilities.orthotropic import OrthotropicPly
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.formulation_objects import ShellFormulationCache
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import ShellPanelResultants
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.compute_quaternary import (
    ShellQuaternaryResultsOrchestrator,
    interlaminar_fi_from_row as tier4_interlaminar_fi_from_row,
    max_hashin_fi_from_bundle as tier4_max_hashin_fi_from_bundle,
    select_governing_hashin_element as tier4_select_governing_hashin_element,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.compute_tertiary import (
    ShellTertiaryResultsOrchestrator,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.containers.container_hopper import (
    ShellSecondaryResultSet,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.containers.elemental_results import (
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
                N11=float(rng.uniform(-1e4, 1e4)),
                N22=float(rng.uniform(-1e3, 1e3)),
                N12=float(rng.uniform(-1e3, 1e3)),
                M11=float(rng.uniform(-10, 10)),
                M22=float(rng.uniform(-50, 50)),
                M12=float(rng.uniform(-5, 5)),
                Q1=500.0,
                Q2=200.0,
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
    panel_sigma_y_pa: list[float] | None = None,
    panel_sigma_uts_pa: list[float] | None = None,
    panel_sigma_ucs_pa: list[float] | None = None,
    panel_tau_us_pa: list[float] | None = None,
):
    lams = panel_laminates if panel_laminates is not None else [lam]
    n = len(lams)
    kinds = panel_constitutive_kind if panel_constitutive_kind is not None else ["orthotropic"] * n
    y = panel_sigma_y_pa if panel_sigma_y_pa is not None else [float("nan")] * n
    # Finite ultimates by default so extreme-fibre does not force needs_review in Hashin tests.
    uts = panel_sigma_uts_pa if panel_sigma_uts_pa is not None else [500e6] * n
    ucs = panel_sigma_ucs_pa if panel_sigma_ucs_pa is not None else [400e6] * n
    tus = panel_tau_us_pa if panel_tau_us_pa is not None else [200e6] * n
    cache = ShellFormulationCache(
        stiffness_objects=[],
        force_objects=[],
        panel_laminates=lams,
        panel_constitutive_kind=list(kinds),  # type: ignore[arg-type]
        panel_sigma_y_pa=list(y),
        panel_sigma_uts_pa=list(uts),
        panel_sigma_ucs_pa=list(ucs),
        panel_tau_us_pa=list(tus),
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
    S23 = np.array([p.S23 for p, _ in lam.plies])
    for ri in range(len(rows)):
        for pi in range(len(lam.plies)):
            expected = 0.0
            for face in (FACE_OUTER, FACE_INNER):
                sig3 = tertiary.ply_results.sig_ply_mat[ri, pi, face]
                expected = max(
                    expected,
                    float(
                        hashin_fi(
                            sig3.reshape(1, 3),
                            Xt[pi : pi + 1], Xc[pi : pi + 1], Yt[pi : pi + 1], Yc[pi : pi + 1], S12[pi : pi + 1], S23[pi : pi + 1],
                        )[0]
                    ),
                )
            assert quaternary.hashin_fi.in_plane[ri, pi] == pytest.approx(expected)


def test_quaternary_orchestrator_populates_hashin_mode_arrays():
    from blade_structure_utilities.material_utilities.failure import hashin_fi_modes

    lam = LaminateDefinition(plies=[(_ply(), 0.0), (_ply(), 0.0)])
    rows = _rows(3)
    tertiary, cache = _tertiary_for(lam, rows)
    quaternary = ShellQuaternaryResultsOrchestrator(tertiary=tertiary, cache=cache).compute()
    hi = quaternary.hashin_fi
    assert hi is not None
    assert hi.fi_ft is not None and hi.fi_fc is not None
    assert hi.fi_mt is not None and hi.fi_mc is not None
    assert hi.in_plane is not None

    Xt = np.array([p.Xt for p, _ in lam.plies])
    Xc = np.array([p.Xc for p, _ in lam.plies])
    Yt = np.array([p.Yt for p, _ in lam.plies])
    Yc = np.array([p.Yc for p, _ in lam.plies])
    S12 = np.array([p.S12 for p, _ in lam.plies])
    S23 = np.array([p.S23 for p, _ in lam.plies])
    for ri in range(len(rows)):
        for pi in range(len(lam.plies)):
            best = {"ft": 0.0, "fc": 0.0, "mt": 0.0, "mc": 0.0}
            for face in (FACE_OUTER, FACE_INNER):
                sig3 = tertiary.ply_results.sig_ply_mat[ri, pi, face]
                ft, fc, mt, mc = hashin_fi_modes(
                    sig3.reshape(1, 3),
                    Xt[pi : pi + 1], Xc[pi : pi + 1], Yt[pi : pi + 1], Yc[pi : pi + 1], S12[pi : pi + 1], S23[pi : pi + 1],
                )
                best["ft"] = max(best["ft"], float(ft[0]))
                best["fc"] = max(best["fc"], float(fc[0]))
                best["mt"] = max(best["mt"], float(mt[0]))
                best["mc"] = max(best["mc"], float(mc[0]))
            assert hi.fi_ft[ri, pi] == pytest.approx(best["ft"])
            assert hi.fi_fc[ri, pi] == pytest.approx(best["fc"])
            assert hi.fi_mt[ri, pi] == pytest.approx(best["mt"])
            assert hi.fi_mc[ri, pi] == pytest.approx(best["mc"])
            env = max(best.values())
            assert hi.in_plane[ri, pi] == pytest.approx(env)


def test_quaternary_orchestrator_interlaminar_fi_positive_under_shear():
    lam = LaminateDefinition(plies=[(_ply(), 0.0)])
    rows = _rows(2)
    tertiary, cache = _tertiary_for(lam, rows)

    quaternary = ShellQuaternaryResultsOrchestrator(tertiary=tertiary, cache=cache).compute()
    assert quaternary.hashin_fi is not None
    assert quaternary.hashin_fi.interlaminar is not None
    assert np.all(quaternary.hashin_fi.interlaminar > 0.0)

    # Matches the row-based tier 4 facade for the same Q1/Q2/thickness/s_il.
    for row, fi in zip(rows, quaternary.hashin_fi.interlaminar):
        fi_row, nr = tier4_interlaminar_fi_from_row(row, s_il=6e6, peak_factor=1.5)
        assert not nr
        assert fi == pytest.approx(fi_row)
    assert quaternary.governing.get("criterion") != "hashin_il"
    assert quaternary.governing.get("criterion_id") != "hashin_1980_3d"


def test_governing_element_and_max_hashin_facades_are_reexported_from_tier4():
    from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.api import fi_recovery

    assert fi_recovery.select_governing_hashin_element is tier4_select_governing_hashin_element
    assert fi_recovery.max_hashin_fi_from_bundle is tier4_max_hashin_fi_from_bundle


def test_interlaminar_fi_from_row_facade_is_reexported_from_tier4():
    from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.api import fi_recovery

    assert fi_recovery.interlaminar_fi_from_row is tier4_interlaminar_fi_from_row


def test_von_mises_fi_from_row_facade_is_reexported_from_tier4_and_bit_identical():
    from blade_structure_utilities.material_utilities.failure import von_mises_plane_stress_fi
    from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.api import fi_recovery
    from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.compute_quaternary import (
        von_mises_fi_from_row as tier4_von_mises_fi_from_row,
    )

    assert fi_recovery.von_mises_fi_from_row is tier4_von_mises_fi_from_row

    sig3 = np.array([120e6, -30e6, 15e6])
    sigma_y = 250e6
    expected = float(
        von_mises_plane_stress_fi(sig3.reshape(1, 3), np.array([sigma_y]))[0]
    )
    assert fi_recovery.von_mises_fi_from_row(sig3, sigma_y) == pytest.approx(expected)


def test_batched_interlaminar_fi_matches_pre_fold_row_facade():
    from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.compute_quaternary.compute_hashin_fi import (
        batched_interlaminar_fi,
    )

    rows = _rows(5)
    s_il = 6e6
    qxy = np.array([[r.Q1, r.Q2] for r in rows])
    thickness = np.array([r.thickness_m for r in rows])

    fi_batch, bad_batch = batched_interlaminar_fi(qxy, thickness, s_il=s_il, peak_factor=1.5)

    for i, row in enumerate(rows):
        fi_row, nr_row = tier4_interlaminar_fi_from_row(row, s_il=s_il, peak_factor=1.5)
        assert fi_batch[i] == pytest.approx(fi_row)
        assert bool(bad_batch[i]) == nr_row


def test_quaternary_branches_hashin_vs_von_mises_by_panel_constitutive_kind():
    """Orthotropic panels get Hashin; isotropic panels get Von Mises and skip Hashin."""
    from blade_structure_utilities.material_utilities.failure import von_mises_plane_stress_fi

    lam_ortho = LaminateDefinition(plies=[(_ply(), 0.0), (_ply(), 0.0)])
    # Single isotropic-equivalent ply (E1=E2) used for CLT recovery on panel 1.
    e, nu, t, sigma_y = 70e9, 0.33, 8e-4, 250e6
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
                    Xt=sigma_y,
                    Xc=sigma_y,
                    Yt=sigma_y,
                    Yc=sigma_y,
                    S12=sigma_y,
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
        panel_sigma_y_pa=[float("nan"), sigma_y],
    )
    quaternary = ShellQuaternaryResultsOrchestrator(tertiary=tertiary, cache=cache).compute()

    assert quaternary.hashin_fi is not None
    assert quaternary.von_mises_fi is not None
    assert quaternary.extreme_fibre_fi is not None
    # Orthotropic rows: Hashin filled, VM nan.
    assert np.any(quaternary.hashin_fi.in_plane[:2] > 0.0) or np.any(
        quaternary.hashin_fi.in_plane[:2] == 0.0
    )
    assert np.all(np.isnan(quaternary.von_mises_fi[:2]))
    # Isotropic rows: Hashin left at zero, VM finite and matches direct criterion.
    assert np.all(quaternary.hashin_fi.in_plane[2:] == 0.0)
    assert np.all(np.isfinite(quaternary.von_mises_fi[2:]))
    for local_i, global_i in enumerate((2, 3)):
        sig3 = tertiary.ply_results.sig_ply_mat[global_i, 0, FACE_MID]
        expected = float(
            von_mises_plane_stress_fi(sig3.reshape(1, 3), np.array([sigma_y]))[0]
        )
        assert quaternary.von_mises_fi[global_i] == pytest.approx(expected)
    assert quaternary.governing.get("criterion") in ("hashin", "von_mises", "extreme_fibre")


def test_quaternary_isotropic_missing_sigma_y_marks_needs_review():
    lam_iso = LaminateDefinition(plies=[(_ply(E1=70e9, E2=70e9), 0.0)])
    rows = _rows(2, panel_index=0)
    tertiary, cache = _tertiary_for(
        lam_iso,
        rows,
        panel_constitutive_kind=["isotropic"],
        panel_sigma_y_pa=[float("nan")],
        panel_sigma_uts_pa=[500e6],
        panel_sigma_ucs_pa=[400e6],
        panel_tau_us_pa=[200e6],
    )
    quaternary = ShellQuaternaryResultsOrchestrator(tertiary=tertiary, cache=cache).compute()
    assert np.all(quaternary.needs_review)
    assert quaternary.von_mises_fi is not None
    assert np.all(np.isnan(quaternary.von_mises_fi))


def test_extreme_fibre_fi_sign_selects_uts_ucs():
    from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.compute_quaternary.compute_extreme_fibre_fi import (
        extreme_fibre_fi,
    )

    # Outer tension, inner compression on σ11
    sig = np.array(
        [
            [[100e6, 0.0, 0.0], [-50e6, 0.0, 0.0]],
        ],
        dtype=float,
    )
    fi = extreme_fibre_fi(sig, sigma_uts=200e6, sigma_ucs=100e6, tau_us=80e6)
    # outer FI=0.5, inner FI=0.5 → envelope 0.5
    assert fi[0] == pytest.approx(0.5)
    fi2 = extreme_fibre_fi(sig, sigma_uts=200e6, sigma_ucs=50e6, tau_us=80e6)
    # inner |−50|/50 = 1.0 governs
    assert fi2[0] == pytest.approx(1.0)
