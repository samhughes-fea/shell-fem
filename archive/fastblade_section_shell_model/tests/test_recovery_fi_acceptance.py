"""Provenance-gated FI and recovery acceptance tests."""

from __future__ import annotations

import numpy as np
import pytest

from blade_precompute.section_shell_model.api.fi_recovery import (
    clpt_inputs_trusted,
    hashin_fi_per_ply_from_row,
    section_integrals_from_bundle,
)

from blade_precompute.section_shell_model.api.pipeline_b import build_pipeline_b_section
from blade_precompute.section_shell_model.api.periodic import (
    run_section_periodic_bc_mitc4,
)
from blade_precompute.section_shell_model.pre_processing.containers.types import (
    FieldProvenance,
    ProvenanceKind,
    ShellPanelResultants,
)
from blade_precompute.section_shell_model.engine.local_clpt_shell import solve_station_clpt_shell
from blade_precompute.materials.failure_criteria import hashin_fi


def _section():
    return build_pipeline_b_section("2412", [0.35], chord_m=1.0)


def test_provenance_gate_excludes_placeholder_m():
    row = ShellPanelResultants(
        Nx=1e3, Ny=0.0, Nxy=0.0, Mx=0.0, My=100.0, Mxy=0.0,
        provenance={
            "My": FieldProvenance(ProvenanceKind.PLACEHOLDER, "stub"),
        },
    )
    assert not clpt_inputs_trusted(row)
    fi, nr = hashin_fi_per_ply_from_row(row, _dummy_lam(), [])
    assert nr
    assert np.all(fi == 0.0)


def _dummy_lam():
    from blade_precompute.materials.laminate import LaminateDefinition
    from blade_precompute.materials.ply import OrthotropicPly

    p = OrthotropicPly(
        name="t", E1=30e9, E2=8e9, G12=3e9, nu12=0.3, rho=1.5e3, t_ply=2e-4,
        Xt=500e6, Xc=300e6, Yt=50e6, Yc=200e6, S12=40e6,
        Zt=1e6, S13=1e6, S23=1e6,
    )
    return LaminateDefinition(plies=[(p, 0.0)])


def test_recovery_fi_acceptance_my_load():
    pb = _section()
    load = dict(My=1.5e4)
    bm = run_section_periodic_bc_mitc4(pb, n_elements_per_panel=6, **load)

    ints = section_integrals_from_bundle(bm)
    assert abs(ints["Nx"]) < 1e9  # finite

    pm = (bm.all_panel_results or [[]])[0]
    assert pm
    for row in pm:
        if row is None:
            continue
        for f in ("Nx", "My", "Mz"):
            prov = row.provenance.get(f)
            if prov is not None:
                assert prov.is_trusted


def test_golden_clpt_element_hashin_crosscheck():
    """Known [N;M] -> CLPT -> materials.hashin_fi matches legacy envelope."""
    from blade_precompute.section_optimisation.engine.shell_fi_eval import LaminateDefinitionMitc4SkinAdapter

    lam = _dummy_lam()
    ad = LaminateDefinitionMitc4SkinAdapter(lam)
    plies = ad.build_plies()
    p0, _ = lam.plies[0]
    row = ShellPanelResultants(
        Nx=8e3, Ny=200.0, Nxy=50.0,
        Mx=5.0, My=120.0, Mxy=2.0,
    )
    legacy = solve_station_clpt_shell(
        row, plies,
        Xt=float(p0.Xt), Xc=float(p0.Xc), Yt=float(p0.Yt),
        Yc=float(p0.Yc), S12=float(p0.S12),
    )
    fi_new, nr = hashin_fi_per_ply_from_row(row, lam, plies)
    assert not nr
    assert fi_new.shape == legacy.fi.shape
    np.testing.assert_allclose(fi_new, legacy.fi, rtol=0.05, atol=0.02)

    sig0 = np.asarray(legacy.sig_lam_mid[0], dtype=float).ravel()[:3]
    fi_direct = float(
        hashin_fi(
            sig0.reshape(1, 3),
            np.array([p0.Xt]), np.array([p0.Xc]),
            np.array([p0.Yt]), np.array([p0.Yc]), np.array([p0.S12]),
        )[0]
    )
    assert fi_new[0] == pytest.approx(fi_direct, rel=0.02)
