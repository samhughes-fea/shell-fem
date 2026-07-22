"""Element-agnostic kernel dispatch, contract aliases, and provenance predicates."""

from __future__ import annotations

import numpy as np
import pytest

from blade_precompute.section_shell_model.pre_processing.containers.types import (
    FieldProvenance,
    ProvenanceKind,
    SectionShellRecoveryBundle,
    ShellPanelSolveResult,
    ShellTheory,
    provenance_is_placeholder,
    provenance_is_trusted,
)
from blade_precompute.section_shell_model.pre_processing.element_library.kernel import (
    MITC4_KERNEL,
    resolve_kernel,
)
from blade_precompute.section_shell_model.pre_processing.element_library.mitc4.utilities.mitc4_solve_orchestrator import (
    collect_panel_abd_stack,
)


def test_resolve_kernel_fsdt_is_mitc4():
    k = resolve_kernel(ShellTheory.FSDT)
    assert k is MITC4_KERNEL
    assert k.ndof == 6


def test_provenance_predicates():
    assert provenance_is_trusted(ProvenanceKind.MITC4)
    assert provenance_is_trusted(ProvenanceKind.CALCULIX_CCX)
    assert not provenance_is_trusted(ProvenanceKind.PLACEHOLDER)
    assert not provenance_is_trusted(ProvenanceKind.RESERVED)
    assert provenance_is_placeholder(ProvenanceKind.PLACEHOLDER)
    fp = FieldProvenance(ProvenanceKind.MITC4, "test")
    assert fp.is_trusted
    assert not fp.is_placeholder


def test_collect_panel_abd_stack_is_callable():
    assert callable(collect_panel_abd_stack)


def test_section_shell_recovery_bundle_field_aliases():
    bundle = SectionShellRecoveryBundle(
        panels=[],
        booms=[],
        webs_geom=[],
        q_tot=[],
        sig_p=[],
        sig_b=[],
        q0=[],
        props=None,
        y_sc=0.0,
        z_sc=0.0,
        areas=[],
        I_omega=0.0,
        gamma_y=0.0,
        gamma_z=0.0,
        GA_y=0.0,
        GA_z=0.0,
        q_primary=[],
        q_warp=[],
        all_panel_results=[[None]],
        panel_labels=["skin"],
    )
    assert bundle.all_panel_results == [[None]]
    assert bundle.panel_labels == ["skin"]
    assert isinstance(bundle.shell_solve, ShellPanelSolveResult)


def test_kernel_recovery_matches_legacy_mitc4_path():
    """Kernel recover_resultants / recover_shear match direct mitc4 helpers for one element."""
    from blade_precompute.section_shell_model.pre_processing.element_library.mitc4.element import (
        mitc4_resultants,
        mitc4_shear_resultants,
    )

    ndof = 6
    d_elem = np.array([1e-4, 0.0, 1e-5, 0.0, 0.0, 0.0] * 4)
    abd = np.eye(6) * 1e6
    abd[0, 0] = abd[1, 1] = abd[2, 2] = 1e8
    abd[3, 3] = abd[4, 4] = abd[5, 5] = 1e6
    L_s = 0.1
    direct = mitc4_resultants(d_elem, L_s, 1.0, abd, ndof=ndof)
    via_kernel = MITC4_KERNEL.recover_resultants(d_elem, L_s, 1.0, abd)
    for key in ("Nx", "Ny", "Nxy", "Mx", "My", "Mxy"):
        assert direct[key] == pytest.approx(via_kernel[key], rel=0, abs=1e-12)
    qx_d, qy_d = mitc4_shear_resultants(d_elem, L_s, 1.0, 5e9, 0.002, ndof=ndof)
    qx_k, qy_k = MITC4_KERNEL.recover_shear(
        d_elem, L_s, 1.0, G_eff=5e9, thickness=0.002,
    )
    assert qx_k == pytest.approx(qx_d, rel=0, abs=1e-9)
    assert qy_k == pytest.approx(qy_d, rel=0, abs=1e-9)

