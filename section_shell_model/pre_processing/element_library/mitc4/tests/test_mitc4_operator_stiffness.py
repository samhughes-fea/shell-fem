"""MITC4 operator-template parity checks."""

from __future__ import annotations

import numpy as np

from blade_structure_utilities.section_shell_model.pre_processing.element_library.mitc4.element import (
    _b_bending,
    _b_drilling,
    _b_membrane,
    _b_shear_mitc,
    mitc4_stiffness,
)


def _reference_mitc4_stiffness(
    L_s: float,
    L_x: float,
    ABD: np.ndarray,
    thickness: float,
    *,
    ks: float = 5.0 / 6.0,
    G_eff: float | None = None,
    include_drilling: bool = True,
    alpha_drill: float = 0.25,
) -> np.ndarray:
    a_mat = ABD[:3, :3]
    b_mat = ABD[:3, 3:]
    d_mat = ABD[3:, 3:]
    if G_eff is None:
        g_eff = float(a_mat[2, 2]) / max(thickness, 1e-30)
    else:
        g_eff = float(G_eff)
    h_s = ks * g_eff * thickness
    jdet = (L_s / 2.0) * (L_x / 2.0)
    ndof = 6 if include_drilling else 5
    n_elem = 4 * ndof
    k = np.zeros((n_elem, n_elem))
    gp3 = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
    gw3 = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    for xi, wi in zip(gp3, gw3):
        for eta, wj in zip(gp3, gw3):
            bm = _b_membrane(xi, eta, L_s, L_x, ndof)
            bb = _b_bending(xi, eta, L_s, L_x, ndof)
            w = wi * wj * jdet
            k += w * (
                bm.T @ a_mat @ bm
                + bm.T @ b_mat @ bb
                + bb.T @ b_mat.T @ bm
                + bb.T @ d_mat @ bb
            )
    gp2 = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
    gw2 = np.array([1.0, 1.0])
    h2 = np.array([[h_s, 0.0], [0.0, h_s]])
    for xi, wi in zip(gp2, gw2):
        for eta, wj in zip(gp2, gw2):
            bs = _b_shear_mitc(xi, eta, L_s, L_x, ndof)
            w = wi * wj * jdet
            k += w * bs.T @ h2 @ bs
    if include_drilling:
        c_drill = alpha_drill * g_eff * thickness
        for xi, wi in zip(gp2, gw2):
            for eta, wj in zip(gp2, gw2):
                bd = _b_drilling(xi, eta, L_s, L_x)
                w = wi * wj * jdet
                k += w * c_drill * (bd.T @ bd)
    return k


def test_mitc4_stiffness_einsum_matches_legacy_gauss_loops() -> None:
    from blade_structure_utilities.section_shell_model.pre_processing.element_library.mitc4 import element as mitc4_element

    mitc4_element._KE_TEMPLATE_CACHE.clear()
    abd = np.eye(6) * 1e6
    abd[2, 2] = 5e5
    abd[5, 5] = 2e5
    legacy = _reference_mitc4_stiffness(0.4, 1.0, abd, 0.002, G_eff=3e9)
    modern = mitc4_stiffness(0.4, 1.0, abd, 0.002, G_eff=3e9)
    assert np.allclose(legacy, modern, rtol=0.0, atol=1e-9)
