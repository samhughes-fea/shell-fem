"""sig_ply_mat -> unit_histories reshape (FE-free)."""

from __future__ import annotations

import numpy as np
import pytest

from blade_structure_utilities.fem_utilities.section_shell_fea.interface.shell_unit_histories import (
    FACE_MID,
    composite_unit_histories_from_sig_ply_mat,
    isotropic_unit_histories_from_sig_ply_mat,
    unit_histories_from_sig_ply_mat,
)


def test_isotropic_three_faces():
    sig = np.zeros((2, 1, 3, 3), dtype=np.float64)
    sig[0, 0, FACE_MID] = (10.0, 1.0, 0.5)
    hist = isotropic_unit_histories_from_sig_ply_mat(sig, element_index=0)
    assert set(hist) == {"OUTER", "MID", "INNER"}
    assert float(hist["MID"]["s11"][0]) == pytest.approx(10.0)


def test_composite_mid_plies():
    sig = np.zeros((1, 2, 3, 3), dtype=np.float64)
    sig[0, 0, FACE_MID] = (100.0, 5.0, 2.0)
    sig[0, 1, FACE_MID] = (80.0, -3.0, 1.0)
    hist = composite_unit_histories_from_sig_ply_mat(sig)
    assert "ply_0" in hist and "ply_1" in hist
    assert float(hist["ply_0"]["sigma_1"][0]) == pytest.approx(100.0)


def test_unit_histories_keyed_by_subcomponent_feeds_shell_recovery():
    pytest.importorskip("blade_structure_fatigue")
    from blade_structure_fatigue.utilities.stress_recovery import (
        ShellElementStressRecovery,
    )

    sig = np.zeros((1, 1, 3, 3), dtype=np.float64)
    sig[0, 0, FACE_MID] = (8.0, 0.0, 0.0)
    unit = unit_histories_from_sig_ply_mat(sig, kind="isotropic", subcomponent="skin")
    recovery = ShellElementStressRecovery(
        assignment={"skin": type("M", (), {"E": 70e9})()},
        unit_histories=unit,
    )
    field = recovery.build_field(subcomponent="skin")
    assert field.kind == "isotropic"
    s1, _, _ = field.face_tensor("MID")
    assert float(s1[0]) == pytest.approx(8.0)
