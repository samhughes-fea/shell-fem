"""Unit tests for SFV → CCX nx/nxy panel load adapter."""

from __future__ import annotations

import numpy as np

from blade_structure_utilities.geometry_utilities.section_contract.beam_resultant_vector import (
    BeamResultantVector,
)
from blade_structure_utilities.load_utilities.section_loads import (
    ContourForceSample,
    SectionForceVector,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.calculix_ccx.sfv_panel_loads import (
    sfv_to_nx_nxy,
)


class _Panel:
    def __init__(self, nodes, s, label):
        self.nodes = np.asarray(nodes, dtype=float)
        self.s = np.asarray(s, dtype=float)
        self.label = label


def _sfv(samples: list[ContourForceSample]) -> SectionForceVector:
    r = BeamResultantVector.from_named(Vy=1.0)
    return SectionForceVector(samples=tuple(samples), resultants=r, reconstructed=r)


def test_sfv_to_nx_nxy_label_match_and_nearest():
    samples = [
        ContourForceSample(
            label="skin:a",
            kind="skin",
            y=0.0,
            z=0.0,
            t=0.01,
            f_x=1000.0,
            f_s=50.0,
            f_w=0.0,
        ),
        ContourForceSample(
            label="skin:a",
            kind="skin",
            y=1.0,
            z=0.0,
            t=0.01,
            f_x=2000.0,
            f_s=150.0,
            f_w=0.0,
        ),
        ContourForceSample(
            label="web:b",
            kind="web",
            y=1.0,
            z=0.5,
            t=0.01,
            f_x=0.0,
            f_s=200.0,
            f_w=0.0,
        ),
    ]
    panels = [
        _Panel([[0, 0], [1, 0]], [0.0, 1.0], "skin:a"),
        _Panel([[1, 0], [1, 1]], [0.0, 1.0], "web:b"),
    ]
    nx, nxy = sfv_to_nx_nxy(_sfv(samples), panels)
    assert nx[0][0] == 1000.0
    assert nxy[0][0] == 50.0
    assert nx[0][1] == 2000.0
    assert nxy[0][1] == 150.0
    assert nxy[1][0] == 200.0
    assert abs(nxy[1][1] - 200.0) < 1e-12


def test_sfv_to_nx_nxy_empty_samples_zeros():
    panels = [_Panel([[0, 0], [1, 0]], [0.0, 1.0], "skin:a")]
    nx, nxy = sfv_to_nx_nxy(_sfv([]), panels)
    assert np.allclose(nx[0], 0.0)
    assert np.allclose(nxy[0], 0.0)
