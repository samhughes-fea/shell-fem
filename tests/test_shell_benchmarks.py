"""The shell obstacle course (MacNeal & Harder 1985, Belytschko et al. 1985) on every shell element.

Tolerances are the accuracy each element reaches on the given mesh; the full tables
and convergence plots are in examples/27_shell_obstacle_course.
"""

import warnings

import numpy as np
import pytest

from shell_fem import FSDTModel, Quad4ShellElement, Quad8ShellElement
from shell_fem.verification.shell_benchmarks import (
    PROBLEMS,
    isotropic_laminate,
    pinched_cylinder,
    pinched_hemisphere,
    scordelis_lo,
    twisted_beam,
)

WARPED_Q4 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.15], [1.0, 1.0, 0.0], [0.0, 1.0, 0.15]])
WARPED_Q8 = np.vstack([WARPED_Q4, 0.5 * (WARPED_Q4 + np.roll(WARPED_Q4, -1, axis=0))])


def rigid_body_modes(X):
    modes = [np.concatenate([np.r_[t, np.zeros(3)] for _ in X]) for t in np.eye(3)]
    modes += [np.concatenate([np.r_[np.cross(r, x), r] for x in X]) for r in np.eye(3)]
    return modes


def rigid_body_residual(element_class, X, **options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        element = element_class(1, tuple(range(1, len(X) + 1)), X, isotropic_laminate(1e7, 0.3, 0.05), FSDTModel(),
                                **options)
    K = element.stiffness_matrix()
    return max(np.linalg.norm(K @ m) / (np.linalg.norm(K) * np.linalg.norm(m)) for m in rigid_body_modes(X))


@pytest.mark.parametrize("element_class, X", [(Quad4ShellElement, WARPED_Q4), (Quad8ShellElement, WARPED_Q8)])
def test_warped_flat_elements_move_rigidly_without_strain(element_class, X):
    assert rigid_body_residual(element_class, X) < 1e-12
    # without the warping correction, a rigid rotation strains the projected element
    assert rigid_body_residual(element_class, X, warping_correction=False) > 1e-3


def test_warping_correction_leaves_flat_elements_unchanged(cfrp):
    from shell_fem import Laminate

    lam = Laminate.from_angles(cfrp, [0, 45, -45, 90], 0.25e-3)
    X = np.array([[0.0, 0.0, 0.0], [0.11, 0.01, 0.0], [0.13, 0.12, 0.0], [-0.01, 0.09, 0.0]])
    on = Quad4ShellElement(1, (1, 2, 3, 4), X, lam, FSDTModel()).stiffness_matrix()
    off = Quad4ShellElement(1, (1, 2, 3, 4), X, lam, FSDTModel(), warping_correction=False).stiffness_matrix()
    assert np.array_equal(on, off)


@pytest.mark.parametrize("problem", sorted(PROBLEMS))
def test_curved_quad8_passes_every_problem_on_an_8_by_8_mesh(problem):
    assert PROBLEMS[problem]("quad8_curved", 8).ratio == pytest.approx(1.0, abs=0.025)


@pytest.mark.parametrize("problem, n, tol", [
    ("scordelis_lo", 16, 0.02),
    ("pinched_cylinder", 16, 0.02),
    ("twisted_beam_in_plane", 8, 0.02),
    ("twisted_beam_out_of_plane", 8, 0.02),
])
def test_tri6(problem, n, tol):
    assert PROBLEMS[problem]("tri6", n).ratio == pytest.approx(1.0, abs=tol)


def test_flat_quads_on_singly_curved_shells():
    assert scordelis_lo("quad4", 16).ratio == pytest.approx(1.0, abs=0.01)
    assert scordelis_lo("quad8", 16).ratio == pytest.approx(1.0, abs=0.005)
    assert pinched_cylinder("quad8", 16).ratio == pytest.approx(1.0, abs=0.005)
    assert pinched_cylinder("quad4", 16).ratio == pytest.approx(1.0, abs=0.08)  # converges slowly
    assert pinched_hemisphere("quad4", 8).ratio == pytest.approx(1.0, abs=0.015)


def test_flat_quads_on_the_twisted_beam_need_drilling_stiffness():
    # Each element of the twisted strip lies at an angle to its neighbours, so one
    # element's bending rotation is the next one's drilling rotation. With the default
    # drilling stiffness (1e-3 A66) the strip is about 30 % too flexible; at 0.1 A66
    # it is within 1 %.
    default = twisted_beam("quad4", 12)
    stiffer = twisted_beam("quad4", 12, drilling_factor=0.1)
    assert default.ratio == pytest.approx(1.3, abs=0.05)
    assert stiffer.ratio == pytest.approx(1.0, abs=0.01)
    assert twisted_beam("quad4", 12, "out_of_plane", drilling_factor=0.1).ratio == pytest.approx(1.0, abs=0.01)


def test_curved_quad8_converges_towards_the_reference():
    errors = [abs(pinched_cylinder("quad8_curved", n).ratio - 1.0) for n in (4, 8, 16)]
    assert errors[0] > errors[1] > errors[2]
