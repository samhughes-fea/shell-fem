"""Geometric nonlinearity is scoped, not implemented.

Documents the gap for post-buckling of panels and webs and for large deflection:
linear statics and linear buckling exist; Newton, follower pressure and an
updated-geometry tangent do not.
"""

import inspect

import numpy as np
import pytest

from conftest import simply_supported
from shell_fem import (
    AssembleFollowerPressure,
    DOF,
    FEAMesh,
    FSDTModel,
    GeometricNonlinearityNotImplemented,
    GeometricNonlinearRunner,
    GeometricNonlinearityScope,
    Laminate,
    LinearBucklingRunner,
    LinearStaticRunner,
    LoadCase,
    NewtonEquilibrium,
    geometric_nonlinearity_scope,
    linear_buckling,
)
from shell_fem.pre_processing.element_library.shell import ShellElement


def _supported_plate(aluminium, n=4):
    a, h = 0.2, 0.002
    mesh = FEAMesh.rectangular_plate(a, a, n, n, Laminate.from_angles(aluminium, [0.0], h), FSDTModel())
    return mesh, simply_supported(mesh, a, a), a


def test_scope_lists_linear_operators_and_the_post_buckling_gap():
    scope = geometric_nonlinearity_scope()
    assert isinstance(scope, GeometricNonlinearityScope)
    joined_exists = " ".join(scope.exists)
    joined_missing = " ".join(scope.missing)
    joined_uses = " ".join(scope.intended_uses)
    assert "linear static" in joined_exists and "K u = F" in joined_exists
    assert "linear buckling" in joined_exists and "Kg" in joined_exists
    assert "membrane" in joined_exists
    assert "Cauchy" in joined_exists
    assert "dead pressure" in joined_exists
    assert "Newton" in joined_missing
    assert "follower pressure" in joined_missing
    assert "static tangent" in joined_missing
    assert "arc-length" in joined_missing
    assert "post-buckling" in joined_uses
    assert "large deflection" in joined_uses
    assert "Newton" not in joined_exists
    assert "follower" not in joined_exists


def test_named_stages_and_the_runner_raise_without_inventing_a_solver(aluminium):
    mesh, bcs, _ = _supported_plate(aluminium, n=2)
    load = LoadCase().add_pressure(1.0e3)
    with pytest.raises(GeometricNonlinearityNotImplemented, match="Newton"):
        NewtonEquilibrium().run()
    with pytest.raises(GeometricNonlinearityNotImplemented, match="Follower"):
        AssembleFollowerPressure().run()
    runner = GeometricNonlinearRunner(mesh, bcs, load)
    with pytest.raises(GeometricNonlinearityNotImplemented, match="not implemented"):
        runner.execute()
    with pytest.raises(RuntimeError, match="execute"):
        runner.get_results()


def test_linear_static_is_one_factorisation_on_the_undeformed_mesh(aluminium):
    mesh, bcs, _ = _supported_plate(aluminium, n=2)
    xyz0 = mesh.coordinates_array().copy()
    results = LinearStaticRunner(mesh, bcs, LoadCase().add_pressure(1.0e3)).execute().get_results()
    phases = results.diagnostics.phase_times
    assert "factorise" in phases and "solve" in phases
    assert "newton" not in phases and "residual" not in phases and "follower" not in phases
    assert np.array_equal(mesh.coordinates_array(), xyz0)


def test_pressure_is_dead_along_the_undeformed_normal(aluminium):
    mesh, _, _ = _supported_plate(aluminium, n=1)
    element = mesh.elements[0]
    params = list(inspect.signature(element.pressure_load_vector).parameters)
    assert params == ["pressure"]
    F = element.pressure_load_vector(2.0e3)
    assert np.sum(F[2::element.dofs_per_node]) == pytest.approx(2.0e3 * element.area)
    # A follower load would take a displacement; this vector does not.


def test_static_solve_does_not_assemble_geometric_stiffness(aluminium, monkeypatch):
    calls = []
    original = ShellElement.geometric_stiffness_matrix

    def wrapped(self, *args, **kwargs):
        calls.append("kg")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ShellElement, "geometric_stiffness_matrix", wrapped)
    mesh, bcs, a = _supported_plate(aluminium, n=4)
    bcs.fix(mesh.nodes_on_line(0, 0.0), [DOF.UX])
    load = LoadCase("compression").add_line_load(mesh.nodes_on_line(0, a), (-1.0, 0.0, 0.0))
    LinearStaticRunner(mesh, bcs, load).execute()
    assert calls == []
    LinearBucklingRunner(mesh, bcs, load).execute()
    assert calls, "linear buckling already assembles Kg; the static tangent does not"


def test_linear_buckling_still_uses_geometric_stiffness(aluminium):
    mesh, bcs, a = _supported_plate(aluminium, n=4)
    bcs.fix(mesh.nodes_on_line(0, 0.0), [DOF.UX])
    load = LoadCase("compression").add_line_load(mesh.nodes_on_line(0, a), (-1.0, 0.0, 0.0))
    result = linear_buckling(mesh, bcs, load, n_modes=1)
    assert result.load_factors[0] > 0.0
    assert np.isfinite(result.load_factors[0])
