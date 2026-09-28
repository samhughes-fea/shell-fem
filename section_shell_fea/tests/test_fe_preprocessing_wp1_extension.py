"""WP1 — formulation extension contract.

Proves a *non-MITC4* formulation can register into the unified graph and drive
``build_formulation_cache`` through ``formulation_id`` alone, without importing any
MITC4 module in the exercise (a meta-path guard blocks fresh ``mitc4`` imports).

Also an AST gate: the generic element-library modules must not import MITC4
classes at module top level or hard-code MITC4 Gauss rules.
"""

from __future__ import annotations

import ast
import importlib.abc
import importlib.machinery
import sys
from pathlib import Path

import numpy as np
import pytest

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.formulation_objects import (
    ShellForceObject,
    ShellStiffnessObject,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import (
    ShellTheory,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.element_library.contracts import (
    mitc4_fsdt_descriptor,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.element_library.formulation_factory import (
    build_formulation_cache,
    resolve_element_factory,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.element_library.registry import (
    ShellFormulationRegistration,
    clear_formulation_registry_for_tests,
    default_formulation_id,
    ensure_default_formulations,
    register_formulation,
    resolve_formulation,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.element_library.template_cache import (
    ShellFormulationTemplateCache,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.tests._panel_topology_factory import (
    build_shell_panel_topology,
)
from blade_structure_utilities.mesh_utilities.section_mesh import build_section_strip_mesh
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.shell_mesh_map import (
    panel_maps_from_section_strip_mesh,
)

_FAKE_ID = "FAKE-FSDT"
_FAKE_THEORY = "FAKE-THEORY"
_FAKE_NDOF = 5


# --------------------------------------------------------------------------- #
# Minimal non-MITC4 stub formulation
# --------------------------------------------------------------------------- #
class _StubElement:
    """Trivial element: identity ``K_e`` and zero ``F_e`` (no MITC4 kernels)."""

    def __init__(self, *, element_id, panel_index, local_element_index, L_s, L_x):
        self.element_id = element_id
        self.panel_index = panel_index
        self.local_element_index = local_element_index
        self.dof_per_node = _FAKE_NDOF
        self._L_s = float(L_s)
        self._L_x = float(L_x)

    def element_stiffness_matrix(self) -> ShellStiffnessObject:
        n = 4 * _FAKE_NDOF
        return ShellStiffnessObject(
            element_id=self.element_id,
            element_type=_FAKE_ID,
            K_e=np.eye(n, dtype=np.float64),
            panel_index=self.panel_index,
            local_element_index=self.local_element_index,
            L_s=self._L_s,
            L_s_donnell=self._L_s,
            L_x=self._L_x,
            formulation_id=_FAKE_ID,
        )

    def element_force_vector(self) -> ShellForceObject:
        n = 4 * _FAKE_NDOF
        return ShellForceObject(
            element_id=self.element_id,
            element_type=_FAKE_ID,
            F_e=np.zeros(n, dtype=np.float64),
            panel_index=self.panel_index,
            local_element_index=self.local_element_index,
            L_s=self._L_s,
            L_s_donnell=self._L_s,
            L_x=self._L_x,
            formulation_id=_FAKE_ID,
        )


class _StubFactory:
    theory = _FAKE_THEORY
    n_dof_node = _FAKE_NDOF
    formulation_id = _FAKE_ID

    def iter_elements(
        self,
        panel_maps,
        zones,
        Nx_panels,
        Nxy_panels,
        *,
        L_x: float = 1.0,
        zones_outboard=None,
        panel_abd_stacks=None,
        panel_constitutive_specs=None,
        template_cache=None,
    ):
        elem_id = 0
        for pi, pm in enumerate(panel_maps):
            if len(pm.s_nodes) == 0:
                continue
            s_nodes = np.asarray(pm.s_nodes, dtype=float)
            for ei, e in enumerate(pm.elements):
                i0, i1 = e[0], e[1]
                L_s = float(abs(s_nodes[i1] - s_nodes[i0]))
                if L_s < 1e-30:
                    continue
                # Exercise the generic per-build template cache with a stub key.
                if template_cache is not None:
                    template_cache.get_or_build(
                        (pi, L_s), lambda: np.eye(4 * _FAKE_NDOF)
                    )
                yield _StubElement(
                    element_id=elem_id,
                    panel_index=pi,
                    local_element_index=ei,
                    L_s=L_s,
                    L_x=L_x,
                )
                elem_id += 1


class _StubKernel:
    name = "stub_kernel"
    ndof = _FAKE_NDOF

    def recover_resultants(self, d_elem, L_s, L_x, ABD):
        return {k: 0.0 for k in ("N11", "N22", "N12", "M11", "M22", "M12")}


class _StubProjector:
    formulation_id = _FAKE_ID

    def project_resultants(self, native):
        names = ("N11", "N22", "N12", "M11", "M22", "M12", "Q1", "Q2")
        return tuple(float(native.get(n, 0.0)) for n in names), tuple(
            "modelled" for _ in names
        )

    def project_material_stress(self, s11, s22, t12, *, sigma33=0.0, tau23=0.0, tau13=0.0):
        return (s11, s22, 0.0, tau23, tau13, t12), tuple(
            "constitutive_recovered" for _ in range(6)
        )


class _BlockMitc4Finder(importlib.abc.MetaPathFinder):
    """Raise if any fresh ``mitc4`` module import is attempted."""

    def find_spec(self, fullname, path, target=None):
        if fullname.endswith(".mitc4") or ".mitc4." in fullname:
            raise AssertionError(
                f"fake-formulation exercise imported a mitc4 module: {fullname}"
            )
        return None


def _fake_descriptor():
    """Clone the MITC4 descriptor contract with a fake id/theory (contract is generic)."""
    desc = mitc4_fsdt_descriptor()
    return desc.__class__(
        **{
            **{f.name: getattr(desc, f.name) for f in desc.__dataclass_fields__.values()},
            "formulation_id": _FAKE_ID,
            "continuum_theory": _FAKE_THEORY,
        }
    )


def _register_fake():
    calls = {"template_key": 0, "validate_element": 0, "validate_cache": 0}

    def _template_key_strategy(element):
        calls["template_key"] += 1
        return (element.panel_index, element.local_element_index)

    def _validate_element(element):
        calls["validate_element"] += 1
        if element.dof_per_node != _FAKE_NDOF:
            raise ValueError("bad ndof")

    def _validate_cache(cache):
        calls["validate_cache"] += 1
        if cache.formulation_id != _FAKE_ID:
            raise ValueError("bad cache formulation_id")

    register_formulation(
        ShellFormulationRegistration(
            descriptor=_fake_descriptor(),
            factory_cls=_StubFactory,
            kernel=_StubKernel(),
            projector=_StubProjector(),
            template_key_strategy=_template_key_strategy,
            validate_element=_validate_element,
            validate_cache=_validate_cache,
        )
    )
    return calls


def _panel_maps_zones_loads():
    panel_topology = build_shell_panel_topology("2412", [0.35], chord_m=1.0)
    zones = list(panel_topology.zones)
    nx = [np.linspace(0.0, 1000.0, len(np.asarray(p.s, dtype=float))) for p in zones]
    nxy = [np.zeros_like(n) for n in nx]
    strip = build_section_strip_mesh(panel_topology.shell_inputs, n_elements_per_panel=3)
    panel_maps, _ = panel_maps_from_section_strip_mesh(strip, zones)
    return panel_maps, zones, nx, nxy


def test_fake_formulation_registers_and_builds_without_mitc4_imports():
    # Ensure MITC4 is already registered/imported so ensure_default_formulations is a
    # no-op inside the guarded region (we only forbid *fresh* mitc4 imports).
    ensure_default_formulations()
    panel_maps, zones, nx, nxy = _panel_maps_zones_loads()

    finder = _BlockMitc4Finder()
    sys.meta_path.insert(0, finder)
    try:
        calls = _register_fake()

        # Registration graph resolves the fake by id and by its (fake) theory default.
        assert default_formulation_id(_FAKE_THEORY) == _FAKE_ID
        registration = resolve_formulation(_FAKE_ID)
        assert registration.template_key_strategy is not None
        assert registration.validate_element is not None

        # resolve_element_factory prefers formulation_id (single graph).
        factory = resolve_element_factory(formulation_id=_FAKE_ID)
        assert isinstance(factory, _StubFactory)

        # Cache build via formulation_id drives the stub factory end-to-end.
        cache = build_formulation_cache(
            panel_maps, zones, nx, nxy, formulation_id=_FAKE_ID, L_x=1.0
        )
        assert cache.formulation_id == _FAKE_ID
        assert cache.n_dof_node == _FAKE_NDOF
        assert cache.stiffness_objects

        # DOF assignment (generic) filled 4*ndof indices per element.
        for stiff in cache.stiffness_objects:
            assert stiff.global_dof_indices is not None
            assert stiff.global_dof_indices.size == 4 * _FAKE_NDOF
            assert stiff.K_e.shape == (4 * _FAKE_NDOF, 4 * _FAKE_NDOF)

        # Validation hooks are invoked automatically by build_formulation_cache.
        assert calls["validate_element"] == len(cache.stiffness_objects)
        assert calls["validate_cache"] == 1

        # Template-key strategy + recovery/projection hooks are callable.
        key = registration.template_key_strategy(cache.stiffness_objects[0])
        assert key is not None
        res = registration.kernel.recover_resultants(
            np.zeros(4 * _FAKE_NDOF), 0.1, 1.0, np.eye(6)
        )
        assert set(res) >= {"N11", "M12"}
        values, caps = registration.projector.project_resultants(
            {"N11": 1.0, "Q1": 3.0}
        )
        assert values[0] == 1.0 and caps[0] == "modelled"
    finally:
        sys.meta_path.remove(finder)
        clear_formulation_registry_for_tests()

    # Default restored to production MITC4 after cleanup.
    assert default_formulation_id(ShellTheory.FSDT) == "MITC4-FSDT"


def test_template_cache_refuses_incomplete_identity_and_counts():
    cache = ShellFormulationTemplateCache()
    a = np.eye(3)
    assert cache.get_or_build(("k", 1), lambda: a) is a
    # Second identical key: a hit (same object, no rebuild).
    assert cache.get_or_build(("k", 1), lambda: np.eye(3)) is a
    # None key: refused — always rebuilds, never stored.
    b = cache.get_or_build(None, lambda: np.zeros(2))
    c = cache.get_or_build(None, lambda: np.zeros(2))
    assert b is not c
    stats = cache.stats()
    assert stats.hits == 1
    assert stats.misses == 1
    assert stats.refusals == 2
    assert stats.unique == 1


# --------------------------------------------------------------------------- #
# AST gate: generic modules stay formulation-agnostic
# --------------------------------------------------------------------------- #
_PKG = Path(__file__).resolve().parents[1]
_ELIB = _PKG / "pre_processing" / "element_library"
_GENERIC_MODULES = (
    _ELIB / "formulation_factory.py",
    _ELIB / "registry.py",
    _ELIB / "contracts.py",
    _ELIB / "shell_element_base.py",
    _ELIB / "formulation_validation.py",
)


def _top_level_import_modules(tree: ast.AST) -> list[str]:
    """Module names imported at module scope (not nested inside functions)."""
    modules: list[str] = []

    def _visit(body):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue  # lazy in-function imports are the allowed compat seam
            if isinstance(node, ast.Import):
                modules.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    modules.append(node.module)
            elif isinstance(node, (ast.If, ast.Try, ast.With)):
                for attr in ("body", "orelse", "finalbody", "handlers"):
                    sub = getattr(node, attr, None)
                    if sub:
                        if attr == "handlers":
                            for h in sub:
                                _visit(h.body)
                        else:
                            _visit(sub)

    _visit(tree.body)  # type: ignore[attr-defined]
    return modules


@pytest.mark.parametrize("path", _GENERIC_MODULES, ids=lambda p: p.name)
def test_generic_module_has_no_top_level_mitc4_import(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = [m for m in _top_level_import_modules(tree) if "mitc4" in m]
    assert not offenders, f"{path.name} imports mitc4 at top level: {offenders}"


@pytest.mark.parametrize("path", _GENERIC_MODULES, ids=lambda p: p.name)
def test_generic_module_has_no_hardcoded_gauss_rule(path: Path):
    src = path.read_text(encoding="utf-8")
    # No Gauss quadrature helper defs/calls, no Gauss-point sqrt literals.
    assert "_gauss(" not in src, f"{path.name} references a Gauss rule helper"
    assert "def _gauss" not in src, f"{path.name} defines a Gauss rule"
    assert "sqrt(3.0" not in src and "sqrt(3.0/5.0)" not in src, (
        f"{path.name} hard-codes Gauss points"
    )
