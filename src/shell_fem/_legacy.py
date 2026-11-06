"""Old module paths (``shell_fem.laminate_*``, ``shell_fem.kernels``), kept for one release.

Version 0.2.0 moved every module into the pre_processing / processing /
simulation_runner / post_processing layout. Importing an old path still works and
returns a module holding the names of the new module (or, where one old module was
split, of all its parts), with a DeprecationWarning. These aliases go in 0.3.0.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
import types
import warnings

_P = "shell_fem."
OLD_PATHS = {
    "laminate_elements": ("pre_processing.element_library.shell",),
    "laminate_curved_shell": ("pre_processing.element_library.curved_shell",),
    "laminate_solid": ("pre_processing.element_library.solid",),
    "laminate_connectors": ("pre_processing.element_library.connectors",),
    "laminate_layup": ("pre_processing.property_library.layup",),
    "laminate_orientation": ("pre_processing.property_library.orientation",),
    "laminate_mesh_builders": ("pre_processing.mesh_library.builders",),
    "laminate_gmsh": ("pre_processing.mesh_library.gmsh",),
    "laminate_blade_geometry": ("pre_processing.mesh_library.blade_geometry",),
    "laminate_blade": ("pre_processing.mesh_library.blade", "post_processing.beam_section_properties"),
    "laminate_model_check": ("pre_processing.model_check",),
    "laminate_solver_pipeline": (
        "pre_processing.containers.mesh",
        "pre_processing.containers.boundary_conditions",
        "pre_processing.load_library.load_case",
        "processing.static.operations.linear_solvers",
        "processing.static.operations.assembly",
        "processing.static.operations.constraints",
        "processing.static.results.containers",
        "processing.static.results.compute_primary",
        "simulation_runner.static.linear_static_simulation",
    ),
    "laminate_stress_recovery": ("processing.static.results.stress_recovery",),
    "laminate_equilibrium": ("processing.static.results.equilibrium",),
    "laminate_results": ("processing.static.results.containers", "processing.static.results.results_extractor"),
    "laminate_progressive_damage": ("processing.progressive_damage.degradation",
                                    "simulation_runner.progressive_damage.progressive_damage_simulation"),
    "laminate_vtk": ("post_processing.vtk",),
    "laminate_benchmark_stress": ("verification.benchmark_stress",),
    "run_solver_benchmarks": ("verification.solver_benchmarks",),
    "kernels": ("simulation_runner.backends",),
    "kernels.base": ("simulation_runner.backends.base",),
    "kernels.calculix": ("simulation_runner.backends.calculix",),
    "kernels.decks": ("simulation_runner.backends.decks",),
    "kernels.nastran": ("simulation_runner.backends.nastran",),
    "kernels.native": ("simulation_runner.backends.native",),
    "kernels.registry": ("simulation_runner.backends.registry",),
    "laminate_kernels": ("simulation_runner.backends",),
    "laminate_fea_engine": ("pre_processing.element_library.shell",),
}


def _build(old: str) -> types.ModuleType:
    targets = [importlib.import_module(_P + t) for t in OLD_PATHS[old]]
    module = types.ModuleType(_P + old, f"Deprecated: names from {', '.join(t.__name__ for t in targets)}.")
    if old == "laminate_fea_engine":  # also re-exported the kinematic theories
        targets.append(importlib.import_module("laminate_constitutive_model.laminate_kinematics"))
    for target in targets:
        module.__dict__.update({k: v for k, v in vars(target).items() if not k.startswith("__")})
    return module


class _OldPathFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(_P) and fullname[len(_P):] in OLD_PATHS:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=fullname == _P + "kernels")
        return None

    def create_module(self, spec):
        old = spec.name[len(_P):]
        new = ", ".join(_P + t for t in OLD_PATHS[old])
        warnings.warn(f"{spec.name} has moved to {new}; the old path will be removed in shell-fem 0.3.0",
                      DeprecationWarning, stacklevel=2)
        return _build(old)

    def exec_module(self, module):
        pass


def install() -> None:
    if not any(isinstance(f, _OldPathFinder) for f in sys.meta_path):
        sys.meta_path.append(_OldPathFinder())
