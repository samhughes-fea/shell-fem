"""Package layout: import direction between the layers, the static stages and runner, and the 0.1 paths.

    pre_processing  <-  processing  <-  simulation_runner  <-  post_processing / verification

Each layer imports only the layers to its left (and itself).
"""

import ast
import importlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import shell_fem
from shell_fem import (
    AssembleGlobalSystem,
    AssembleLoadVector,
    BoundaryConditions,
    ComputePrimaryResults,
    ConstrainGlobalSystem,
    FEAMesh,
    FSDTModel,
    Laminate,
    LinearStaticRunner,
    LoadCase,
    LoadCombination,
    ReconstructGlobalField,
    ReduceConstrainedSystem,
    SolveReducedSystem,
    StructuralGlobalSolver,
)

ROOT = Path(shell_fem.__file__).parent
LAYERS = ["pre_processing", "processing", "simulation_runner", "post_processing", "verification"]
ALLOWED = {
    "pre_processing": {"pre_processing"},
    "processing": {"pre_processing", "processing"},
    "simulation_runner": {"pre_processing", "processing", "simulation_runner", "verification"},
    "post_processing": {"pre_processing", "processing", "simulation_runner", "post_processing"},
    "verification": {"verification"},
}

# Known exception: Layup.ply_failure reads a solved result (a lazy import inside the method).
# It moves to processing/static/results with the per-ply failure field.
EXCEPTIONS = {"pre_processing/property_library/layup.py imports shell_fem.processing"}


def _shell_fem_imports(path):
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("shell_fem."):
            yield node.module.split(".")[1], node.lineno
        elif isinstance(node, ast.ImportFrom) and node.level:
            yield None, node.lineno


def test_layers_import_in_one_direction_only():
    wrong = []
    for layer in LAYERS:
        for path in sorted((ROOT / layer).rglob("*.py")):
            for target, line in _shell_fem_imports(path):
                where = f"{path.relative_to(ROOT).as_posix()} imports shell_fem.{target}"
                if target is not None and target not in ALLOWED[layer] and where not in EXCEPTIONS:
                    wrong.append(f"{where} (line {line})")
    assert wrong == []


def test_top_level_has_only_the_package_entry_points():
    assert sorted(p.name for p in ROOT.glob("*.py")) == ["__init__.py", "_legacy.py"]


@pytest.fixture
def plate(cfrp):
    lam = Laminate.from_angles(cfrp, [0, 90, 90, 0], 0.002)
    mesh = FEAMesh.rectangular_plate(0.2, 0.1, 6, 3, lam, FSDTModel())
    bcs = BoundaryConditions().clamp(mesh.nodes_where(lambda p: p[0] < 1e-9))
    tip = mesh.nodes_where(lambda p: p[0] > 0.2 - 1e-9)
    bend, pull = LoadCase("bend"), LoadCase("pull")
    for node in tip:
        bend.add_nodal_load(node, 2, -1.0)
        pull.add_nodal_load(node, 0, 10.0)
    return mesh, bcs, [bend, pull]


def test_stages_reproduce_the_runner(plate):
    mesh, bcs, cases = plate
    system = AssembleGlobalSystem(mesh, bcs).run()
    constrained = ConstrainGlobalSystem(mesh, bcs).run(system)
    reduced = ReduceConstrainedSystem().run(constrained)
    solver = SolveReducedSystem()
    factorised = solver.factorise(reduced)
    F = AssembleLoadVector(mesh).run(cases[0])
    U = ReconstructGlobalField().run(reduced, solver.solve(factorised, F))
    by_hand = ComputePrimaryResults(mesh, system).run(U, F, cases[0])

    results = LinearStaticRunner(mesh, bcs, cases).execute().get_results()
    assert np.allclose(by_hand.displacements, results["bend"].displacements, rtol=1e-12, atol=0.0)
    assert np.allclose(by_hand.reactions, results["bend"].reactions, rtol=1e-12, atol=1e-12)
    old = StructuralGlobalSolver().solve(mesh, bcs, cases[0])
    assert np.array_equal(old.displacements, results["bend"].displacements)


def test_runner_superposes_combinations_and_records_stage_times(plate):
    mesh, bcs, cases = plate
    combo = LoadCombination("uls", {"bend": 1.35, "pull": 1.5})
    results = LinearStaticRunner(mesh, bcs, cases, [combo]).execute().get_results()
    assert list(results) == ["bend", "pull", "uls"]
    expected = 1.35 * results["bend"].displacements + 1.5 * results["pull"].displacements
    assert np.allclose(results["uls"].displacements, expected)
    d = results.diagnostics
    assert set(d.phase_times) == {"assemble", "constrain", "reduce", "factorise", "assemble_loads", "solve",
                                  "reconstruct", "primary_results", "combinations"}
    assert d.n_dof == mesh.n_dof and 0 < d.n_free < d.n_dof


def test_runner_needs_execute_and_unique_names(plate):
    mesh, bcs, cases = plate
    with pytest.raises(RuntimeError):
        LinearStaticRunner(mesh, bcs, cases).get_results()
    with pytest.raises(ValueError):
        LinearStaticRunner(mesh, bcs, [cases[0], cases[0]])


@pytest.mark.parametrize("old, name, new", [
    ("laminate_solver_pipeline", "StructuralGlobalSolver", "simulation_runner.static.linear_static_simulation"),
    ("laminate_solver_pipeline", "FEAMesh", "pre_processing.containers.mesh"),
    ("laminate_solver_pipeline", "_dof_key", "pre_processing.containers.boundary_conditions"),
    ("laminate_elements", "Quad4ShellElement", "pre_processing.element_library.shell"),
    ("laminate_blade", "beam_section_properties", "post_processing.beam_section_properties"),
    ("laminate_blade", "BladeBuilder", "pre_processing.mesh_library.blade"),
    ("laminate_blade_kernel_mesh", "KernelMeshBladeBuilder", "pre_processing.mesh_library.blade_kernel_mesh"),
    ("laminate_blade_kernel_mesh", "solve_kernel_pcomp", "simulation_runner.static.kernel_pcomp"),
    ("laminate_mesh_builders", "_structured", "pre_processing.mesh_library.builders"),
    ("laminate_progressive_damage", "SmearedCrackDegradation", "processing.progressive_damage.degradation"),
    ("laminate_results", "ResultsExtractor", "processing.static.results.results_extractor"),
    ("kernels.base", "AnalysisModel", "simulation_runner.backends.base"),
    ("kernels", "CalculiXDeckWriter", "simulation_runner.backends"),
    ("run_solver_benchmarks", "build_model", "verification.solver_benchmarks"),
])
def test_old_module_paths_still_import_with_a_warning(old, name, new):
    sys.modules.pop(f"shell_fem.{old}", None)
    with pytest.warns(DeprecationWarning, match="0.3.0"):
        module = importlib.import_module(f"shell_fem.{old}")
    assert getattr(module, name) is getattr(importlib.import_module(f"shell_fem.{new}"), name)


def test_new_paths_import_without_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        for path in sorted(ROOT.rglob("*.py")):
            rel = path.relative_to(ROOT.parent).with_suffix("")
            parts = rel.parts[:-1] if rel.name == "__init__" else rel.parts
            importlib.import_module(".".join(parts))
