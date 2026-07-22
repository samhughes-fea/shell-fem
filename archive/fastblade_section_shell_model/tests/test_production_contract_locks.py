"""Production-contract locks for section_shell_model next-steps baseline.

Locks the accepted production defaults without scheduling parked research:
FSDT-only theory support, lazy CCX registration, deleted clamped/HSDT paths,
beam-only optimiser FI surface, and single periodic free-set for face traction.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from blade_precompute.section_shell_model.engine.shell_solve_registry import BACKENDS
from blade_precompute.section_shell_model.engine.shell_theory_registry import (
    global_backend_theories,
    periodic_solver_map,
    station_solver_map,
)
from blade_precompute.section_shell_model.pre_processing.containers.types import (
    EXPERIMENTAL_SHELL_THEORIES_BY_REGIME,
    SUPPORTED_SHELL_THEORY_BY_REGIME,
    ShellAnalysisRegime,
    ShellTheory,
    validate_shell_configuration,
)
from blade_precompute.section_shell_model.pre_processing.element_library.formulation_factory import (
    resolve_element_factory,
)

_PKG = Path(__file__).resolve().parents[1]


def test_fsdt_is_sole_production_linear_theory() -> None:
    assert SUPPORTED_SHELL_THEORY_BY_REGIME[ShellAnalysisRegime.LINEAR] == (
        ShellTheory.FSDT,
    )
    assert SUPPORTED_SHELL_THEORY_BY_REGIME[ShellAnalysisRegime.NONLINEAR] == ()
    assert EXPERIMENTAL_SHELL_THEORIES_BY_REGIME[ShellAnalysisRegime.LINEAR] == ()
    validate_shell_configuration(ShellAnalysisRegime.LINEAR, ShellTheory.FSDT)
    with pytest.raises(ValueError, match="Unsupported shell configuration"):
        validate_shell_configuration(
            ShellAnalysisRegime.LINEAR, ShellTheory.CALCULIX_CCX
        )


def test_shell_theory_enum_has_no_hsdt() -> None:
    assert not hasattr(ShellTheory, "HSDT")
    assert {t.value for t in ShellTheory} == {"fsdt", "calculix_ccx"}


def test_fsdt_element_factory_resolves_without_opt_in() -> None:
    factory = resolve_element_factory(ShellTheory.FSDT)
    assert factory is not None


def test_production_modules_do_not_eagerly_import_ccx_register() -> None:
    """AST guard: production dispatch modules must not import calculix_ccx at load."""
    targets = [
        _PKG / "api" / "periodic.py",
        _PKG / "engine" / "shell_solve_registry.py",
        _PKG / "engine" / "station_theory_dispatch.py",
        _PKG / "engine" / "shell_theory_registry.py",
    ]
    for path in targets:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                mod = node.module
                assert "calculix_ccx" not in mod, (
                    f"{path.name} ImportFrom {mod} (eager CCX import)"
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert "calculix_ccx" not in alias.name, (
                        f"{path.name} Import {alias.name} (eager CCX import)"
                    )
    assert ShellTheory.FSDT in periodic_solver_map()


def test_ccx_registration_is_lazy_and_idempotent() -> None:
    from blade_precompute.section_shell_model.calculix_ccx.register import (
        register_calculix_ccx_backend,
    )

    register_calculix_ccx_backend()
    register_calculix_ccx_backend()  # idempotent
    assert ShellTheory.CALCULIX_CCX in BACKENDS
    assert ShellTheory.CALCULIX_CCX in periodic_solver_map()
    assert ShellTheory.CALCULIX_CCX in station_solver_map()
    assert ShellTheory.CALCULIX_CCX in global_backend_theories()


def test_deleted_paths_absent_from_package() -> None:
    assert not (_PKG / "api" / "legacy_clamped.py").exists()
    assert not (_PKG / "pre_processing" / "element_library" / "hsdt").exists()
    assert not (
        _PKG / "pre_processing" / "element_library" / "mitc4" / "utilities" / "panel_mitc4_model.py"
    ).exists()


def test_optimiser_shell_fi_solve_is_beam_only_kwargs() -> None:
    """``_run_section_shell_solve`` must not accept face-traction channel kwargs."""
    from blade_precompute.section_optimisation.engine import shell_fi_eval as sfe

    sig = inspect.signature(sfe._run_section_shell_solve)
    forbidden = {"traction_field", "traction_points_yz", "q_y", "q_z", "face_traction"}
    assert forbidden.isdisjoint(sig.parameters)


def test_periodic_traction_uses_periodic_reduced_bvp() -> None:
    src = (_PKG / "engine" / "periodic_traction_superposition.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(src)
    names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    } | {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    }
    assert "PeriodicReducedBVP" in names
    # Contour clamp overlay must not reappear on the traction path.
    assert "build_clamp_supports" not in names


def test_interlaminar_allowable_policy_is_first_ply_shared() -> None:
    from types import SimpleNamespace

    from blade_precompute.section_shell_model.api.fi_recovery import (
        interlaminar_allowable_from_lam,
        interlaminar_allowable_pa,
    )
    from blade_precompute.section_shell_model.processing.results.compute_quaternary.compute_hashin_fi import (
        interlaminar_allowable_from_lam as from_hashin,
    )

    ply0 = SimpleNamespace(S13=12e6, S12=40e6)
    ply1 = SimpleNamespace(S13=80e6, S12=40e6)  # higher mid-stack must not govern
    lam = SimpleNamespace(plies=[(ply0, 0.001), (ply1, 0.001)])
    assert interlaminar_allowable_pa(ply0) == pytest.approx(12e6)
    assert interlaminar_allowable_from_lam(lam) == pytest.approx(12e6)
    assert from_hashin(lam) == pytest.approx(12e6)
