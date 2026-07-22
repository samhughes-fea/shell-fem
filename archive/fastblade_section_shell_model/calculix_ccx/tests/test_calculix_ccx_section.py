"""Section-level tests for external CalculiX ccx backend."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from blade_precompute.section_shell_model.pre_processing.containers.types import ProvenanceKind, ShellTheory
from blade_precompute.section_shell_model.calculix_ccx.mesh import build_ccx_section_mesh
from blade_precompute.section_shell_model.engine.shell_solve_registry import (
    BACKENDS,
    resolve_backend,
)
from blade_precompute.section_shell_model.calculix_ccx.section_export import write_section_inp


class _Lam:
    E = 70e9
    nu = 0.33
    t = 0.01


class _Panel:
    def __init__(self, nodes, s, label="skin:test"):
        self.nodes = np.asarray(nodes, dtype=float)
        self.s = np.asarray(s, dtype=float)
        self.label = label
        self.lam = _Lam()


def _l_shaped_panels():
    p0 = _Panel([[0, 0], [1, 0]], [0, 1], "skin:a")
    p1 = _Panel([[1, 0], [1, 1]], [0, 1], "web:b")
    nx0 = np.array([1000.0, 1000.0])
    nxy0 = np.array([0.0, 0.0])
    nx1 = np.array([0.0, 0.0])
    nxy1 = np.array([200.0, 200.0])
    return [p0, p1], [nx0, nx1], [nxy0, nxy1]


def test_shell_theory_calculix_ccx_registered():
    """CCX is quarantined (Phase 6): BACKENDS only gains it after opt-in registration."""
    from blade_precompute.section_shell_model.calculix_ccx.register import (
        register_calculix_ccx_backend,
    )
    from blade_precompute.section_shell_model.pre_processing.containers.types import (
        ShellAnalysisRegime,
        supported_shell_theories,
    )

    register_calculix_ccx_backend()
    assert ShellTheory.CALCULIX_CCX in BACKENDS
    assert resolve_backend(ShellTheory.CALCULIX_CCX).label == "external_calculix_ccx"
    # CCX is validation-only (Phase 6 quarantine): it is deliberately absent
    # from the production regime tuple even once opt-in registered — callers
    # select it explicitly (register_calculix_ccx_backend() + direct dispatch),
    # not through the general regime/theory config validator.
    assert ShellTheory.CALCULIX_CCX not in supported_shell_theories(ShellAnalysisRegime.LINEAR)


def test_build_ccx_section_mesh_l_junction():
    panels, _, _ = _l_shaped_panels()
    mesh = build_ccx_section_mesh(panels, 2, interface_constraint_mode="shared")
    assert len(mesh.nodes) > 0
    assert len(mesh.elements) >= 2
    assert len(mesh.elsets) == 2


def test_write_section_inp_l_shaped():
    panels, nx_panels, nxy_panels = _l_shaped_panels()
    mesh = build_ccx_section_mesh(panels, 2, interface_constraint_mode="shared")
    with tempfile.TemporaryDirectory() as tmp:
        path = write_section_inp(
            mesh, panels, nx_panels, nxy_panels, Path(tmp) / "section.inp"
        )
        text = path.read_text(encoding="utf-8")
        assert "*ELEMENT, TYPE=S4R" in text
        assert "*SHELL SECTION" in text
        assert "*CLOAD" in text


@pytest.mark.ccx
@pytest.mark.skipif(shutil.which("ccx") is None, reason="CalculiX ccx not on PATH")
def test_ccx_l_junction_solves():
    from blade_precompute.section_shell_model.calculix_ccx.solve import (
        solve_global_coupled_ccx,
    )

    panels, nx_panels, nxy_panels = _l_shaped_panels()
    with tempfile.TemporaryDirectory() as tmp:
        results, diag, _inp = solve_global_coupled_ccx(
            panels,
            nx_panels,
            nxy_panels,
            work_dir=tmp,
            n_elements_per_panel=2,
            job_name="lj",
        )
        assert diag[0]["solver"] == "global_coupled_ccx"
        if results[0]:
            assert results[0][0].provenance["Nx"].kind is ProvenanceKind.CALCULIX_CCX
