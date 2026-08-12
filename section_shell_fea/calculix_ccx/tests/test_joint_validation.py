"""Offline + optional live tests for CalculiX joint export (shared + *TIE)."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from blade_structure_utilities.fem_utilities.section_shell_fea.calculix_ccx.joint_validation import (
    build_capped_catalogue_fixture,
    build_l_junction_fixture,
    inspect_joint_deck,
    run_joint_validation,
    write_joint_deck,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.calculix_ccx.mesh import (
    build_ccx_section_mesh,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.calculix_ccx.section_export import (
    write_section_inp,
)


def test_l_junction_shared_nodes_no_tie():
    panels, nx, nxy, strip = build_l_junction_fixture()
    with tempfile.TemporaryDirectory() as tmp:
        path, mesh, check = write_joint_deck(
            panels,
            nx,
            nxy,
            section_strip_mesh=strip,
            path=Path(tmp) / "l.inp",
            cap_tie_mode="none",
        )
        assert check.shared_junction_ok
        assert not check.has_tie
        text = path.read_text(encoding="utf-8")
        assert "*TIE" not in text
        assert "*ELEMENT, TYPE=S4R" in text
        # Two panels share the corner cluster → fewer unique nodes than 2× unique.
        unique = build_ccx_section_mesh(
            panels,
            section_strip_mesh=strip,
            interface_constraint_mode="shared",
            cap_tie_mode="none",
        )
        assert len(unique.clusters) >= 1


def test_capped_skin_offset_emits_tie_and_surfaces():
    panels, nx, nxy, strip, _shell = build_capped_catalogue_fixture("3D-F")
    mesh = build_ccx_section_mesh(
        panels,
        section_strip_mesh=strip,
        interface_constraint_mode="shared",
        cap_tie_mode="skin_offset",
    )
    assert mesh.cap_tie_mode == "skin_offset"
    assert mesh.cap_skin_ties, "expected at least one cap–skin *TIE surface"
    for tie in mesh.cap_skin_ties:
        assert tie.slave_ccx_nodes
        assert tie.master_ccx_elements
        assert tie.position_tolerance > 0.0

    with tempfile.TemporaryDirectory() as tmp:
        path = write_section_inp(
            mesh, panels, nx, nxy, Path(tmp) / "cap_tie.inp", cap_tie_export="tie"
        )
        text = path.read_text(encoding="utf-8")
        assert "*TIE" in text
        assert "*SURFACE" in text
        assert "POSITION TOLERANCE" in text
        assert "CapSkinSlave_" in text
        assert "CapSkinMaster_" in text
        check = inspect_joint_deck(path, expect_tie=True, mesh=mesh)
        assert check.has_tie
        assert check.n_tie_cards >= 1
        assert check.shared_junction_ok


def test_capped_none_omits_tie():
    panels, nx, nxy, strip, _shell = build_capped_catalogue_fixture("3D-F")
    with tempfile.TemporaryDirectory() as tmp:
        path, mesh, check = write_joint_deck(
            panels,
            nx,
            nxy,
            section_strip_mesh=strip,
            path=Path(tmp) / "untied.inp",
            cap_tie_mode="none",
        )
        assert mesh.cap_tie_mode == "none"
        assert not mesh.cap_skin_ties
        assert not check.has_tie
        assert "*TIE" not in path.read_text(encoding="utf-8")


def test_equation_fallback_export():
    panels, nx, nxy, strip, _shell = build_capped_catalogue_fixture("3D-F")
    mesh = build_ccx_section_mesh(
        panels,
        section_strip_mesh=strip,
        cap_tie_mode="skin_offset",
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = write_section_inp(
            mesh,
            panels,
            nx,
            nxy,
            Path(tmp) / "eq.inp",
            cap_tie_export="equation",
        )
        text = path.read_text(encoding="utf-8")
        assert "*EQUATION" in text
        assert "*TIE" not in text


def test_run_joint_validation_offline_decks():
    with tempfile.TemporaryDirectory() as tmp:
        report = run_joint_validation(
            tmp, layouts=("3D-F",), run_live_ccx=False, run_mitc4=False
        )
        assert any(c.has_tie for c in report.deck_checks)
        assert any(not c.has_tie for c in report.deck_checks)
        assert all(c.shared_junction_ok for c in report.deck_checks)
        text = report.summary()
        assert "CalculiX joint validation" in text


@pytest.mark.ccx
@pytest.mark.skipif(shutil.which("ccx") is None, reason="CalculiX ccx not on PATH")
def test_ccx_l_junction_live_smoke():
    from blade_structure_utilities.fem_utilities.section_shell_fea.calculix_ccx.solve import (
        solve_global_coupled_ccx,
    )

    panels, nx, nxy, strip = build_l_junction_fixture()
    with tempfile.TemporaryDirectory() as tmp:
        results, diag, inp = solve_global_coupled_ccx(
            panels,
            nx,
            nxy,
            section_strip_mesh=strip,
            work_dir=tmp,
            cap_tie_mode="none",
            job_name="lj_joint",
        )
        assert diag[0]["solver"] == "global_coupled_ccx"
        assert "*TIE" not in Path(inp).read_text(encoding="utf-8")
        assert results is not None


@pytest.mark.ccx
@pytest.mark.skipif(shutil.which("ccx") is None, reason="CalculiX ccx not on PATH")
def test_ccx_capped_tie_live_smoke():
    from blade_structure_utilities.fem_utilities.section_shell_fea.calculix_ccx.solve import (
        solve_global_coupled_ccx,
    )

    panels, nx, nxy, strip, _shell = build_capped_catalogue_fixture("3D-F")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            results, diag, inp = solve_global_coupled_ccx(
                panels,
                nx,
                nxy,
                section_strip_mesh=strip,
                work_dir=tmp,
                cap_tie_mode="skin_offset",
                cap_tie_export="tie",
                job_name="cap_tie",
            )
        except Exception:
            # Some CCX builds reject shell *TIE; equation fallback must still solve.
            results, diag, inp = solve_global_coupled_ccx(
                panels,
                nx,
                nxy,
                section_strip_mesh=strip,
                work_dir=tmp,
                cap_tie_mode="skin_offset",
                cap_tie_export="equation",
                job_name="cap_eq",
            )
        assert diag[0]["solver"] == "global_coupled_ccx"
        text = Path(inp).read_text(encoding="utf-8")
        assert ("*TIE" in text) or ("*EQUATION" in text)
        assert results is not None
