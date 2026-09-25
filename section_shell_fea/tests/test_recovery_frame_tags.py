"""FEA-R-001: recovery containers carry validated FrameTag fields."""

from __future__ import annotations

import pytest

from blade_structure_utilities.geometry_utilities.coordinate_frames import FRAME_B, FRAME_E, FRAME_M
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.recovery_types import (
    ShellPanelResultants,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.containers.elemental_results import (
    ShellElementalResults,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.containers.ply_results import (
    ShellPlyResults,
)


def test_shell_panel_resultants_default_frame_is_e():
    row = ShellPanelResultants(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert row.frame == FRAME_E


def test_shell_panel_resultants_rejects_non_e_frame():
    with pytest.raises(ValueError, match="FRAME_E|element/laminate"):
        ShellPanelResultants(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, frame=FRAME_B)


def test_shell_ply_results_frame_tags():
    ply = ShellPlyResults()
    assert ply.mid_surface_frame == FRAME_E
    assert ply.ply_material_frame == FRAME_M
    with pytest.raises(ValueError, match="mid_surface_frame"):
        ShellPlyResults(mid_surface_frame=FRAME_M)
    with pytest.raises(ValueError, match="ply_material_frame"):
        ShellPlyResults(ply_material_frame=FRAME_E)


def test_shell_elemental_results_resultant_frame():
    el = ShellElementalResults()
    assert el.resultant_frame == FRAME_E
    with pytest.raises(ValueError, match="resultant_frame"):
        ShellElementalResults(resultant_frame=FRAME_B)
