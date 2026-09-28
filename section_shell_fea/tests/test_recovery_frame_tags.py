"""FEA-R-001: recovery containers carry validated FrameTag fields."""

from __future__ import annotations

import numpy as np
import pytest

from blade_structure_utilities.geometry_utilities.coordinate_frames import FRAME_B, FRAME_E, FRAME_M
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.recovery_types import (
    ShellPanelResultants,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.station_analysis_types import (
    SectionShellRecoveryBundle,
    ThinWallResult,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.compute_tertiary.local_clpt_shell import (
    StationCLPTShellResult,
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


def test_station_clpt_shell_result_frame_tags():
    row = StationCLPTShellResult(
        fi=np.zeros(1),
        eps0=np.zeros(3),
        kappa=np.zeros(3),
        sig_lam_mid=[],
        eps_lam_mid=[],
        plies=[],
        N_vec=np.zeros(3),
        M_vec=np.zeros(3),
    )
    assert row.mid_surface_frame == FRAME_E
    assert row.ply_material_frame == FRAME_M
    with pytest.raises(ValueError, match="FEA-R-003"):
        StationCLPTShellResult(
            fi=np.zeros(1),
            eps0=np.zeros(3),
            kappa=np.zeros(3),
            sig_lam_mid=[],
            eps_lam_mid=[],
            plies=[],
            N_vec=np.zeros(3),
            M_vec=np.zeros(3),
            mid_surface_frame=FRAME_M,
        )
    with pytest.raises(ValueError, match="FEA-R-003"):
        StationCLPTShellResult(
            fi=np.zeros(1),
            eps0=np.zeros(3),
            kappa=np.zeros(3),
            sig_lam_mid=[],
            eps_lam_mid=[],
            plies=[],
            N_vec=np.zeros(3),
            M_vec=np.zeros(3),
            ply_material_frame=FRAME_E,
        )


def test_recovery_bundle_pierce_frame_is_b():
    tw = ThinWallResult(
        zones=[],
        booms=[],
        webs_geom=[],
        q_tot=[],
        sig_p=[],
        sig_b=[],
        q0=[],
        props=None,
        y_sc=0.0,
        z_sc=0.0,
        areas=[],
        I_omega=0.0,
        gamma_y=0.0,
        gamma_z=0.0,
        GA_y=0.0,
        GA_z=0.0,
        q_primary=[],
        q_warp=[],
    )
    assert tw.pierce_frame == FRAME_B
    with pytest.raises(ValueError, match="FEA-R-003"):
        ThinWallResult(
            zones=[],
            booms=[],
            webs_geom=[],
            q_tot=[],
            sig_p=[],
            sig_b=[],
            q0=[],
            props=None,
            y_sc=0.0,
            z_sc=0.0,
            areas=[],
            I_omega=0.0,
            gamma_y=0.0,
            gamma_z=0.0,
            GA_y=0.0,
            GA_z=0.0,
            q_primary=[],
            q_warp=[],
            pierce_frame=FRAME_E,
        )
    bundle = SectionShellRecoveryBundle(
        zones=[],
        booms=[],
        webs_geom=[],
        q_tot=[],
        sig_p=[],
        sig_b=[],
        q0=[],
        props=None,
        y_sc=0.1,
        z_sc=-0.2,
        areas=[],
        I_omega=0.0,
        gamma_y=0.0,
        gamma_z=0.0,
        GA_y=0.0,
        GA_z=0.0,
        q_primary=[],
        q_warp=[],
    )
    assert bundle.pierce_frame == FRAME_B
    assert bundle.y_e == 0.1
    assert bundle.z_e == -0.2
    assert bundle.thin_wall.pierce_frame == FRAME_B
    with pytest.raises(ValueError, match="FEA-R-003"):
        SectionShellRecoveryBundle(
            zones=[],
            booms=[],
            webs_geom=[],
            q_tot=[],
            sig_p=[],
            sig_b=[],
            q0=[],
            props=None,
            y_sc=0.0,
            z_sc=0.0,
            areas=[],
            I_omega=0.0,
            gamma_y=0.0,
            gamma_z=0.0,
            GA_y=0.0,
            GA_z=0.0,
            q_primary=[],
            q_warp=[],
            pierce_frame=FRAME_E,
        )


def test_global_node_meta_and_load_request_pierce_frame():
    from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.topology.shell_mesh_map import (
        GlobalNodeMeta,
    )
    from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.load_channel_types import (
        SectionExternalLoadRequest,
    )

    meta = GlobalNodeMeta(
        n_nodes=1,
        yz=np.zeros((1, 2)),
        is_top=np.zeros(1, dtype=bool),
        tangent_yz=np.zeros((1, 2)),
        arc_length_s=np.zeros(1),
        pair_group=np.full(1, -1, dtype=int),
    )
    assert meta.frame == FRAME_B
    with pytest.raises(ValueError, match="FEA-R-002"):
        GlobalNodeMeta(
            n_nodes=1,
            yz=np.zeros((1, 2)),
            is_top=np.zeros(1, dtype=bool),
            tangent_yz=np.zeros((1, 2)),
            arc_length_s=np.zeros(1),
            pair_group=np.full(1, -1, dtype=int),
            frame=FRAME_E,
        )
    req = SectionExternalLoadRequest(f_beam_mode=np.zeros(6))
    assert req.traction_pierce_frame == FRAME_B
