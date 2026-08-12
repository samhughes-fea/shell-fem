"""Shell-native smoke station builders for support callers."""

from __future__ import annotations

import numpy as np

from blade_structure_utilities.geometry_utilities.section_contract.shell_midline_strip import ShellMidlineStrip

from .constitutive_adapter import build_shell_constitutive_stack
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import (
    ShellRecoveryMaterialPoint,
    ShellRecoverySectionPointSet,
    ShellSectionStationAnalysis,
    ShellSectionStationGeometry,
    ShellStationProvenance,
)


def build_smoke_shell_station_analysis(
    *,
    z_stations: np.ndarray,
    nodal_R: np.ndarray | None = None,
) -> tuple[list[ShellSectionStationAnalysis], np.ndarray]:
    """Build minimal shell-native smoke stations for CLI workflows."""

    z = np.asarray(z_stations, dtype=float).ravel()
    provenance = ShellStationProvenance(mesh_n_elements_per_panel=4)
    constitutive = build_shell_constitutive_stack(
        station_z=z,
        K6=np.stack([np.eye(6)] * z.size, axis=0),
        provenance=provenance,
    )
    stations: list[ShellSectionStationAnalysis] = []
    for i, z_station in enumerate(z):
        stations.append(
            ShellSectionStationAnalysis(
                geometry=ShellSectionStationGeometry(
                    station_id=f"station-{i}",
                    z_station=float(z_station),
                    strips=(
                        ShellMidlineStrip(
                            label="panel",
                            kind="skin",
                            midline_b=np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
                            thickness_m=0.02,
                        ),
                    ),
                    provenance=provenance,
                ),
                constitutive=constitutive[i],
                recovery_point_set=ShellRecoverySectionPointSet(
                    composite_labels=("panel",),
                    isotropic_labels=tuple(),
                    composite_resultant_basis=np.zeros((1, 6, 6), dtype=float),
                    isotropic_resultant_basis=np.zeros((0, 6, 3), dtype=float),
                    abd_inv=np.stack([np.eye(6)], axis=0),
                    material_points=(
                        ShellRecoveryMaterialPoint(
                            subcomponent_label="panel",
                            ply_index=0,
                            ply_count=1,
                            z_coord=0.0,
                            q_bar=np.eye(3),
                            transform=np.eye(3),
                            strengths={"Xt": 1.0, "Xc": 1.0, "Yt": 1.0, "Yc": 1.0, "S12": 1.0},
                        ),
                    ),
                ),
                provenance=provenance,
            )
        )
    if nodal_R is None:
        nodal_R = np.stack([np.eye(3)] * z.size, axis=0)
    return stations, np.asarray(nodal_R, dtype=float)


__all__ = ["build_smoke_shell_station_analysis"]
