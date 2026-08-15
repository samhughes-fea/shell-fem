"""Shell-owned recovery-point-set extractor and builder."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..pre_processing.constitutive.panel_abd import panel_abd_tuple_from_lam
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import (
    BeamSectionConstitutiveInput,
    ShellStationBucklingBundle,
    ShellRecoveryMaterialPoint,
    ShellRecoverySectionPointSet,
    ShellSectionStationAnalysis,
    ShellSectionStationGeometry,
    ShellStationProvenance,
)


def extract_shell_laminate_recovery_data(
    zones: Sequence[Any],
    *,
    panel_abd: np.ndarray | None = None,
    panel_labels: Sequence[str] | None = None,
) -> ShellRecoverySectionPointSet:
    """Extract shell-owned laminate recovery data from MITC4/thin-wall zone objects.

    ``panel_abd`` / ``panel_labels`` keep their names: they are per-zone rows
    aligned to ``zones``, and every caller already agrees on those spellings.
    """

    from blade_structure_utilities.geometry_utilities.coordinate_frames import (
        ply_stress_transform_E_to_M,
    )
    from blade_structure_utilities.material_utilities.laminate import LaminateDefinition, _Q_bar_single
    from blade_structure_utilities.material_utilities.orthotropic import plane_stress_Q
    from ..pre_processing.constitutive.panel_abd import panel_abd_tuple_from_lam

    composite_labels: list[str] = []
    material_points: list[ShellRecoveryMaterialPoint] = []
    abd_inv_rows: list[np.ndarray] = []
    basis_rows: list[np.ndarray] = []

    abds = np.asarray(panel_abd, dtype=float) if panel_abd is not None else None
    label_seq = list(panel_labels) if panel_labels is not None else None
    valid_index = 0
    for pi, zone in enumerate(zones):
        s_coords = getattr(zone, "s", None)
        if s_coords is None or len(s_coords) < 2:
            continue
        label = str(getattr(zone, "label", "") or f"panel_{pi}")
        if label_seq is not None:
            if valid_index >= len(label_seq) or label_seq[valid_index] != label:
                raise ValueError("panel_labels ordering does not match zone traversal")
        composite_labels.append(label)
        lam_raw = getattr(zone.lam, "_ld", zone.lam)
        if isinstance(lam_raw, LaminateDefinition):
            plies = lam_raw.build_clpt_ply_rows()
        else:
            plies = list(zone.lam.build_plies())
        if not plies:
            raise ValueError(f"Zone {label!r} has no plies for recovery extraction")
        if abds is not None:
            abd = np.asarray(abds[valid_index], dtype=float)
        else:
            abd, _t, _nu, _E, _G = panel_abd_tuple_from_lam(zone.lam)
        abd_inv_rows.append(np.linalg.inv(abd))
        basis_rows.append(_default_composite_resultant_basis())

        total_t = float(sum(float(ply.t) for ply in plies))
        z_bot = -0.5 * total_t
        ply_count = len(plies)
        for ply_index, ply in enumerate(plies):
            thickness = float(ply.t)
            z_mid = z_bot + 0.5 * thickness
            if isinstance(lam_raw, LaminateDefinition):
                oply, ang = lam_raw.plies[ply_index]
                q_bar = _Q_bar_single(plane_stress_Q(oply), float(ang))
            else:
                q_bar = np.asarray(ply.Q_laminate(), dtype=float)
            # E → M Voigt stress transform; coordinate_frames owns T_sigma.
            transform = ply_stress_transform_E_to_M(float(ply.theta_deg))
            strengths = _ply_strengths_from_shell_ply(ply)
            material_points.append(
                ShellRecoveryMaterialPoint(
                    subcomponent_label=label,
                    ply_index=ply_index,
                    ply_count=ply_count,
                    z_coord=z_mid,
                    q_bar=np.asarray(q_bar, dtype=float),
                    transform=transform,
                    strengths=strengths,
                )
            )
            z_bot += thickness
        valid_index += 1

    n_comp = len(composite_labels)
    if n_comp == 0:
        composite_basis = np.zeros((0, 6, 6), dtype=float)
        abd_inv = np.zeros((0, 6, 6), dtype=float)
    else:
        composite_basis = np.stack(basis_rows, axis=0)
        abd_inv = np.stack(abd_inv_rows, axis=0)

    return ShellRecoverySectionPointSet(
        composite_labels=tuple(composite_labels),
        isotropic_labels=tuple(),
        composite_resultant_basis=composite_basis,
        isotropic_resultant_basis=np.zeros((0, 6, 3), dtype=float),
        abd_inv=abd_inv,
        material_points=tuple(material_points),
        isotropic_points=tuple(),
    )


def build_shell_station_analysis_with_recovery_point_set(
    *,
    geometry: ShellSectionStationGeometry,
    constitutive: BeamSectionConstitutiveInput,
    zones: Sequence[Any],
    provenance: ShellStationProvenance,
    panel_abd: np.ndarray | None = None,
    panel_labels: Sequence[str] | None = None,
    shell_resultants: tuple = (),
    buckling: ShellStationBucklingBundle | None = None,
    recovery_metadata: dict[str, Any] | None = None,
) -> ShellSectionStationAnalysis:
    """Build a station analysis artifact with attached shell-native recovery point set."""

    point_set = extract_shell_laminate_recovery_data(
        zones,
        panel_abd=panel_abd,
        panel_labels=panel_labels,
    )
    return ShellSectionStationAnalysis(
        geometry=geometry,
        constitutive=constitutive,
        shell_resultants=tuple(shell_resultants),
        recovery_point_set=point_set,
        buckling=buckling,
        recovery_metadata=dict(recovery_metadata or {}),
        provenance=provenance,
    )


def _default_composite_resultant_basis() -> np.ndarray:
    basis = np.zeros((6, 6), dtype=float)
    basis[0, 0] = 1.0
    basis[1, 4] = 1.0
    basis[2, 3] = 1.0
    basis[3, 5] = 1.0
    basis[4, 2] = 1.0
    basis[5, 1] = 1.0
    return basis




def _ply_strengths_from_shell_ply(ply: Any) -> dict[str, float]:
    """Hashin allowables off a ply row, or ``{}`` when it carries none.

    An empty result is reported by the caller rather than swallowed here: a
    recovery point with no allowables cannot yield a meaningful failure index,
    and for a long time it produced an index of zero instead — indistinguishable
    from a comfortable margin (``docs/bug_log.md`` BL-011).
    """
    out: dict[str, float] = {}
    for key in ("Xt", "Xc", "Yt", "Yc", "S12"):
        value = getattr(ply, key, None)
        if value is not None and np.isfinite(float(value)):
            out[key] = float(value)
    return out


__all__ = [
    "build_shell_station_analysis_with_recovery_point_set",
    "extract_shell_laminate_recovery_data",
]
