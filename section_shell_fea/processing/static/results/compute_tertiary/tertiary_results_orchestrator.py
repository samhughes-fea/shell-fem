"""Tier 3 orchestrator: CLT recovery (no failure-criterion logic).

CLPT treats each laminate as a **stack of thin plates** (plies). Isotropic panels
are one plate (``n_ply=1``); composites are many plates — the in-plane recovery
algorithm is the same per plate. Each plate stores outer / mid / inner plane
stress (``sig_ply_mat`` shape ``(n_elem, n_ply, 3, 3)``) in ply material frame **M**.

Frames (``geometry_utilities.coordinate_frames``):

* **E** — mid-surface ``eps0``/``kappa``, interface ``tau_iface``, pass-through
  ``Q1``/``Q2`` (= \\(Q_1,Q_2\\))
* **M** — ``eps_ply_mat`` / ``sig_ply_mat`` after ``T_ply`` / ``T_ply_eps``

Consumes only ``ShellSecondaryResultSet`` + ``ShellFormulationCache``; laminates
resolve via ``cache.panel_laminates[row.panel_index]`` (no laminate side argument).
Composes :class:`InPlaneLaminateRecoveryOperator` and
:class:`FSDTInterfaceShearRecoveryOperator`. See PROCESSING_ARCHITECTURE_PROPOSAL.md
tier 3.
"""

from __future__ import annotations

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.formulation_objects import ShellFormulationCache
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import ShellPanelResultants
from ..containers.container_hopper import ShellSecondaryResultSet, ShellTertiaryResultSet
from ..containers.ply_results import ShellPlyResults
from .provenance import interlaminar_inputs_trusted
from .recovery_operators import (
    FSDTInterfaceShearRecoveryOperator,
    InPlaneLaminateRecoveryOperator,
    SHEAR_RECOVERY_MODE_FSDT_PARABOLIC,
    _ply_transverse_g,
)


def _pack_sig_ply_mat_canonical(sig_ply_mat: np.ndarray) -> np.ndarray:
    """Vectorized MITC4-style canonical stress packing from plane stress.

    Layout matches ``CANONICAL_STRESS_NAMES``:
    ``(sigma11, sigma22, sigma33=0, tau23=0, tau13=0, tau12)``.
    """
    out = np.zeros(sig_ply_mat.shape[:-1] + (6,), dtype=np.float64)
    out[..., 0] = sig_ply_mat[..., 0]
    out[..., 1] = sig_ply_mat[..., 1]
    # sigma33, tau23, tau13 remain zero (MITC4 weak-form)
    out[..., 5] = sig_ply_mat[..., 2]
    return out


class ShellTertiaryResultsOrchestrator:
    """CLT plate-stack stress recovery from secondary resultants + formulation cache.

    Each laminate is a stack of plates; recovery evaluates plane stress at each
    plate's outer / mid / inner faces. Rows are grouped by ``panel_index`` because
    the batched ABD solve is per-laminate. Rows failing provenance gating
    (see ``compute_tertiary.provenance``) are left untrusted (zeros) rather than
    raising, matching the fused path's ``needs_review``-style bookkeeping one tier up.
    """

    def __init__(
        self,
        *,
        secondary: ShellSecondaryResultSet,
        cache: ShellFormulationCache,
        in_plane_operator: InPlaneLaminateRecoveryOperator | None = None,
        shear_operator: FSDTInterfaceShearRecoveryOperator | None = None,
    ) -> None:
        self.secondary = secondary
        self.cache = cache
        self.in_plane_operator = in_plane_operator or InPlaneLaminateRecoveryOperator()
        self.shear_operator = shear_operator or FSDTInterfaceShearRecoveryOperator()

    def compute(self) -> ShellTertiaryResultSet:
        rows: list[ShellPanelResultants] = []
        if self.secondary.elemental_results is not None:
            rows = list(self.secondary.elemental_results.resultant_rows)
        n_rows = len(rows)

        by_panel: dict[int, list[int]] = {}
        thickness_m = np.zeros(n_rows, dtype=np.float64)
        element_index = np.zeros(n_rows, dtype=np.int64)
        panel_index_arr = np.zeros(n_rows, dtype=np.int64)
        transverse_shear = np.full((n_rows, 2), np.nan, dtype=np.float64)
        for i, row in enumerate(rows):
            by_panel.setdefault(row.panel_index, []).append(i)
            thickness_m[i] = row.thickness_m
            element_index[i] = row.station_index
            panel_index_arr[i] = row.panel_index
            if row.Q1 is not None and row.Q2 is not None:
                transverse_shear[i, 0] = row.Q1
                transverse_shear[i, 1] = row.Q2

        # First pass: recover per panel to learn max ply / iface counts, then
        # preallocate final arrays and write panel slices directly (WP4).
        panel_laminates = self.cache.panel_laminates
        panel_in_plane: dict[int, object] = {}
        panel_shear_static: dict[int, tuple] = {}
        n_ply_max = 0
        n_iface_max = 0
        for panel_index, idxs in by_panel.items():
            lam = panel_laminates[panel_index] if panel_index < len(panel_laminates) else None
            if lam is None or not getattr(lam, "plies", None):
                continue
            panel_rows = [rows[i] for i in idxs]
            in_plane = self.in_plane_operator.recover_panel(panel_rows, lam)
            panel_in_plane[panel_index] = in_plane
            n_ply_max = max(n_ply_max, in_plane.n_ply)
            n_iface_max = max(n_iface_max, in_plane.n_ply + 1)
            g13_ply, g23_ply = _ply_transverse_g(lam)
            # Panel-static shear stations/coeffs from laminate thickness; rows
            # with a different positive thickness rebuild coeffs for that h.
            lam_h = float(lam.total_thickness())
            if hasattr(self.shear_operator, "panel_static_coefficients"):
                c13, c23, h0, _x3 = self.shear_operator.panel_static_coefficients(
                    lam, thickness_m=lam_h, g13_ply=g13_ply, g23_ply=g23_ply,
                )
                panel_shear_static[panel_index] = (lam, g13_ply, g23_ply, c13, c23, h0)
            else:
                panel_shear_static[panel_index] = (lam, g13_ply, g23_ply, None, None, lam_h)

        eps0 = np.zeros((n_rows, 3), dtype=np.float64)
        kappa = np.zeros((n_rows, 3), dtype=np.float64)
        trusted_mask = np.zeros(n_rows, dtype=bool)
        shear_trusted_mask = np.zeros(n_rows, dtype=bool)
        eps_ply_mat = np.zeros((n_rows, n_ply_max, 3, 3), dtype=np.float64)
        sig_ply_mat = np.zeros((n_rows, n_ply_max, 3, 3), dtype=np.float64)
        tau_iface = np.zeros((n_rows, max(n_iface_max, 0), 2), dtype=np.float64)
        physical_ply_count = np.zeros(n_rows, dtype=np.int32)

        for panel_index, idxs in by_panel.items():
            in_plane = panel_in_plane.get(panel_index)
            if in_plane is None:
                continue
            shear_pack = panel_shear_static.get(panel_index)
            assert shear_pack is not None
            lam, g13_ply, g23_ply, c13, c23, h0 = shear_pack
            for local_i, global_i in enumerate(idxs):
                if in_plane.trusted_mask[local_i]:
                    trusted_mask[global_i] = True
                    eps0[global_i] = in_plane.eps0[local_i]
                    kappa[global_i] = in_plane.kappa[local_i]
                    eps_i = in_plane.eps_ply_mat[local_i]
                    sig_i = in_plane.sig_ply_mat[local_i]
                    if eps_i is not None and sig_i is not None:
                        n_ply_i = int(eps_i.shape[0])
                        physical_ply_count[global_i] = n_ply_i
                        eps_ply_mat[global_i, :n_ply_i] = eps_i
                        sig_ply_mat[global_i, :n_ply_i] = sig_i

                row = rows[global_i]
                if not interlaminar_inputs_trusted(row):
                    continue
                shear_trusted_mask[global_i] = True
                row_h = float(row.thickness_m) if row.thickness_m and row.thickness_m > 0.0 else h0
                if c13 is not None and c23 is not None and abs(row_h - h0) <= 1e-15:
                    tau = self.shear_operator.recover_interfaces(
                        q1=float(row.Q1),
                        q2=float(row.Q2),
                        lam=lam,
                        coeff13=c13,
                        coeff23=c23,
                    )
                else:
                    tau = self.shear_operator.recover_interfaces(
                        q1=float(row.Q1),
                        q2=float(row.Q2),
                        lam=lam,
                        thickness_m=row_h,
                        g13_ply=g13_ply,
                        g23_ply=g23_ply,
                    )
                n_if = int(tau.shape[0])
                tau_iface[global_i, :n_if] = tau

        from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.element_library.registry import (
            resolve_formulation,
        )

        formulation_id = str(self.cache.formulation_id)
        projector = resolve_formulation(formulation_id).projector
        caps = np.array(projector.project_material_stress(0.0, 0.0, 0.0)[1], dtype=object)
        # Vectorized canonical packing (exact MITC4 layout; sigma33/tau23/tau13 = 0).
        sig_canonical = _pack_sig_ply_mat_canonical(sig_ply_mat)

        ply_results = ShellPlyResults(
            eps0=eps0,
            kappa=kappa,
            eps_ply_mat=eps_ply_mat,
            sig_ply_mat=sig_ply_mat,
            trusted_mask=trusted_mask,
            tau_iface=tau_iface if n_iface_max > 0 else None,
            shear_trusted_mask=shear_trusted_mask,
            sig_ply_mat_canonical=sig_canonical,
            stress_capability=caps,
            physical_ply_count=physical_ply_count,
            formulation_id=np.full(n_rows, formulation_id, dtype=object),
            element_key=np.column_stack(
                [
                    element_index,
                    panel_index_arr,
                    np.arange(n_rows, dtype=np.int64),
                ]
            ),
        )
        return ShellTertiaryResultSet(
            ply_results=ply_results,
            transverse_shear=transverse_shear,
            thickness_m=thickness_m,
            element_index=element_index,
            panel_index=panel_index_arr,
            meta={
                "n_rows": n_rows,
                "n_ply": n_ply_max,
                "n_iface": n_iface_max,
                "shear_recovery_mode": SHEAR_RECOVERY_MODE_FSDT_PARABOLIC,
                "formulation_id": formulation_id,
                "analysis_dof_layout_id": str(self.cache.analysis_dof_layout_id),
                "mid_surface_frame": ply_results.mid_surface_frame,
                "ply_material_frame": ply_results.ply_material_frame,
            },
        )


__all__ = ["ShellTertiaryResultsOrchestrator"]
