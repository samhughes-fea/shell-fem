"""Secondary (**E**-frame resultant) recovery from formulation cache + ``U_e``.

``U_e`` is in curvilinear **C** (≡ **E** on each facet). Recovered
``ShellPanelResultants`` use legacy ``N11…``/``Q1`` names for
:math:`N_{11}\\ldots` / :math:`Q_1,Q_2`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.formulation_objects import ShellFormulationCache
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import FieldProvenance, ProvenanceKind, ShellPanelResultants, ShellTheory
from .....pre_processing.element_library.kernel import resolve_kernel, resolve_kernel_by_id
from ..containers.container_hopper import ShellPrimaryResultSet, ShellSecondaryResultSet
from ..containers.elemental_results import ShellElementalResults


class ShellSecondaryResultsOrchestrator:
    """Recover shell resultants per element from tier 1's disassembled ``U_e``.

    Matches the ``ShellPrimaryResultsOrchestrator`` / ``ShellTertiaryResultsOrchestrator``
    class-style pattern so ``SectionLinearStaticRunner.compute_secondary_results()``
    instantiates it uniformly with the upstream result set + cache.

    Optional Gaussian / nodal resolutions (R10) are off by default so the FI hot
    path stays unaffected.
    """

    def __init__(
        self,
        *,
        primary: ShellPrimaryResultSet,
        cache: ShellFormulationCache,
        L_x: float = 1.0,
        theory: str | None = None,
        panel_meta: list[dict[str, Any]] | None = None,
        panel_maps: list[Any] | None = None,
        zones: list[Any] | None = None,
        provenance_tag: str = "processing.compute_secondary",
        compute_gaussian: bool = False,
        compute_nodal: bool = False,
    ) -> None:
        self.primary = primary
        self.cache = cache
        self.L_x = L_x
        self.theory = theory
        self.panel_meta = panel_meta
        self.panel_maps = panel_maps
        self.zones = zones
        self.provenance_tag = provenance_tag
        self.compute_gaussian = bool(compute_gaussian)
        self.compute_nodal = bool(compute_nodal)

    def compute(self) -> ShellSecondaryResultSet:
        U_e = self.primary.elemental_results.U_e if self.primary.elemental_results else []
        secondary = _compute_secondary_resultants(
            self.cache,
            U_e,
            L_x=self.L_x,
            theory=self.theory,
            panel_meta=self.panel_meta,
            panel_maps=self.panel_maps,
            zones=self.zones,
            provenance_tag=self.provenance_tag,
        )
        if not self.compute_gaussian and not self.compute_nodal:
            return secondary
        from .compute_gaussian_fields import compute_gaussian_resultants
        from .compute_nodal_fields import compute_nodal_resultants

        gaussian = None
        nodal = None
        if self.compute_gaussian or self.compute_nodal:
            gaussian = compute_gaussian_resultants(
                self.cache,
                U_e,
                L_x=self.L_x,
                panel_maps=self.panel_maps,
                shell_theory=self.theory,
            )
        if self.compute_nodal and gaussian is not None:
            nodal = compute_nodal_resultants(
                gaussian, self.cache, panel_maps=self.panel_maps,
            )
        return ShellSecondaryResultSet(
            elemental_results=secondary.elemental_results,
            gaussian_results=gaussian if self.compute_gaussian else None,
            nodal_results=nodal,
        )


def _element_L_s(
    *,
    pi: int,
    local_element_index: int,
    panel_maps: list[Any] | None,
    panel_meta: list[dict[str, Any]] | None,
) -> float:
    if panel_maps is not None and 0 <= pi < len(panel_maps):
        pm = panel_maps[pi]
        elems = getattr(pm, "elements", None) or []
        if 0 <= local_element_index < len(elems):
            i0, i1, _i2, _i3 = elems[local_element_index]
            s_nodes = np.asarray(pm.s_nodes, dtype=float)
            L_s = float(abs(s_nodes[i1] - s_nodes[i0]))
            if L_s >= 1e-30:
                return L_s
    meta = panel_meta or []
    if 0 <= pi < len(meta):
        pm_meta = meta[pi]
        elem_lengths = pm_meta.get("element_L_s")
        if elem_lengths is not None and 0 <= local_element_index < len(elem_lengths):
            return float(elem_lengths[local_element_index])
        return float(pm_meta.get("L_s", 1.0))
    return 1.0


def _resolve_secondary_kernel(theory: str | ShellTheory | None, cache: ShellFormulationCache):
    """Resolve the recovery kernel, preferring the cache's own ``formulation_id``.

    ``theory is None`` no longer means the unsafe ``"mitc4"`` string default: the
    cache's authoritative ``formulation_id`` selects the kernel (WP2); absent that
    it falls back to ``ShellTheory.FSDT``. An explicit ``theory`` is honoured and
    mapped to :class:`ShellTheory` safely.
    """
    if theory is None:
        fid = str(getattr(cache, "formulation_id", "") or "")
        if fid:
            return resolve_kernel_by_id(fid)
        return resolve_kernel(ShellTheory.FSDT)
    if isinstance(theory, ShellTheory):
        return resolve_kernel(theory)
    token = str(theory).strip().lower()
    if token in ("mitc4", "mitc4-fsdt", "fsdt"):
        return resolve_kernel(ShellTheory.FSDT)
    try:
        return resolve_kernel(ShellTheory(token))
    except ValueError:
        return resolve_kernel(ShellTheory.FSDT)


def _compute_secondary_resultants(
    cache: ShellFormulationCache,
    U_e: list,
    *,
    L_x: float = 1.0,
    theory: str | None = None,
    panel_meta: list[dict[str, Any]] | None = None,
    panel_maps: list[Any] | None = None,
    zones: list[Any] | None = None,
    provenance_tag: str = "processing.compute_secondary",
) -> ShellSecondaryResultSet:
    kernel = _resolve_secondary_kernel(theory, cache)
    rows: list[ShellPanelResultants] = []
    for stiff, u_elem in zip(cache.stiffness_objects, U_e):
        pi = stiff.panel_index
        local_ei = int(stiff.local_element_index)
        # WP2: prefer the persisted assembly length (blended outboard L_s) so
        # recovery agrees with the stiffness that produced ``U_e``; fall back to
        # recomputed plain contour length only when the cache predates WP2.
        L_s = float(getattr(stiff, "L_s", float("nan")))
        if not math.isfinite(L_s) or L_s < 1e-30:
            L_s = _element_L_s(
                pi=pi,
                local_element_index=local_ei,
                panel_maps=panel_maps,
                panel_meta=panel_meta,
            )
        if L_s < 1e-30:
            continue
        panel_spec = (
            cache.panel_constitutive_specs[pi]
            if pi < len(cache.panel_constitutive_specs)
            else None
        )
        if panel_spec is not None and hasattr(kernel, "recover_generalized"):
            res = kernel.recover_generalized(
                u_elem, L_s, L_x, panel_spec.fsdt
            )
            thickness = float(panel_spec.thickness_m)
            Q1, Q2 = float(res["Q1"]), float(res["Q2"])
        else:
            abd_stack = (
                cache.panel_abd_stacks[pi]
                if pi < len(cache.panel_abd_stacks)
                else None
            )
            if abd_stack is None:
                continue
            ABD = abd_stack[0]
            thickness = float(abd_stack[1])
            G_eff = float(abd_stack[4])
            res = kernel.recover_resultants(u_elem, L_s, L_x, ABD)
            Q1, Q2 = kernel.recover_shear(
                u_elem, L_s, L_x, G_eff=G_eff, thickness=thickness,
            )
        prov_kind = kernel.provenance_kind
        if Q1 is None:
            prov = {
                k: FieldProvenance(prov_kind, provenance_tag)
                for k in ("N11", "N22", "N12", "M11", "M22", "M12")
            }
            prov["Q1"] = FieldProvenance(ProvenanceKind.RESERVED, "FSDT / G_eff or thickness unavailable")
            prov["Q2"] = FieldProvenance(ProvenanceKind.RESERVED, "FSDT / G_eff or thickness unavailable")
        else:
            prov = {
                k: FieldProvenance(prov_kind, provenance_tag)
                for k in ("N11", "N22", "N12", "M11", "M22", "M12", "Q1", "Q2")
            }
        panel_label = ""
        if panel_maps is not None and 0 <= pi < len(panel_maps):
            panel_label = str(getattr(panel_maps[pi], "panel_label", "") or "")
        elif zones is not None and 0 <= pi < len(zones):
            panel_label = str(getattr(zones[pi], "label", "") or "")
        nx = float(res["N11"])
        nxy = float(res["N12"])
        rows.append(
            ShellPanelResultants(
                N11=nx,
                N22=float(res["N22"]),
                N12=nxy,
                M11=float(res["M11"]),
                M22=float(res["M22"]),
                M12=float(res["M12"]),
                Q1=Q1,
                Q2=Q2,
                provenance=prov,
                sigma_11_pa=nx / max(thickness, 1e-30),
                tau_12_pa=nxy,
                q_n_per_m=nxy,
                thickness_m=thickness,
                panel_label=panel_label,
                panel_index=pi,
                station_index=local_ei,
            )
        )
    return ShellSecondaryResultSet(
        elemental_results=ShellElementalResults(resultant_rows=rows),
    )


__all__ = ["ShellSecondaryResultsOrchestrator"]
