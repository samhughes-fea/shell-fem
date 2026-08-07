"""Theory-resolved element factories and formulation-cache builder."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator

import numpy as np
from numpy.typing import NDArray

from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.containers.formulation_objects import (
    PanelConstitutiveKind,
    ShellFormulationCache,
)
from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.containers.types import (
    ShellTheory,
    resolve_shell_theory_capability,
)
from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.constitutive.panel_abd import (
    panel_abd_tuple_from_lam,
)
from ..mesh.shell_mesh_map import PanelGlobalMap
from .mitc4.mitc4_quad4_element import (
    Mitc4ElementGeometry,
    Mitc4ElementLoads,
    Mitc4ElementMaterial,
    Mitc4Quad4Element,
)
from .shell_element_base import ShellElementBase
from blade_structure_utilities.fem_utilities.section_shell_model.processing.static.diagnostics.solve_phase_timing import (
    PHASE_ELEMENT_FORCE_VECTORS,
    PHASE_ELEMENT_STIFFNESS_MATRICES,
    PhaseCollector,
    _NULL_PHASE_COLLECTOR,
)


def _panel_kappa_nodes(nodes_yz: NDArray[np.float64], s_panel_full: NDArray[np.float64], s_nodes: NDArray[np.float64]) -> NDArray[np.float64] | None:
    if len(nodes_yz) < 3:
        return None
    y_p, z_p = nodes_yz[:, 0], nodes_yz[:, 1]
    dy_p, dz_p = np.diff(y_p), np.diff(z_p)
    ds_p = np.maximum(np.hypot(dy_p, dz_p), 1e-30)
    theta_p = np.arctan2(dz_p, dy_p)
    kappa_p = np.zeros(len(nodes_yz))
    for ii in range(1, len(nodes_yz) - 1):
        dth = theta_p[ii] - theta_p[ii - 1]
        dth = (dth + np.pi) % (2 * np.pi) - np.pi
        arc = 0.5 * (ds_p[ii - 1] + ds_p[ii])
        kappa_p[ii] = dth / arc if arc > 1e-30 else 0.0
    if len(kappa_p) > 1:
        kappa_p[0] = kappa_p[1]
        kappa_p[-1] = kappa_p[-2]
    return np.interp(s_nodes, s_panel_full, kappa_p)


def _element_L_s(
    pi: int,
    s_nodes: NDArray[np.float64],
    i0: int,
    i1: int,
    panels: list[Any],
    panels_outboard: list[Any] | None,
    L_x: float,
) -> tuple[float, float]:
    L_s = float(abs(s_nodes[i1] - s_nodes[i0]))
    L_x_elem = L_x
    if panels_outboard is not None:
        p = panels[pi]
        p_out = panels_outboard[pi]
        s_panel_out = np.asarray(p_out.s, dtype=float)
        s0_in = float(np.asarray(p.s, dtype=float)[0])
        s1_in = float(np.asarray(p.s, dtype=float)[-1])
        if abs(s1_in - s0_in) <= 1e-30:
            xi0 = 0.0
            xi1 = 0.0
        else:
            xi0 = float((s_nodes[i0] - s0_in) / (s1_in - s0_in))
            xi1 = float((s_nodes[i1] - s0_in) / (s1_in - s0_in))
        s_out0 = float(s_panel_out[0] + xi0 * (s_panel_out[-1] - s_panel_out[0]))
        s_out1 = float(s_panel_out[0] + xi1 * (s_panel_out[-1] - s_panel_out[0]))
        L_s_out = abs(s_out1 - s_out0)
        L_s = 0.5 * (L_s + L_s_out)
    return L_s, L_x_elem


class ShellElementFactory(ABC):
    theory: ShellTheory
    n_dof_node: int

    @abstractmethod
    def iter_elements(
        self,
        panel_maps: list[PanelGlobalMap],
        panels: list[Any],
        Nx_panels: list[NDArray[np.float64]],
        Nxy_panels: list[NDArray[np.float64]],
        *,
        L_x: float = 1.0,
        panels_outboard: list[Any] | None = None,
    ) -> Iterator[ShellElementBase]:
        ...


class Mitc4ElementFactory(ShellElementFactory):
    theory = ShellTheory.FSDT
    n_dof_node = 6

    def iter_elements(
        self,
        panel_maps: list[PanelGlobalMap],
        panels: list[Any],
        Nx_panels: list[NDArray[np.float64]],
        Nxy_panels: list[NDArray[np.float64]],
        *,
        L_x: float = 1.0,
        panels_outboard: list[Any] | None = None,
        panel_abd_stacks: list[tuple] | None = None,
    ) -> Iterator[Mitc4Quad4Element]:
        elem_id = 0
        for pi, pm in enumerate(panel_maps):
            if len(pm.s_nodes) == 0:
                continue
            p = panels[pi]
            nodes_yz = np.asarray(getattr(p, "nodes"), dtype=float)
            s_nodes = pm.s_nodes
            s_panel_full = np.asarray(p.s, dtype=float)
            Nx_nodes = np.interp(s_nodes, s_panel_full, np.asarray(Nx_panels[pi], dtype=float))
            Nxy_nodes = np.interp(s_nodes, s_panel_full, np.asarray(Nxy_panels[pi], dtype=float))
            kappa_nodes = _panel_kappa_nodes(nodes_yz, s_panel_full, s_nodes)
            if panel_abd_stacks is not None:
                ABD, thickness, _, _, G_eff = panel_abd_stacks[pi]
            else:
                ABD, thickness, _, _, G_eff = panel_abd_tuple_from_lam(p.lam)

            for ei, e in enumerate(pm.elements):
                i0, i1, i2, i3 = e
                L_s_plain = float(abs(s_nodes[i1] - s_nodes[i0]))
                L_s, L_x_elem = _element_L_s(pi, s_nodes, i0, i1, panels, panels_outboard, L_x)
                if L_s < 1e-30:
                    continue
                nx_avg = 0.5 * (Nx_nodes[i0] + Nx_nodes[i1])
                nxy_avg = 0.5 * (Nxy_nodes[i0] + Nxy_nodes[i1])
                kappa_avg = None
                if kappa_nodes is not None:
                    kappa_avg = float(0.5 * (kappa_nodes[i0] + kappa_nodes[i1]))
                yield Mitc4Quad4Element(
                    element_id=elem_id,
                    panel_index=pi,
                    local_element_index=ei,
                    geometry=Mitc4ElementGeometry(
                        L_s=L_s,
                        L_s_donnell=L_s_plain,
                        L_x=L_x,
                        local_nodes=(i0, i1, i2, i3),
                    ),
                    material=Mitc4ElementMaterial(ABD=ABD, thickness=thickness, G_eff=G_eff),
                    loads=Mitc4ElementLoads(nx_avg=nx_avg, nxy_avg=nxy_avg, kappa_avg=kappa_avg),
                    dof_per_node=self.n_dof_node,
                )
                elem_id += 1


_ELEMENT_FACTORY_REGISTRY: dict[ShellTheory, type[ShellElementFactory]] = {}


def register_element_factory(theory: ShellTheory, factory_cls: type[ShellElementFactory]) -> None:
    """Register ``factory_cls`` as the element factory for ``theory``.

    The direct analogue of 1d-fem's ``ElementFactory.ELEMENT_CLASS_MAP``. FSDT/MITC4
    registers unconditionally at import time below (the only production default
    theory); CalculiX registers only via its opt-in
    ``register_calculix_ccx_backend()`` hook, keeping production dispatch free of
    eager imports for non-default theories (see
    ``docs/ELEMENT_AGNOSTIC_ARCHITECTURE.md``).
    """
    _ELEMENT_FACTORY_REGISTRY[theory] = factory_cls


def resolve_element_factory(shell_theory: ShellTheory) -> ShellElementFactory:
    resolve_shell_theory_capability(shell_theory)
    try:
        factory_cls = _ELEMENT_FACTORY_REGISTRY[shell_theory]
    except KeyError:
        raise ValueError(f"No element factory registered for {shell_theory!r}") from None
    return factory_cls()


register_element_factory(ShellTheory.FSDT, Mitc4ElementFactory)


def _panel_load_totals_from_forces(
    panel_maps: list[PanelGlobalMap],
    force_objects: list,
    *,
    n_dof_node: int,
) -> list[dict[str, float]]:
    totals: list[dict[str, float]] = []
    _U_X = 0
    _U_S = 1
    for pi, pm in enumerate(panel_maps):
        if len(pm.s_nodes) == 0:
            totals.append({})
            continue
        fx = fs = 0.0
        for fo in force_objects:
            if fo.panel_index != pi:
                continue
            fe = fo.F_e
            for node_slot in (2, 3):
                fx += fe[node_slot * n_dof_node + _U_X]
                fs += fe[node_slot * n_dof_node + _U_S]
        totals.append({"Fx_total": fx, "Fs_total": fs})
    return totals


def _panel_abd_stacks(panel_maps: list[PanelGlobalMap], panels: list[Any]) -> list[tuple]:
    from blade_structure_utilities.material_utilities.isotropic import IsotropicMaterial

    stacks: list[tuple] = []
    for pi, pm in enumerate(panel_maps):
        if len(pm.s_nodes) == 0:
            stacks.append((np.zeros((6, 6)), 0.0, 0.0, 0.0, 0.0))
            continue
        lam = panels[pi].lam
        if isinstance(lam, IsotropicMaterial):
            ld = _isotropic_to_laminate_definition(lam, panels[pi])
            ABD = np.asarray(ld.build_ABD(), dtype=np.float64)
            t = float(ld.total_thickness())
            e = float(lam.E)
            nu = float(lam.nu)
            g_eff = e / (2.0 * (1.0 + nu))
            stacks.append((ABD, t, nu, e, g_eff))
            continue
        stacks.append(panel_abd_tuple_from_lam(lam))
    return stacks


def _panel_laminates(panel_maps: list[PanelGlobalMap], panels: list[Any]) -> list[Any]:
    """Resolve each panel's laminate to a full ``LaminateDefinition`` for post-solve reuse.

    Mirrors the object resolution already done inside ``panel_abd_tuple_from_lam``
    (``stiffness_bundle()`` / ``build_ABD()`` / legacy conversion), but retains the
    laminate object itself rather than collapsing it to an ABD tuple, so tiers 3/4
    (CLT ply recovery, failure indices) can resolve ``cache.panel_laminates[panel_index]``
    without a laminate side argument.
    """
    from blade_structure_utilities.material_utilities.laminate import (
        LaminateDefinition,
        laminate_definition_from_legacy,
    )
    from blade_structure_utilities.material_utilities.isotropic import IsotropicMaterial

    laminates: list[Any] = []
    for pi, pm in enumerate(panel_maps):
        if len(pm.s_nodes) == 0:
            laminates.append(None)
            continue
        lam = panels[pi].lam
        if isinstance(lam, IsotropicMaterial):
            # FE stiffness still needs a laminate-equivalent; isotropic FI uses
            # panel_constitutive_kind / panel_sigma_allow_pa, not Hashin strengths.
            laminates.append(_isotropic_to_laminate_definition(lam, panels[pi]))
            continue
        if isinstance(lam, LaminateDefinition):
            laminates.append(lam)
            continue
        laminates.append(laminate_definition_from_legacy(lam))
    return laminates


def _isotropic_to_laminate_definition(mat: Any, panel: Any) -> Any:
    """Single-ply isotropic-equivalent ``LaminateDefinition`` for CLT recovery."""
    from blade_structure_utilities.material_utilities.laminate import LaminateDefinition
    from blade_structure_utilities.material_utilities.orthotropic import OrthotropicPly

    t = float(getattr(panel, "t", 0.0) or 0.0)
    if t <= 0.0:
        t = float(getattr(getattr(panel, "lam", None), "t", 0.0) or 1e-3)
    e = float(mat.E)
    nu = float(mat.nu)
    g = e / (2.0 * (1.0 + nu))
    sa = float(mat.sigma_allow)
    ply = OrthotropicPly(
        name=str(getattr(mat, "name", "isotropic")),
        E1=e,
        E2=e,
        G12=g,
        nu12=nu,
        rho=float(mat.rho),
        t_ply=t,
        Xt=sa,
        Xc=sa,
        Yt=sa,
        Yc=sa,
        S12=sa,
        Zt=sa,
        S13=sa,
        S23=sa,
    )
    return LaminateDefinition(plies=[(ply, 0.0)])


def _panel_constitutive_metadata(
    panel_maps: list[PanelGlobalMap],
    panels: list[Any],
) -> tuple[list[PanelConstitutiveKind], list[float]]:
    """Per-panel isotropic/orthotropic kind and Von Mises allowable [Pa].

    Two ways a panel can be classified isotropic: a bare
    ``material_utilities.isotropic.IsotropicMaterial`` (never constructed in
    production today -- kept for compatibility, not relied upon), or a
    ``PanelLaminateAdapter`` built by
    ``materials_adapter._isotropic_ply`` and marked ``is_isotropic_source``
    (the actual production path). Either way routes to von Mises FI.
    """
    from blade_structure_utilities.material_utilities.isotropic import IsotropicMaterial

    kinds: list[PanelConstitutiveKind] = []
    allows: list[float] = []
    for pi, pm in enumerate(panel_maps):
        if len(pm.s_nodes) == 0:
            kinds.append("orthotropic")
            allows.append(float("nan"))
            continue
        lam = panels[pi].lam
        if isinstance(lam, IsotropicMaterial) or getattr(lam, "is_isotropic_source", False):
            kinds.append("isotropic")
            sigma_allow = getattr(lam, "sigma_allow", None)
            if sigma_allow is None:
                sigma_allow = getattr(lam, "sigma_allow_pa", None)
            allows.append(float(sigma_allow) if sigma_allow is not None else float("nan"))
            continue
        kinds.append("orthotropic")
        allows.append(float("nan"))
    return kinds, allows


def fill_global_dof_indices(
    cache: ShellFormulationCache,
    panel_maps: list[PanelGlobalMap],
) -> None:
    n = cache.n_dof_node
    for stiff, force in zip(cache.stiffness_objects, cache.force_objects):
        pm = panel_maps[stiff.panel_index]
        i0, i1, i2, i3 = pm.elements[stiff.local_element_index]
        gids = [pm.global_nodes[ln] for ln in (i0, i1, i2, i3)]
        g_dofs = np.empty(4 * n, dtype=np.int64)
        for i, gn in enumerate(gids):
            base = i * n
            g_dofs[base : base + n] = gn * n + np.arange(n, dtype=np.int64)
        stiff.global_dof_indices = g_dofs
        force.global_dof_indices = g_dofs


def build_formulation_cache(
    panel_maps: list[PanelGlobalMap],
    panels: list[Any],
    Nx_panels: list[NDArray[np.float64]],
    Nxy_panels: list[NDArray[np.float64]],
    *,
    shell_theory: ShellTheory = ShellTheory.FSDT,
    L_x: float = 1.0,
    panels_outboard: list[Any] | None = None,
    phase_collector: PhaseCollector | None = None,
) -> ShellFormulationCache:
    factory = resolve_element_factory(shell_theory)
    panel_abd_stacks = _panel_abd_stacks(panel_maps, panels)
    panel_laminates = _panel_laminates(panel_maps, panels)
    panel_constitutive_kind, panel_sigma_allow_pa = _panel_constitutive_metadata(
        panel_maps, panels
    )
    # Materialised (not a single-pass generator) so stiffness and force build
    # each get their own aggregated phase timing, one start/end pair per
    # build_formulation_cache call rather than per element.
    elements = list(
        factory.iter_elements(
            panel_maps,
            panels,
            Nx_panels,
            Nxy_panels,
            L_x=L_x,
            panels_outboard=panels_outboard,
            panel_abd_stacks=panel_abd_stacks,
        )
    )
    collector = phase_collector if phase_collector is not None else _NULL_PHASE_COLLECTOR
    with collector.phase(PHASE_ELEMENT_STIFFNESS_MATRICES):
        stiff = [elem.element_stiffness_matrix() for elem in elements]
    with collector.phase(PHASE_ELEMENT_FORCE_VECTORS):
        force = [elem.element_force_vector() for elem in elements]
    cache = ShellFormulationCache(
        stiffness_objects=stiff,
        force_objects=force,
        n_dof_node=factory.n_dof_node,
        panel_load_totals=_panel_load_totals_from_forces(
            panel_maps, force, n_dof_node=factory.n_dof_node
        ),
        panel_abd_stacks=panel_abd_stacks,
        panel_laminates=panel_laminates,
        panel_constitutive_kind=panel_constitutive_kind,
        panel_sigma_allow_pa=panel_sigma_allow_pa,
    )
    fill_global_dof_indices(cache, panel_maps)
    from .formulation_validation import validate_formulation_cache_shapes

    validate_formulation_cache_shapes(cache)
    return cache


__all__ = [
    "Mitc4ElementFactory",
    "ShellElementFactory",
    "build_formulation_cache",
    "fill_global_dof_indices",
    "register_element_factory",
    "resolve_element_factory",
]
