"""Theory-resolved element factories and formulation-cache builder."""

from __future__ import annotations

import inspect
import math
from abc import ABC, abstractmethod
from typing import Any, Iterator

import numpy as np
from numpy.typing import NDArray

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.formulation_objects import (
    PanelConstitutiveKind,
    PanelConstitutiveSpec,
    ShellFormulationCache,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import (
    ShellTheory,
    resolve_shell_theory_capability,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.constitutive.panel_abd import (
    panel_abd_tuple_from_lam,
    panel_constitutive_spec_from_lam,
)
from ..mesh.panel_node_assembly import PanelNodeAssemblyMap
from .shell_element_base import ShellElementBase
from .phase_timing import (
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
    zones: list[Any],
    zones_outboard: list[Any] | None,
    L_x: float,
) -> tuple[float, float]:
    L_s = float(abs(s_nodes[i1] - s_nodes[i0]))
    L_x_elem = L_x
    if zones_outboard is not None:
        p = zones[pi]
        p_out = zones_outboard[pi]
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
        panel_maps: list[PanelNodeAssemblyMap],
        zones: list[Any],
        Nx_panels: list[NDArray[np.float64]],
        Nxy_panels: list[NDArray[np.float64]],
        *,
        L_x: float = 1.0,
        zones_outboard: list[Any] | None = None,
    ) -> Iterator[ShellElementBase]:
        ...


def _factory_accepts_template_cache(factory: ShellElementFactory) -> bool:
    """Whether ``factory.iter_elements`` opts into the per-build template cache."""
    try:
        params = inspect.signature(factory.iter_elements).parameters
    except (TypeError, ValueError):
        return False
    return "template_cache" in params


def _formulation_id_from_factory(factory: ShellElementFactory) -> str:
    """Prefer the factory's own id; otherwise the registry default for its theory."""
    fid = getattr(factory, "formulation_id", None)
    if fid:
        return str(fid)
    from .registry import default_formulation_id

    return default_formulation_id(factory.theory)


def resolve_element_factory(
    shell_theory: ShellTheory = ShellTheory.FSDT,
    *,
    formulation_id: str | None = None,
) -> ShellElementFactory:
    """Resolve the element factory from the unified ``formulation_id`` graph.

    ``formulation_id`` is primary. Otherwise ``default_formulation_id(shell_theory)``
    selects the native registration. Unregistered native ids / theories raise
    ``ValueError``. External backends (CalculiX) use ``shell_solve_registry``,
    not this path.
    """
    from .registry import default_formulation_id, ensure_default_formulations, resolve_formulation

    ensure_default_formulations()
    if formulation_id is not None:
        return resolve_formulation(formulation_id).factory_cls()
    resolve_shell_theory_capability(shell_theory)
    resolved_id = default_formulation_id(shell_theory)
    return resolve_formulation(resolved_id).factory_cls()


def __getattr__(name: str) -> Any:
    if name == "Mitc4ElementFactory":
        from .mitc4.factory import Mitc4ElementFactory as _Factory

        return _Factory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _panel_load_totals_from_forces(
    panel_maps: list[PanelNodeAssemblyMap],
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


def _panel_abd_stacks(panel_maps: list[PanelNodeAssemblyMap], zones: list[Any]) -> list[tuple]:
    from blade_structure_utilities.material_utilities.isotropic import IsotropicMaterial

    stacks: list[tuple] = []
    for pi, pm in enumerate(panel_maps):
        if len(pm.s_nodes) == 0:
            stacks.append((np.zeros((6, 6)), 0.0, 0.0, 0.0, 0.0))
            continue
        lam = zones[pi].lam
        if isinstance(lam, IsotropicMaterial):
            ld = _isotropic_to_laminate_definition(lam, zones[pi])
            ABD = np.asarray(ld.build_ABD(), dtype=np.float64)
            t = float(ld.total_thickness())
            e = float(lam.E)
            nu = float(lam.nu)
            g_eff = e / (2.0 * (1.0 + nu))
            stacks.append((ABD, t, nu, e, g_eff))
            continue
        stacks.append(panel_abd_tuple_from_lam(lam))
    return stacks


def _panel_laminates(panel_maps: list[PanelNodeAssemblyMap], zones: list[Any]) -> list[Any]:
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
        lam = zones[pi].lam
        if isinstance(lam, IsotropicMaterial):
            # FE stiffness still needs a laminate-equivalent; isotropic FI uses
            # panel_constitutive_kind / panel_sigma_y_pa, not Hashin strengths.
            laminates.append(_isotropic_to_laminate_definition(lam, zones[pi]))
            continue
        if isinstance(lam, LaminateDefinition):
            laminates.append(lam)
            continue
        laminates.append(laminate_definition_from_legacy(lam))
    return laminates


def _isotropic_to_laminate_definition(mat: Any, zone: Any) -> Any:
    """Single-ply isotropic-equivalent ``LaminateDefinition`` for CLT recovery."""
    from blade_structure_utilities.material_utilities.laminate import LaminateDefinition
    from blade_structure_utilities.material_utilities.orthotropic import OrthotropicPly

    t = float(getattr(zone, "t", 0.0) or 0.0)
    if t <= 0.0:
        t = float(getattr(getattr(zone, "lam", None), "t", 0.0) or 0.0)
    if not np.isfinite(t) or t <= 0.0:
        raise ValueError(
            "isotropic zone/laminate thickness missing or ≤0; refusing silent "
            f"t=1e-3 default (zone label={getattr(zone, 'label', None)!r}; FEA-T-001)"
        )
    e = float(mat.E)
    nu = float(mat.nu)
    g = e / (2.0 * (1.0 + nu))
    sy = float(
        getattr(mat, "sigma_y", None)
        or getattr(mat, "yield_strength", None)
        or getattr(mat, "sigma_allow", None)
        or 0.0
    )
    ply = OrthotropicPly(
        name=str(getattr(mat, "name", "isotropic")),
        E1=e,
        E2=e,
        G12=g,
        nu12=nu,
        rho=float(mat.rho),
        t_ply=t,
        Xt=sy,
        Xc=sy,
        Yt=sy,
        Yc=sy,
        S12=sy,
        Zt=sy,
        S13=sy,
        S23=sy,
    )
    return LaminateDefinition(plies=[(ply, 0.0)])


def _finite_or_nan(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) and v > 0.0 else float("nan")


def _strength_from_obj(obj: Any, *names: str) -> float:
    for name in names:
        if hasattr(obj, name):
            v = _finite_or_nan(getattr(obj, name))
            if math.isfinite(v):
                return v
    return float("nan")


def _panel_strength_fields(lam: Any) -> tuple[float, float, float, float]:
    """Return (σ_y, σ_uts, σ_ucs, τ_us) [Pa] from panel laminate / material attrs.

    ``σ_y`` prefers ``sigma_y_pa`` / ``sigma_y`` / ``yield_strength``. Legacy
    material field ``sigma_allow`` is used only when those are absent (bridge).
    """
    sigma_y = _strength_from_obj(lam, "sigma_y_pa", "sigma_y", "yield_strength")
    if not math.isfinite(sigma_y):
        # Material-field bridge only — not a panel-cache ``*_allow*`` alias.
        sigma_y = _strength_from_obj(lam, "sigma_allow")
    sigma_uts = _strength_from_obj(
        lam, "sigma_uts_pa", "sigma_uts", "sigma_ult_tension", "sigma_ult"
    )
    sigma_ucs = _strength_from_obj(
        lam, "sigma_ucs_pa", "sigma_ucs", "sigma_ult_compression", "sigma_ult", "sigma_utc"
    )
    tau_us = _strength_from_obj(lam, "tau_us_pa", "tau_us", "tau_ult")
    return sigma_y, sigma_uts, sigma_ucs, tau_us


def _panel_constitutive_metadata(
    panel_maps: list[PanelNodeAssemblyMap],
    zones: list[Any],
) -> tuple[list[PanelConstitutiveKind], list[float], list[float], list[float], list[float]]:
    """Per-panel kind + σ_y / σ_uts / σ_ucs / τ_us [Pa].

    Two ways a panel can be classified isotropic: a bare
    ``material_utilities.isotropic.IsotropicMaterial``, or a
    ``ZoneLaminateAdapter`` marked ``is_isotropic_source``.
    """
    from blade_structure_utilities.material_utilities.isotropic import IsotropicMaterial

    kinds: list[PanelConstitutiveKind] = []
    sigma_y_list: list[float] = []
    sigma_uts_list: list[float] = []
    sigma_ucs_list: list[float] = []
    tau_us_list: list[float] = []
    for pi, pm in enumerate(panel_maps):
        if len(pm.s_nodes) == 0:
            kinds.append("orthotropic")
            sigma_y_list.append(float("nan"))
            sigma_uts_list.append(float("nan"))
            sigma_ucs_list.append(float("nan"))
            tau_us_list.append(float("nan"))
            continue
        lam = zones[pi].lam
        sy, uts, ucs, tus = _panel_strength_fields(lam)
        if isinstance(lam, IsotropicMaterial) or getattr(lam, "is_isotropic_source", False):
            kinds.append("isotropic")
        else:
            kinds.append("orthotropic")
        sigma_y_list.append(sy)
        sigma_uts_list.append(uts)
        sigma_ucs_list.append(ucs)
        tau_us_list.append(tus)
    return kinds, sigma_y_list, sigma_uts_list, sigma_ucs_list, tau_us_list


def _typed_panel_constitutive_specs(
    panel_maps: list[PanelNodeAssemblyMap],
    panel_laminates: list[Any],
    kinds: list[PanelConstitutiveKind],
    sigma_y_pa: list[float],
    sigma_uts_pa: list[float],
    sigma_ucs_pa: list[float],
    tau_us_pa: list[float],
) -> list[PanelConstitutiveSpec | None]:
    """Build the authoritative frozen FE constitutive handoff once per panel."""
    specs: list[PanelConstitutiveSpec | None] = []
    for pi, panel_map in enumerate(panel_maps):
        laminate = panel_laminates[pi]
        if len(panel_map.s_nodes) == 0 or laminate is None:
            specs.append(None)
            continue
        specs.append(
            panel_constitutive_spec_from_lam(
                laminate,
                material_kind=kinds[pi],
                sigma_y_pa=sigma_y_pa[pi],
                sigma_uts_pa=sigma_uts_pa[pi],
                sigma_ucs_pa=sigma_ucs_pa[pi],
                tau_us_pa=tau_us_pa[pi],
            )
        )
    return specs


def fill_global_dof_indices(
    cache: ShellFormulationCache,
    panel_maps: list[PanelNodeAssemblyMap],
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
    panel_maps: list[PanelNodeAssemblyMap],
    zones: list[Any],
    Nx_panels: list[NDArray[np.float64]],
    Nxy_panels: list[NDArray[np.float64]],
    *,
    formulation_id: str | None = None,
    shell_theory: ShellTheory = ShellTheory.FSDT,
    L_x: float = 1.0,
    zones_outboard: list[Any] | None = None,
    phase_collector: PhaseCollector | None = None,
) -> ShellFormulationCache:
    factory = resolve_element_factory(shell_theory, formulation_id=formulation_id)
    panel_abd_stacks = _panel_abd_stacks(panel_maps, zones)
    panel_laminates = _panel_laminates(panel_maps, zones)
    (
        panel_constitutive_kind,
        panel_sigma_y_pa,
        panel_sigma_uts_pa,
        panel_sigma_ucs_pa,
        panel_tau_us_pa,
    ) = _panel_constitutive_metadata(panel_maps, zones)
    panel_constitutive_specs = _typed_panel_constitutive_specs(
        panel_maps,
        panel_laminates,
        panel_constitutive_kind,
        panel_sigma_y_pa,
        panel_sigma_uts_pa,
        panel_sigma_ucs_pa,
        panel_tau_us_pa,
    )
    # Per-build stiffness-template cache (WP3). Passed to the factory only when it
    # opts in via a ``template_cache=`` keyword, so stub/other formulations that do
    # not accept it stay unaffected.
    from .template_cache import ShellFormulationTemplateCache

    template_cache = ShellFormulationTemplateCache()
    iter_kwargs: dict[str, Any] = dict(
        L_x=L_x,
        zones_outboard=zones_outboard,
        panel_abd_stacks=panel_abd_stacks,
        panel_constitutive_specs=panel_constitutive_specs,
    )
    if _factory_accepts_template_cache(factory):
        iter_kwargs["template_cache"] = template_cache
    # Materialised (not a single-pass generator) so stiffness and force build
    # each get their own aggregated phase timing, one start/end pair per
    # build_formulation_cache call rather than per element.
    elements = list(
        factory.iter_elements(
            panel_maps,
            zones,
            Nx_panels,
            Nxy_panels,
            **iter_kwargs,
        )
    )
    from .registry import resolve_formulation

    registration = resolve_formulation(_formulation_id_from_factory(factory))
    if registration.validate_element is not None:
        for elem in elements:
            registration.validate_element(elem)
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
        panel_sigma_y_pa=panel_sigma_y_pa,
        panel_sigma_uts_pa=panel_sigma_uts_pa,
        panel_sigma_ucs_pa=panel_sigma_ucs_pa,
        panel_tau_us_pa=panel_tau_us_pa,
        panel_constitutive_specs=panel_constitutive_specs,
        formulation_id=_formulation_id_from_factory(factory),
    )
    fill_global_dof_indices(cache, panel_maps)
    from .formulation_validation import validate_formulation_cache_shapes

    validate_formulation_cache_shapes(cache)
    if registration.validate_cache is not None:
        registration.validate_cache(cache)
    return cache


__all__ = [
    "Mitc4ElementFactory",
    "ShellElementFactory",
    "build_formulation_cache",
    "fill_global_dof_indices",
    "resolve_element_factory",
]
