"""Load-independent periodic assembly / K6 caches for section solves.

Extracted from ``api/periodic.py`` so the theory router stays thin and workers /
tests can probe or clear caches without importing the full periodic solvers.

Assembly goes through :func:`assemble_section_from_panels` (formulation cache →
scatter-add), not the retired full-solve recovery spine.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from blade_precompute.section_shell_model.pre_processing.containers.formulation_objects import (
    ShellFormulationCache,
)
from blade_precompute.section_shell_model.pre_processing.containers.types import ShellTheory
from blade_precompute.section_shell_model.pre_processing.mesh.mesh_topology import NElements
from blade_precompute.section_shell_model.pre_processing.mesh.section_strip_mesh import MIN_ELEMENTS_PER_PANEL
from ..pre_processing.element_library.mitc4.utilities.mitc4_solve_orchestrator import collect_panel_abd_stack
from .connected_assembly_core import canonicalize_bc_mode
from .section_assembly import SectionAssemblyResult, assemble_section_from_panels

_PERIODIC_ASSEMBLY_CACHE: dict[tuple, SectionAssemblyResult] = {}
_TAPERED_PERIODIC_ASSEMBLY_CACHE: dict[tuple, SectionAssemblyResult] = {}
_PERIODIC_K6_CACHE: dict[tuple, tuple] = {}
_PERIODIC_REDUCED_BVP_CACHE: dict[tuple, Any] = {}


def clear_periodic_caches() -> None:
    """Drop all periodic assembly / K6 memoisation (tests, worker reset)."""
    _PERIODIC_ASSEMBLY_CACHE.clear()
    _TAPERED_PERIODIC_ASSEMBLY_CACHE.clear()
    _PERIODIC_K6_CACHE.clear()
    _PERIODIC_REDUCED_BVP_CACHE.clear()


def _laminate_cache_fingerprint(lam: Any) -> tuple:
    """Stable laminate key for assembly / homogenisation caches."""
    fp = getattr(lam, "fingerprint", None)
    if callable(fp):
        return tuple(fp())
    build_plies = getattr(lam, "build_plies", None)
    if callable(build_plies):
        try:
            plies = build_plies()
        except Exception:
            plies = ()
        parts: list[Any] = []
        for ply in plies:
            parts.extend(
                (
                    round(float(getattr(ply, "theta_deg", getattr(ply, "theta", 0.0))), 9),
                    round(float(getattr(ply, "t", getattr(ply, "t_ply", 0.0))), 12),
                    round(float(getattr(ply, "E1", getattr(ply, "E", 0.0))), 6),
                    round(float(getattr(ply, "E2", 0.0)), 6),
                    round(float(getattr(ply, "G12", 0.0)), 6),
                    round(float(getattr(ply, "nu12", getattr(ply, "nu", 0.0))), 9),
                )
            )
        if parts:
            return ("build_plies", tuple(parts))
    return (
        "scalar",
        round(float(getattr(lam, "t", 0.0)), 12),
        round(float(getattr(lam, "E", 0.0)), 6),
        round(float(getattr(lam, "nu", 0.0)), 9),
        str(getattr(lam, "name", "")),
    )


def _periodic_assembly_cache_key(
    panels: Sequence[Any],
    *,
    n_elements_per_panel: NElements,
    target_element_length_m: float | None,
    min_elements_per_panel: int,
    bc_mode: str,
    interface_constraint_mode: str,
) -> tuple:
    parts: list[tuple] = []
    for p in panels:
        s = tuple(np.asarray(p.s, dtype=float).round(12).tolist())
        nodes = tuple(
            tuple(row) for row in np.asarray(getattr(p, "nodes"), dtype=float).round(12).tolist()
        )
        lam = p.lam
        parts.append(
            (
                s,
                nodes,
                _laminate_cache_fingerprint(lam),
                str(getattr(p, "label", "")),
            )
        )
    return (
        tuple(parts),
        n_elements_per_panel,
        target_element_length_m,
        min_elements_per_panel,
        canonicalize_bc_mode(bc_mode),
        interface_constraint_mode,
    )


def _cached_periodic_assembly(
    panels: list[Any],
    nx_panels: list[np.ndarray],
    nxy_panels: list[np.ndarray],
    *,
    n_elements_per_panel: NElements,
    target_element_length_m: float | None,
    min_elements_per_panel: int,
    bc_mode: str,
    interface_constraint_mode: str,
) -> SectionAssemblyResult:
    """Assemble via formulation cache; ``bc_mode`` is key-only (unused for K/f)."""
    key = _periodic_assembly_cache_key(
        panels,
        n_elements_per_panel=n_elements_per_panel,
        target_element_length_m=target_element_length_m,
        min_elements_per_panel=min_elements_per_panel,
        bc_mode=bc_mode,
        interface_constraint_mode=interface_constraint_mode,
    )
    cached = _PERIODIC_ASSEMBLY_CACHE.get(key)
    if cached is not None:
        return cached
    result = assemble_section_from_panels(
        panels,
        nx_panels,
        nxy_panels,
        n_elements_per_panel=n_elements_per_panel,
        target_element_length_m=target_element_length_m,
        min_elements_per_panel=min_elements_per_panel,
        interface_constraint_mode=interface_constraint_mode,
        shell_theory=ShellTheory.FSDT,
    )
    _PERIODIC_ASSEMBLY_CACHE[key] = result
    return result


def _unpack_assembly_result(
    result: SectionAssemblyResult | tuple,
    panels: Sequence[Any],
) -> tuple[Any, Any, Any, list, np.ndarray, np.ndarray, np.ndarray, list[str], ShellFormulationCache | None]:
    """Normalize assembly cache payload to 8-tuple fields + optional formulation cache."""
    if isinstance(result, SectionAssemblyResult):
        return (*result.as_legacy_8tuple(), result.cache)
    if len(result) >= 9 and isinstance(result[8], ShellFormulationCache):
        return (
            result[0],
            result[1],
            result[2],
            result[3],
            result[4],
            result[5],
            result[6],
            result[7],
            result[8],
        )
    if len(result) >= 8:
        return (*result[:8], None)
    K_global, f_global, node_meta, panel_maps = result[:4]
    try:
        abd_p, t_p, g_p, lab_p = collect_panel_abd_stack(panels)
    except Exception:
        # Test stubs / incomplete panel laminates: empty ABD backfill.
        n = len(panels)
        abd_p = np.zeros((n, 6, 6), dtype=float)
        t_p = np.zeros(n, dtype=float)
        g_p = np.zeros(n, dtype=float)
        lab_p = [str(getattr(p, "label", f"panel_{i}")) for i, p in enumerate(panels)]
    return K_global, f_global, node_meta, panel_maps, abd_p, t_p, g_p, lab_p, None


def _cached_periodic_k6(
    panels: list[Any],
    *,
    n_elements_per_panel: NElements,
    target_element_length_m: float | None,
    min_elements_per_panel: int,
    bc_mode: str,
    interface_constraint_mode: str,
) -> tuple:
    """Assembly + periodic K6 homogenisation, memoised on geometry/laminate/mesh.

    Returns
    -------
    13-tuple
        ``(K, f, node_meta, panel_maps, K6, u_modes, y_e, z_e, abd_p, t_p, g_p,
        lab_p, cache)`` where ``cache`` is the retained :class:`ShellFormulationCache`.
    """
    from .homogenisation import compute_elastic_centroid_from_panels, compute_k6_periodic_bc

    key = _periodic_assembly_cache_key(
        panels,
        n_elements_per_panel=n_elements_per_panel,
        target_element_length_m=target_element_length_m,
        min_elements_per_panel=min_elements_per_panel,
        bc_mode=bc_mode,
        interface_constraint_mode=interface_constraint_mode,
    )
    cached = _PERIODIC_K6_CACHE.get(key)
    if cached is not None:
        return cached

    zeros = [np.zeros(len(np.asarray(p.s, dtype=float))) for p in panels]
    assembly_result = _cached_periodic_assembly(
        panels,
        zeros,
        zeros,
        n_elements_per_panel=n_elements_per_panel,
        target_element_length_m=target_element_length_m,
        min_elements_per_panel=min_elements_per_panel,
        bc_mode=bc_mode,
        interface_constraint_mode=interface_constraint_mode,
    )
    (
        K_global,
        f_global,
        node_meta,
        panel_maps,
        abd_p,
        t_p,
        g_p,
        lab_p,
        cache,
    ) = _unpack_assembly_result(assembly_result, panels)
    y_e, z_e = compute_elastic_centroid_from_panels(panels)
    K6, u_modes, bvp = compute_k6_periodic_bc(
        K_global, node_meta, panel_maps, panels, y_e, z_e, return_bvp=True,
    )
    result = (
        K_global,
        f_global,
        node_meta,
        panel_maps,
        K6,
        u_modes,
        y_e,
        z_e,
        abd_p,
        t_p,
        g_p,
        lab_p,
        cache,
    )
    _PERIODIC_K6_CACHE[key] = result
    _PERIODIC_REDUCED_BVP_CACHE[key] = bvp
    return result


def get_cached_periodic_reduced_bvp(
    panels: list[Any],
    *,
    n_elements_per_panel: NElements,
    target_element_length_m: float | None,
    min_elements_per_panel: int,
    bc_mode: str,
    interface_constraint_mode: str,
) -> Any:
    """Return the memoised :class:`PeriodicReducedBVP` for the geometry key.

    Ensures :func:`_cached_periodic_k6` has run so the BVP factor is populated.
    """
    key = _periodic_assembly_cache_key(
        panels,
        n_elements_per_panel=n_elements_per_panel,
        target_element_length_m=target_element_length_m,
        min_elements_per_panel=min_elements_per_panel,
        bc_mode=bc_mode,
        interface_constraint_mode=interface_constraint_mode,
    )
    bvp = _PERIODIC_REDUCED_BVP_CACHE.get(key)
    if bvp is not None:
        return bvp
    _cached_periodic_k6(
        panels,
        n_elements_per_panel=n_elements_per_panel,
        target_element_length_m=target_element_length_m,
        min_elements_per_panel=min_elements_per_panel,
        bc_mode=bc_mode,
        interface_constraint_mode=interface_constraint_mode,
    )
    bvp = _PERIODIC_REDUCED_BVP_CACHE.get(key)
    if bvp is None:
        raise RuntimeError("periodic reduced BVP missing after K6 cache build")
    return bvp


def get_periodic_geometry_state(
    pb: Any,
    *,
    n_elements_per_panel: NElements = 10,
    target_element_length_m: float | None = None,
    min_elements_per_panel: int = MIN_ELEMENTS_PER_PANEL,
    bc_mode: str = "contour_support",
    interface_constraint_mode: str = "shared",
) -> tuple:
    """Public accessor for the load-independent periodic assembly/K6/ABD cache.

    Thin wrapper over :func:`_cached_periodic_k6` so callers outside this module
    (tests, worker warm-up hooks) don't need the private symbol.
    """
    return _cached_periodic_k6(
        pb.panels,
        n_elements_per_panel=n_elements_per_panel,
        target_element_length_m=target_element_length_m,
        min_elements_per_panel=min_elements_per_panel,
        bc_mode=bc_mode,
        interface_constraint_mode=interface_constraint_mode,
    )


def _tapered_periodic_assembly_cache_key(
    panels_in: Sequence[Any],
    panels_out: Sequence[Any],
    *,
    L_x: float,
    n_elements_per_panel: NElements,
    target_element_length_m: float | None,
    min_elements_per_panel: int,
    bc_mode: str,
    interface_constraint_mode: str,
) -> tuple:
    return (
        _periodic_assembly_cache_key(
            panels_in,
            n_elements_per_panel=n_elements_per_panel,
            target_element_length_m=target_element_length_m,
            min_elements_per_panel=min_elements_per_panel,
            bc_mode=bc_mode,
            interface_constraint_mode=interface_constraint_mode,
        ),
        _periodic_assembly_cache_key(
            panels_out,
            n_elements_per_panel=n_elements_per_panel,
            target_element_length_m=target_element_length_m,
            min_elements_per_panel=min_elements_per_panel,
            bc_mode=bc_mode,
            interface_constraint_mode=interface_constraint_mode,
        ),
        float(L_x),
    )


def _cached_tapered_periodic_assembly(
    panels_in: list[Any],
    panels_out: list[Any],
    nx_panels: list[np.ndarray],
    nxy_panels: list[np.ndarray],
    *,
    L_x: float,
    n_elements_per_panel: NElements,
    target_element_length_m: float | None,
    min_elements_per_panel: int,
    bc_mode: str,
    interface_constraint_mode: str,
) -> SectionAssemblyResult:
    key = _tapered_periodic_assembly_cache_key(
        panels_in,
        panels_out,
        L_x=float(L_x),
        n_elements_per_panel=n_elements_per_panel,
        target_element_length_m=target_element_length_m,
        min_elements_per_panel=min_elements_per_panel,
        bc_mode=bc_mode,
        interface_constraint_mode=interface_constraint_mode,
    )
    cached = _TAPERED_PERIODIC_ASSEMBLY_CACHE.get(key)
    if cached is not None:
        return cached
    result = assemble_section_from_panels(
        panels_in,
        nx_panels,
        nxy_panels,
        panels_outboard=panels_out,
        L_x=float(L_x),
        n_elements_per_panel=n_elements_per_panel,
        target_element_length_m=target_element_length_m,
        min_elements_per_panel=min_elements_per_panel,
        interface_constraint_mode=interface_constraint_mode,
        shell_theory=ShellTheory.FSDT,
    )
    _TAPERED_PERIODIC_ASSEMBLY_CACHE[key] = result
    return result


__all__ = [
    "clear_periodic_caches",
    "get_periodic_geometry_state",
    "_PERIODIC_ASSEMBLY_CACHE",
    "_PERIODIC_K6_CACHE",
    "_TAPERED_PERIODIC_ASSEMBLY_CACHE",
    "_PERIODIC_REDUCED_BVP_CACHE",
    "_cached_periodic_assembly",
    "_cached_periodic_k6",
    "_cached_tapered_periodic_assembly",
    "_unpack_assembly_result",
    "get_cached_periodic_reduced_bvp",
]
