"""Section-integral comparison utilities for shell validation.

Theory-agnostic helpers shared by the FSDT/CCX benchmark and CCX acceptance tests:

- :func:`section_element_geometry` -- per-element ``(ds, y, z, t)`` aligned with the
  recovered ``all_panel_results`` ordering (deterministic via
  ``build_panel_mesh_maps``).
- :func:`integrated_section_resultants` -- equilibrium integrals of the recovered
  membrane stress over the contour about the elastic centroid.
- :func:`compare_resultants` -- per-element relative-error stats between two solvers'
  ``ShellPanelResultants`` (e.g. FSDT vs S4R).
- :func:`equilibrium_interlaminar_tau_max` -- independent interlaminar reference from
  the spanwise sigma_xx gradient across two stations.

No external dependencies (no ``ccx``). Formerly ``hsdt_benchmark``; renamed when the
HSDT package was deleted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..engine.homogenisation import compute_elastic_centroid_from_panels
from blade_precompute.section_shell_model.pre_processing.mesh.shell_mesh_map import build_panel_mesh_maps, effective_n_elements_spec


@dataclass(frozen=True)
class ElementGeom:
    """Per-element contour geometry aligned with recovered results."""

    ds: float       # element arc length along the contour [m]
    y: float        # element-centre y (B-frame) [m]
    z: float        # element-centre z (B-frame) [m]
    thickness: float  # wall thickness [m]


def section_element_geometry(
    panels: list[Any], n_elements_per_panel: Any
) -> list[list[ElementGeom]]:
    """Per-panel, per-element ``ElementGeom`` matching the recovery element ordering.

    Recomputes the deterministic strip mesh (``build_panel_mesh_maps``) so element
    ``[pi][ei]`` aligns with ``bundle.all_panel_results[pi][ei]``. Element
    centre ``(y, z)`` is the midline interpolated at the element-centre arc length.
    """
    spec = effective_n_elements_spec(panels, n_elements_per_panel, None)
    panel_maps, _ = build_panel_mesh_maps(panels, spec)
    out: list[list[ElementGeom]] = []
    for p, pm in zip(panels, panel_maps):
        rows: list[ElementGeom] = []
        if len(pm.s_nodes) == 0:
            out.append(rows)
            continue
        s_p = np.asarray(p.s, dtype=float)
        nodes = np.asarray(p.nodes, dtype=float)
        y_n = np.interp(pm.s_nodes, s_p, nodes[:, 0])
        z_n = np.interp(pm.s_nodes, s_p, nodes[:, 1])
        t = float(getattr(getattr(p, "lam", None), "t", 0.0) or 0.0)
        for e in pm.elements:
            i0, i1 = int(e[0]), int(e[1])
            ds = float(abs(pm.s_nodes[i1] - pm.s_nodes[i0]))
            rows.append(ElementGeom(ds=ds, y=0.5 * (y_n[i0] + y_n[i1]),
                                    z=0.5 * (z_n[i0] + z_n[i1]), thickness=t))
        out.append(rows)
    return out


def integrated_section_resultants(
    bundle: Any, geom: list[list[ElementGeom]], *, y_e: float, z_e: float
) -> dict[str, float]:
    """Equilibrium integrals of the recovered membrane stress over the contour.

    Returns ``Fx`` [N], ``My`` [N·m], ``Mz`` [N·m] about the elastic centroid
    ``(y_e, z_e)``. Uses ``Nx = sigma_xx_pa * thickness`` (membrane force/length).
    For a correct solve these equal the applied ``(N, My, Mz)``.
    """
    fx = my = mz = 0.0
    for pi, panel_res in enumerate(bundle.all_panel_results or []):
        for ei, r in enumerate(panel_res):
            if ei >= len(geom[pi]):
                continue
            g = geom[pi][ei]
            nx = float(r.sigma_xx_pa) * float(r.thickness_m)  # [N/m]
            f = nx * g.ds
            fx += f
            my += f * (g.z - z_e)
            mz += f * (g.y - y_e)
    return {"Fx": fx, "My": my, "Mz": mz}


def compare_resultants(
    res_a: list[list[Any]],
    res_b: list[list[Any]],
    *,
    fields: tuple[str, ...] = ("Nx", "Ny", "Nxy", "Mx", "My", "Mxy", "Qx", "Qy"),
) -> dict[str, float]:
    """Per-element relative-error stats between two solvers' results (same mesh).

    Returns, per field, the L2 relative error ``||a−b|| / max(||b||, eps)`` over all
    elements where both are finite & non-None. Keys are ``rel_<field>``; also
    ``n_elements``.
    """
    stats: dict[str, float] = {}
    n_elem = 0
    for field in fields:
        a_vals: list[float] = []
        b_vals: list[float] = []
        for pa, pb in zip(res_a, res_b):
            for ra, rb in zip(pa, pb):
                va = getattr(ra, field, None)
                vb = getattr(rb, field, None)
                if va is None or vb is None:
                    continue
                if not (np.isfinite(va) and np.isfinite(vb)):
                    continue
                a_vals.append(float(va))
                b_vals.append(float(vb))
        if not a_vals:
            stats[f"rel_{field}"] = float("nan")
            continue
        a = np.asarray(a_vals)
        b = np.asarray(b_vals)
        stats[f"rel_{field}"] = float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), 1e-30))
        n_elem = max(n_elem, len(a_vals))
    stats["n_elements"] = float(n_elem)
    return stats


def section_sigma_field(bundle: Any, geom: list[list[ElementGeom]]) -> dict[str, NDArray[np.float64]]:
    """Flatten recovered per-element ``(y, z, sigma_xx)`` for field-shape checks."""
    ys: list[float] = []
    zs: list[float] = []
    sx: list[float] = []
    for pi, panel_res in enumerate(bundle.all_panel_results or []):
        for ei, r in enumerate(panel_res):
            if ei >= len(geom[pi]):
                continue
            ys.append(geom[pi][ei].y)
            zs.append(geom[pi][ei].z)
            sx.append(float(r.sigma_xx_pa))
    return {"y": np.asarray(ys), "z": np.asarray(zs), "sigma_xx": np.asarray(sx)}


def elastic_centroid(panels: list[Any]) -> tuple[float, float]:
    """A11-weighted elastic centroid (y_e, z_e) — reuse of the homogenisation helper."""
    return compute_elastic_centroid_from_panels(panels)


def multistation_equilibrium_interlaminar(
    pb: Any,
    solver: Any,
    *,
    My0: float,
    Vz: float,
    dz: float,
    n_elements_per_panel: Any = 8,
) -> dict[str, float]:
    """Physical interlaminar shear from the spanwise σ_xx gradient (two stations).

    Transverse shear ``Vz = dMy/dx`` only exists across the span, so it is imposed
    as a *gradient*: stations at ``My0 ± Vz·dz``. The equilibrium recovery
    ``τ_max = |∂σ_xx/∂x|·t/2`` (``examples/section_stress_model/lib/interlaminar_shear.py``)
    then gives the physical interlaminar shear — the reference the single-section
    FSDT transverse-shear recovery is compared against.

    ``solver`` is e.g. ``run_section_periodic_bc_mitc4``.
    Returns ``tau_eq_max`` / ``tau_eq_mean`` [Pa] and ``dsigma_dx_max`` [Pa/m].
    """
    from lib.interlaminar_shear import interlaminar_shear_max  # type: ignore[import-untyped]

    geom = section_element_geometry(pb.panels, n_elements_per_panel)
    t_wall = float(getattr(getattr(pb.panels[0], "lam", None), "t", 1e-3) or 1e-3)
    sig_prev = section_sigma_field(
        solver(pb, My=My0 - Vz * dz, n_elements_per_panel=n_elements_per_panel), geom
    )["sigma_xx"]
    sig_next = section_sigma_field(
        solver(pb, My=My0 + Vz * dz, n_elements_per_panel=n_elements_per_panel), geom
    )["sigma_xx"]
    tau_eq = interlaminar_shear_max(sig_prev, sig_next, dz, t_wall)
    dsig_dx = np.abs(sig_next - sig_prev) / (2.0 * dz)
    return {
        "tau_eq_max": float(np.max(tau_eq)),
        "tau_eq_mean": float(np.mean(tau_eq)),
        "dsigma_dx_max": float(np.max(dsig_dx)),
    }


def single_section_interlaminar_max(bundle: Any) -> float:
    """Max |tau| from a single-section shell recovery (qx/qy mid layers) [Pa]."""
    vals = [
        max(abs(float(r.qx_mid)), abs(float(r.qy_mid)))
        for panel in (bundle.all_panel_results or [])
        for r in panel
    ]
    return float(max(vals)) if vals else 0.0


__all__ = [
    "ElementGeom",
    "section_element_geometry",
    "integrated_section_resultants",
    "compare_resultants",
    "section_sigma_field",
    "elastic_centroid",
    "multistation_equilibrium_interlaminar",
    "single_section_interlaminar_max",
]
