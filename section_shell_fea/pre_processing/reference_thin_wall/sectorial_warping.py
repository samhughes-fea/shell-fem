"""
Sectorial coordinates and shear-centre from thin-wall sectorial static moments.

Shear centre (relative to modulus-weighted centroid) uses ω with pole at the
centroid (y_c, z_c):

  Iωy = ∫ ω (y - y_c) t ds ,   Iωz = ∫ ω (z - z_c) t ds

  e_y = (Iωz * Iyy - Iωy * Iyz) / D
  e_z = (Iωy * Izz - Iωz * Iyz) / D ,   D = Iyy Izz - Iyz²

  y_sc = y_c + e_y ,  z_sc = z_c + e_z

Straight segment increment (pole P):

  Δω = (y0 - y_p)(z1 - z0) - (z0 - z_p)(y1 - y0)

Closed outer contour: unique vertices (no repeated first point); all n edges
including the closing edge (v_{n-1} → v_0) enter the sectorial integrals.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from blade_structure_utilities.geometry_utilities.section_thin_wall_model import omega_increment


def open_outline_from_airfoil(airfoil: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Open midline for sectorial shear-centre (open thin-wall theory).

    Upper LE→TE, then the reversed lower (TE→LE) **without** repeating the TE
    vertex: ``upper ∪ lower_rev[1:]``.
    """
    n = len(airfoil) // 2
    upper = airfoil[:n]
    lower = airfoil[n:][::-1]
    return np.vstack([upper, lower[1:]])


def omega_vertices_open_chain(verts: NDArray[np.float64], yp: float, zp: float) -> NDArray[np.float64]:
    """ω at vertices along path v0→v1→…→v_{n-1} with ω[0]=0 (pole P)."""
    n = len(verts)
    omega = np.zeros(n, dtype=float)
    for i in range(n - 1):
        omega[i + 1] = omega[i] + omega_increment(
            verts[i, 0], verts[i, 1], verts[i + 1, 0], verts[i + 1, 1], yp, zp
        )
    return omega


def sectorial_static_moments_open(
    verts: NDArray[np.float64],
    omega_v: NDArray[np.float64],
    y_c: float,
    z_c: float,
    t: float,
) -> tuple[float, float]:
    """Open polyline only: edges 0→1 … →(n-2)→(n-1)."""
    n = len(verts)
    Iy = 0.0
    Iz = 0.0
    for i in range(n - 1):
        y0, z0 = verts[i]
        y1, z1 = verts[i + 1]
        om_m = 0.5 * (omega_v[i] + omega_v[i + 1])
        ym = 0.5 * (y0 + y1) - y_c
        zm = 0.5 * (z0 + z1) - z_c
        ds = float(np.hypot(y1 - y0, z1 - z0))
        Iy += om_m * ym * t * ds
        Iz += om_m * zm * t * ds
    return Iy, Iz


def sectorial_static_moments_closed(
    verts: NDArray[np.float64],
    omega_v: NDArray[np.float64],
    y_c: float,
    z_c: float,
    yp: float,
    zp: float,
    t: float,
) -> tuple[float, float]:
    """
    Iωy / Iωz on all boundary edges, including the closing edge v_{n-1} → v_0.
    """
    n = len(verts)
    Iy = 0.0
    Iz = 0.0
    for i in range(n - 1):
        y0, z0 = verts[i]
        y1, z1 = verts[i + 1]
        om_m = 0.5 * (omega_v[i] + omega_v[i + 1])
        ym = 0.5 * (y0 + y1) - y_c
        zm = 0.5 * (z0 + z1) - z_c
        ds = float(np.hypot(y1 - y0, z1 - z0))
        Iy += om_m * ym * t * ds
        Iz += om_m * zm * t * ds
    y0, z0 = verts[n - 1]
    y1, z1 = verts[0]
    dom = omega_increment(y0, z0, y1, z1, yp, zp)
    om_m = omega_v[n - 1] + 0.5 * dom
    ym = 0.5 * (y0 + y1) - y_c
    zm = 0.5 * (z0 + z1) - z_c
    ds = float(np.hypot(y1 - y0, z1 - z0))
    Iy += om_m * ym * t * ds
    Iz += om_m * zm * t * ds
    return Iy, Iz


def shear_center_from_sectorial(
    verts: NDArray[np.float64],
    props_yz: tuple[float, float, float, float, float],
    t_skin: float,
    *,
    closed: bool = False,
) -> tuple[float, float, NDArray[np.float64]]:
    """
    Shear centre using supplied (y_c, z_c, Iyy, Izz, Iyz) matching ``verts``.
    """
    y_c, z_c, Iyy, Izz, Iyz = props_yz
    omega_v = omega_vertices_open_chain(verts, y_c, z_c)
    if closed:
        Iwy, Iwz = sectorial_static_moments_closed(
            verts, omega_v, y_c, z_c, y_c, z_c, t_skin
        )
    else:
        Iwy, Iwz = sectorial_static_moments_open(verts, omega_v, y_c, z_c, t_skin)
    D = Iyy * Izz - Iyz**2
    if abs(D) < 1e-40:
        D = 1e-40
    e_y = (Iwz * Iyy - Iwy * Iyz) / D
    e_z = (Iwy * Izz - Iwz * Iyz) / D
    return float(y_c + e_y), float(z_c + e_z), omega_v


def modulus_weighted_open_line(
    verts: NDArray[np.float64],
    t: float,
    e_n: float,
) -> tuple[float, float, float, float, float, float]:
    """Line properties along an **open** polyline (no closing edge)."""
    n = len(verts)
    EA = 0.0
    EAy = 0.0
    EAz = 0.0
    for i in range(n - 1):
        y0, z0 = verts[i]
        y1, z1 = verts[i + 1]
        ds = float(np.hypot(y1 - y0, z1 - z0))
        w = e_n * t * ds
        ym = 0.5 * (y0 + y1)
        zm = 0.5 * (z0 + z1)
        EA += w
        EAy += w * ym
        EAz += w * zm
    if EA < 1e-30:
        EA = 1e-30
    y_c = EAy / EA
    z_c = EAz / EA
    Iyy = 0.0
    Izz = 0.0
    Iyz = 0.0
    for i in range(n - 1):
        y0, z0 = verts[i]
        y1, z1 = verts[i + 1]
        ds = float(np.hypot(y1 - y0, z1 - z0))
        w = e_n * t * ds
        ym = 0.5 * (y0 + y1)
        zm = 0.5 * (z0 + z1)
        yn = ym - y_c
        zn = zm - z_c
        Iyy += w * zn**2
        Izz += w * yn**2
        Iyz += w * yn * zn
    return float(y_c), float(z_c), float(EA), float(Iyy), float(Izz), float(Iyz)


def shear_center_outer_skin_loop(
    verts_open: NDArray[np.float64],
    t_skin: float,
    e_n: float,
) -> tuple[float, float, NDArray[np.float64], tuple[float, float, float, float, float]]:
    """
    Shear centre from **open** sectorial theory on ``verts_open`` (e.g. from
    ``open_outline_from_airfoil``), with matching line inertia on the same path.
    """
    y_c, z_c, _ea, Iyy, Izz, Iyz = modulus_weighted_open_line(verts_open, t_skin, e_n)
    y_sc, z_sc, omega_v = shear_center_from_sectorial(
        verts_open, (y_c, z_c, Iyy, Izz, Iyz), t_skin, closed=False
    )
    return y_sc, z_sc, omega_v, (y_c, z_c, Iyy, Izz, Iyz)


def polygon_area_signed(poly: NDArray[np.float64]) -> float:
    """Signed area (CCW positive) for a closed polyline (first point may equal last)."""
    y = np.asarray(poly, dtype=float)[:, 0]
    z = np.asarray(poly, dtype=float)[:, 1]
    if np.allclose(y[0], y[-1]) and np.allclose(z[0], z[-1]):
        y, z = y[:-1], z[:-1]
    return 0.5 * float(np.dot(y, np.roll(z, -1)) - np.dot(z, np.roll(y, -1)))


__all__ = [
    "omega_increment",
    "open_outline_from_airfoil",
    "omega_vertices_open_chain",
    "polygon_area_signed",
    "sectorial_static_moments_open",
    "sectorial_static_moments_closed",
    "shear_center_from_sectorial",
    "modulus_weighted_open_line",
    "shear_center_outer_skin_loop",
]
