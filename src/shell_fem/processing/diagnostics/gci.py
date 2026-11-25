"""ASME grid convergence index: the discretisation error of one result from three meshes.

The procedure is that of Celik et al. (2008), "Procedure for estimation and reporting of
uncertainty due to discretization in CFD applications", J. Fluids Eng. 130(7), 078001,
the one the ASME V&V 20 standard adopts. With h1 < h2 < h3 the representative element
sizes of the fine, medium and coarse meshes, r21 = h2/h1 and r32 = h3/h2, and phi_k the
result on mesh k:

    e21 = phi2 - phi1,  e32 = phi3 - phi2,  s = sign(e32 / e21)
    p   = | ln|e32 / e21| + q(p) | / ln r21,   q(p) = ln((r21^p - s) / (r32^p - s))
    phi_ext   = (r21^p phi1 - phi2) / (r21^p - 1)
    e_a       = |(phi1 - phi2) / phi1|
    e_ext     = |(phi_ext - phi1) / phi_ext|
    GCI_fine  = Fs e_a / (r21^p - 1),   Fs = 1.25

p, the observed order, is solved by fixed-point iteration (q = 0 when r21 = r32). s < 0
means the result oscillates between meshes; the estimate is then less reliable and the
result says so. When e21 and e32 are both within ``max(atol, rtol |phi|)`` the result has
converged (a quantity fixed by equilibrium, or zero by symmetry) and GCI is zero; when only
one of them is, the order is undefined and the result holds nan.

``asymptotic_ratio`` is GCI32 / (r21^p GCI21) on absolute differences. It is exactly 1
when the three values follow phi_0 + C h^p, so with an observed order it is 1 whenever a
positive p fits the three values, and far from 1 when none does: the difference between
the finer pair is the larger, so the result has not settled. ``in_asymptotic_range``
checks it. Three meshes cannot show more than that; with four or more,
``MeshConvergenceResults.reliable`` also asks that consecutive triples agree on p.

The meshes should refine systematically (the same mesh topology, every size scaled),
with r above about 1.3. ``mesh_size`` gives the representative size h = (A / N)^(1/2) of
a shell mesh of total area A and N elements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

SAFETY_FACTOR = 1.25


@dataclass(frozen=True)
class GCIResult:
    """The GCI of one result over three meshes (fine first). ``gci`` is relative (GCI_fine21)."""

    h: tuple
    values: tuple
    order: float
    extrapolated: float
    relative_error: float          # e_a21, between the fine and medium meshes
    extrapolated_error: float      # e_ext21, of the fine mesh against phi_ext
    gci: float                     # GCI_fine21, a relative band on the fine-mesh value
    gci_coarse: float              # GCI_fine32, the same on the medium mesh
    oscillatory: bool
    converged: bool                # e21 and e32 zero to round-off

    @property
    def fine(self) -> float:
        return self.values[0]

    @property
    def band(self) -> tuple:
        """(low, high): the fine-mesh value plus or minus its GCI."""
        d = abs(self.fine) * self.gci
        return self.fine - d, self.fine + d

    @property
    def asymptotic_ratio(self) -> float:
        """GCI32 / (r21^p GCI21) on absolute errors: exactly 1 for phi = phi_0 + C h^p (nan when undefined)."""
        h1, h2, h3 = self.h
        phi1, phi2, phi3 = self.values
        p = self.order
        if not math.isfinite(p):
            return float("nan")
        r21, r32 = h2 / h1, h3 / h2
        denominator = r21 ** p * abs(phi2 - phi1) / (r21 ** p - 1.0)
        return abs(phi3 - phi2) / (r32 ** p - 1.0) / denominator if denominator > 0.0 else float("nan")

    def in_asymptotic_range(self, tol: float = 0.1) -> bool:
        """True when converged, or when ``asymptotic_ratio`` is within ``tol`` of 1 (a positive order fits).

        Outside that range the observed order and the GCI are not trustworthy: refine further.
        """
        if self.converged:
            return True
        ratio = self.asymptotic_ratio
        return math.isfinite(self.order) and math.isfinite(ratio) and abs(ratio - 1.0) <= tol


def mesh_size(mesh) -> float:
    """Representative element size h = (A / N)^(1/2) of a shell mesh (N shell elements of total area A)."""
    area, count = 0.0, 0
    for element in mesh.elements:
        if not hasattr(element, "load_rule"):
            continue
        area += sum(element.jacobian(xi, eta)[1] * w for xi, eta, w in element.load_rule())
        count += 1
    if count == 0:
        raise ValueError("the mesh has no shell elements")
    return math.sqrt(area / count)


def _observed_order(r21: float, r32: float, e21: float, e32: float, s: float, max_iter: int, tol: float) -> float:
    ratio = abs(e32 / e21)
    p = abs(math.log(ratio)) / math.log(r21)
    for _ in range(max_iter):
        a, b = r21 ** p - s, r32 ** p - s
        q = math.log(a / b) if a > 0.0 and b > 0.0 else 0.0
        p_new = abs(math.log(ratio) + q) / math.log(r21)
        if abs(p_new - p) <= tol * max(1.0, p):
            return p_new
        p = p_new
    return p


def grid_convergence_index(h: Sequence[float], values: Sequence[float], safety_factor: float = SAFETY_FACTOR,
                           order: Optional[float] = None, rtol: float = 1e-8, atol: float = 0.0,
                           max_iter: int = 200, tol: float = 1e-10) -> GCIResult:
    """GCI of a result ``values`` on three meshes of sizes ``h`` (any order; sorted fine first).

    ``order`` fixes p (the formal order) instead of solving for the observed one; Celik et
    al. then advise ``safety_factor=3``. Changes between meshes within
    ``max(atol, rtol * max|phi|)`` count as none.
    """
    h = np.asarray(h, dtype=float)
    phi = np.asarray(values, dtype=float)
    if h.shape != (3,) or phi.shape != (3,):
        raise ValueError("the GCI needs exactly three meshes and three values")
    if np.any(h <= 0.0):
        raise ValueError("mesh sizes must be positive")
    idx = np.argsort(h)
    h, phi = h[idx], phi[idx]
    if np.any(np.diff(h) <= 0.0):
        raise ValueError("the three mesh sizes must differ")
    r21, r32 = h[1] / h[0], h[2] / h[1]
    e21, e32 = phi[1] - phi[0], phi[2] - phi[1]
    h_out, phi_out = tuple(float(v) for v in h), tuple(float(v) for v in phi)
    eps = max(float(atol), float(rtol) * float(np.abs(phi).max()))
    if abs(e21) <= eps and abs(e32) <= eps:
        return GCIResult(h_out, phi_out, float("inf"), float(phi[0]), 0.0, 0.0, 0.0, 0.0, False, True)
    if abs(e21) <= eps or abs(e32) <= eps:
        nan = float("nan")
        return GCIResult(h_out, phi_out, nan, nan, abs(e21 / phi[0]) if phi[0] else nan, nan, nan, nan, False, False)
    s = 1.0 if e32 / e21 > 0.0 else -1.0
    p = float(order) if order is not None else _observed_order(r21, r32, e21, e32, s, max_iter, tol)
    f21, f32 = r21 ** p - 1.0, r32 ** p - 1.0
    phi_ext = (r21 ** p * phi[0] - phi[1]) / f21
    e_a21 = abs(e21 / phi[0]) if phi[0] != 0.0 else float("inf")
    e_a32 = abs(e32 / phi[1]) if phi[1] != 0.0 else float("inf")
    e_ext = abs((phi_ext - phi[0]) / phi_ext) if phi_ext != 0.0 else float("inf")
    return GCIResult(h_out, phi_out, p, float(phi_ext), float(e_a21), float(e_ext),
                     float(safety_factor * e_a21 / f21), float(safety_factor * e_a32 / f32), s < 0.0, False)


__all__ = ["GCIResult", "SAFETY_FACTOR", "grid_convergence_index", "mesh_size"]
