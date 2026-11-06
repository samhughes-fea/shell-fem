"""Beam section properties extracted from a blade shell model.

``beam_section_properties`` loads a blade segment (clamped at one end, loaded through
a rigid link at the other), solves the shell model and reads the beam stiffness off
the end displacements: the Timoshenko 6 x 6 with all its couplings, the classical
4 x 4 inside it, the Timoshenko-Vlasov 7 x 7 with the warping DOF, and the shear,
tension and mass centres.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from laminate_constitutive_model.laminate_mechanics import Laminate

from shell_fem.pre_processing.mesh_library.blade import BladeMesh

@dataclass(frozen=True)
class BeamSectionProperties:
    """Beam properties of a blade segment from the shell model, in global axes.

    ``stiffness_6x6`` is the Timoshenko section stiffness: it relates the section forces
    [Fx, Fy, N, Mx, My, Mz] to the beam strains [gamma_x, gamma_y, eps, kappa_x,
    kappa_y, kappa_z] about the reference point (the pitch axis by default), in the
    BeamDyn / windIO order, with gamma_x = u_x' - theta_y and gamma_y = u_y' + theta_x.
    Its off-diagonal terms are the couplings: extension-bending, bend-twist (for example
    of an aeroelastically tailored blade), shear-twist (the shear centre away from the
    reference) and shear-bending.

    ``stiffness`` is the classical (Euler-Bernoulli) 4 x 4 for [N, Mx, My, Mz] against
    [eps, kappa_x, kappa_y, kappa_z]: the 6 x 6 with the shear forces free (the inverse
    of the axial-bending-torsion block of the 6 x 6 compliance), as an Euler-Bernoulli
    beam code wants it.

    ``stiffness_7x7`` (when measured, ``warping=True``) adds the warping DOF of
    Timoshenko-Vlasov beam theory: the seventh strain is kappa_z' and its force the
    bimoment, with ``EGamma`` on the diagonal; ``warping_shear_stiffness`` is the
    extra shear stiffness S of Benscoter's theory (closed sections), so that a
    ``RestrainedTorsionBeam(GJ, EGamma, L, S)`` reproduces the shell's restrained twist.

    ``tension_centre`` is where an axial force causes no bending and ``shear_centre``
    where a shear force causes no twist. Lengths are in the mesh units.
    """

    z_range: Tuple[float, float]
    reference: Tuple[float, float]
    stiffness: np.ndarray  # 4 x 4 classical
    tension_centre: Tuple[float, float]
    shear_centre: Tuple[float, float]
    mass_per_length: float
    mass_centre: Tuple[float, float]
    stiffness_6x6: Optional[np.ndarray] = None
    stiffness_7x7: Optional[np.ndarray] = None
    warping_shear_stiffness: Optional[float] = None

    @property
    def EA(self) -> float:
        return float(self.stiffness[0, 0])

    @property
    def EI_x(self) -> float:
        """Bending stiffness about the x axis (curvature in the y-z plane)."""
        return float(self.stiffness[1, 1])

    @property
    def EI_y(self) -> float:
        return float(self.stiffness[2, 2])

    @property
    def GJ(self) -> float:
        return float(self.stiffness[3, 3])

    @property
    def GA(self) -> Tuple[float, float]:
        """Transverse shear stiffnesses (GA_x, GA_y): the 6 x 6 diagonal for gamma_x and gamma_y."""
        return float(self.stiffness_6x6[0, 0]), float(self.stiffness_6x6[1, 1])

    @property
    def EGamma(self) -> Optional[float]:
        """Warping stiffness EΓ (with ``warping=True``)."""
        return None if self.stiffness_7x7 is None else float(self.stiffness_7x7[6, 6])

    @property
    def compliance_6x6(self) -> np.ndarray:
        return np.linalg.inv(self.stiffness_6x6)

    @property
    def bend_twist_coupling(self) -> Tuple[float, float]:
        """Normalised couplings S[1,3]/sqrt(S11 S33) and S[2,3]/sqrt(S22 S33) (zero when uncoupled)."""
        S = self.stiffness
        return (float(S[1, 3] / np.sqrt(S[1, 1] * S[3, 3])), float(S[2, 3] / np.sqrt(S[2, 2] * S[3, 3])))

    def at(self, point: Tuple[float, float]) -> np.ndarray:
        """The 6 x 6 moved to another reference point (same axes)."""
        return transform_beam_stiffness(self.stiffness_6x6, (point[0] - self.reference[0], point[1] - self.reference[1]))


def transform_beam_stiffness(K6: np.ndarray, offset: Tuple[float, float], angle: float = 0.0) -> np.ndarray:
    """Move a 6 x 6 Timoshenko stiffness (BeamDyn order) to a reference point ``offset``
    (x, y) from the current one, and turn its axes by ``angle`` (radians, about z).

    Forces at the new point: F' = F, M' = M - r x F, written f' = T f; the strains
    transform so that the work f . e is unchanged (e = T^T e'), so K' = T K T^T.
    """
    dx, dy = float(offset[0]), float(offset[1])
    T = np.eye(6)
    # M' = M - r x F with r = (dx, dy, 0): (r x F) = (dy Fz, -dx Fz, dx Fy - dy Fx)
    T[3, 2] -= dy
    T[4, 2] += dx
    T[5, 0] += dy
    T[5, 1] -= dx
    if angle:
        c, s = np.cos(angle), np.sin(angle)
        R2 = np.array([[c, s], [-s, c]])  # components in axes turned by +angle
        Rot = np.eye(6)
        Rot[0:2, 0:2] = R2
        Rot[3:5, 3:5] = R2
        T = Rot @ T
    return T @ K6 @ T.T


def beam_section_properties(
    blade: BladeMesh,
    z_range: Optional[Tuple[float, float]] = None,
    reference: Optional[Tuple[float, float]] = None,
    solver=None,
    warping: bool = False,
    section_weighting: str = "stiffness",
    end_ribs: float = 0.0,
) -> BeamSectionProperties:
    """Extract beam properties of a blade segment by loading the shell model as a cantilever.

    The root is clamped and unit loads are applied through a rigid tip section (RBE2
    on the translations, like an end rib). Two more distributing links measure the mean
    section motion (translations and rotations) at the ends of ``z_range`` (the middle
    fifth of the span by default), away from the end effects, and their differences give
    the strains of the beam. Axial force and moments are constant along a cantilever, so
    they give four columns of the 6 x 6 compliance directly; the tip shear forces add a
    bending moment that varies along the span, which is subtracted to give the shear
    columns (the shear strain is the slope of the axis minus the section rotation,
    averaged over ``z_range`` exactly). The blade should be prismatic over ``z_range``
    (a long, uniform segment, five chords or more, gives the cleanest values).
    ``reference`` is the (x, y) point the properties refer to.

    The section rotation of a Timoshenko beam is a definition: shear warps the section,
    so it has no single rotation. ``section_weighting="stiffness"`` (the default) takes
    the axial-stiffness-weighted least-squares rotation (each node weighted by Et times
    its share of the section contour), the rotation that does work with the bending
    moment, as 2D cross-section tools (VABS, ANBA, BECAS) define it. ``"uniform"``
    weights every node alike. The 4 x 4 is the same either way (pure bending keeps the
    section plane); the shear stiffnesses are not.

    The clamped root and the rigid tip restrain warping. Left free to distort, the
    section relieves that restraint by distorting, over a length that is long for a
    section with no ribs, and the distortion reaches the middle of the span and shifts
    GJ by several percent (examples 27 and 28). ``end_ribs`` (a fraction of the span)
    holds the sections within that distance of either end rigid in their plane, as ribs
    or bulkheads would, so that only the short warping boundary layer remains and the
    middle of the span carries the Saint-Venant state: GJ then matches
    ``ThinWalledSection.from_blade_mesh(blade).GJ`` (``anisotropic_section_model``). The ribs leave the sections free to
    warp and the middle of the span free to distort. ``end_ribs=0`` (the default) gives
    the bare cantilever. On a faceted curved section, flat Quad8 elements (not Quad4 or
    curved Quad8) deform in their plane under torsion and the ribs stiffen them (a thick
    tube reads 5% high), so use ribs with Quad4 or curved Quad8 there.

    ``warping=True`` also measures the warping stiffness for ``stiffness_7x7``: one more
    run holds every mesh section rigid in its plane (free to warp, as Vlasov and
    Benscoter assume), clamps the root and twists the tip, and Benscoter's closed form
    is fitted to the twist along the span with GJ from the 6 x 6, giving EΓ and the
    warping shear stiffness S.
    """
    from shell_fem.pre_processing.element_library.connectors import add_reference_node, distributing_link, rigid_link
    from shell_fem.pre_processing.containers.boundary_conditions import BoundaryConditions
    from shell_fem.pre_processing.load_library.load_case import LoadCase
    from shell_fem.simulation_runner.static.linear_static_simulation import StructuralGlobalSolver

    mesh = blade.mesh.copy()
    zs = np.asarray(blade.span_positions)
    z0, z1 = float(zs[0]), float(zs[-1])
    if z_range is None:
        z_range = (z0 + 0.4 * (z1 - z0), z0 + 0.6 * (z1 - z0))
    za, zb = (float(zs[int(np.argmin(np.abs(zs - z)))]) for z in z_range)
    if zb <= za:
        raise ValueError("z_range must span at least one mesh section")
    ref = (0.0, 0.0) if reference is None else (float(reference[0]), float(reference[1]))
    bcs = BoundaryConditions().clamp(blade.nodes("root"))
    tip = add_reference_node(mesh, (ref[0], ref[1], z1))
    rigid_link(mesh, bcs, tip, blade.nodes("tip"), dofs=(0, 1, 2))  # a rigid tip diaphragm
    if end_ribs > 0.0:
        for z in zs[1:-1]:
            if z - z0 <= end_ribs * (z1 - z0) or z1 - z <= end_ribs * (z1 - z0):
                rib = add_reference_node(mesh, (ref[0], ref[1], float(z)))
                rigid_link(mesh, bcs, rib, blade.section_nodes(float(z)), dofs=(0, 1))
                bcs.fix([rib], [2, 3, 4])
    probes = []
    for z in (za, zb):
        n = add_reference_node(mesh, (ref[0], ref[1], z))
        nodes = blade.section_nodes(z)
        if section_weighting == "stiffness":
            weights = _section_weights(mesh, z, nodes)
        elif section_weighting == "uniform":
            weights = None
        else:
            raise ValueError("section_weighting must be 'stiffness' or 'uniform'")
        distributing_link(mesh, bcs, n, nodes, weights)
        probes.append(n)
    cases = []
    for dof in (2, 3, 4, 5, 0, 1):  # Fz, Mx, My, Mz, then shear Fx, Fy
        cases.append(LoadCase(f"unit_{dof}").add_nodal_load(tip, dof, 1.0))
    solver = solver or StructuralGlobalSolver()
    sols = solver.solve_many(mesh, bcs, cases)
    L = zb - za

    def strains(sol, curvature_gradient=(0.0, 0.0)):
        """Mean [gamma_x, gamma_y, eps, kappa_x, kappa_y, kappa_z] over z_range.

        The mean section rotation is the trapezoidal mean of the end rotations, exact
        for a linear rotation and corrected by -L^2/12 theta'' when the curvature varies
        linearly (``curvature_gradient`` = (kappa_x', kappa_y')).
        """
        ua, ub = sol.node_displacement(probes[0]), sol.node_displacement(probes[1])
        d = (ub - ua) / L
        th = 0.5 * (ua[3:5] + ub[3:5]) - L**2 / 12.0 * np.asarray(curvature_gradient)
        return np.array([d[0] - th[1], d[1] + th[0], d[2], d[3], d[4], d[5]])

    C = np.zeros((6, 6))  # compliance, BeamDyn order
    for k, sol in zip((2, 3, 4, 5), sols[:4]):
        C[:, k] = strains(sol)
    arm = z1 - 0.5 * (za + zb)
    for k, sol, dM in ((0, sols[4], np.array([0.0, -1.0])), (1, sols[5], np.array([1.0, 0.0]))):
        # unit Fx: My = (z1 - z), dMy/dz = -1; unit Fy: Mx = -(z1 - z), dMx/dz = +1
        grad = C[3:5, 3:5] @ dM
        e = strains(sol, grad)
        M_mid = np.array([-arm, 0.0]) if k == 1 else np.array([0.0, arm])  # [Mx, My] at mid z_range
        C[:, k] = e - C[:, 3:5] @ M_mid
    C = 0.5 * (C + C.T)
    K6 = np.linalg.inv(C)
    K6 = 0.5 * (K6 + K6.T)
    C4 = C[2:, 2:]
    S = np.linalg.inv(C4)
    S = 0.5 * (S + S.T)
    # tension centre: Fz at (x, y) adds Mx = y Fz, My = -x Fz; no curvature there
    Mt = np.column_stack([-C4[1:3, 2], C4[1:3, 1]])
    xt, yt = np.linalg.solve(Mt, -C4[1:3, 0])
    # shear centre: a shear force there causes no twist (Fx at (x, y) adds Mz = -y Fx, Fy adds x Fy)
    x_sc = -C[5, 1] / C[5, 5]
    y_sc = C[5, 0] / C[5, 5]
    # mass per unit length between the probes
    mass, moment = 0.0, np.zeros(2)
    for e in mesh.shell_elements:
        c = np.asarray(e.centroid)
        if za - 1e-12 <= c[2] <= zb + 1e-12:
            rho_a = sum(p.thickness * p.material.density for p in e.laminate.plies)
            mid = c + e.laminate.mid_plane * np.asarray(e.frame)[2]
            mass += rho_a * e.area
            moment += rho_a * e.area * mid[:2]
    m = mass / L
    centre = tuple(moment / mass) if mass > 0 else (float("nan"), float("nan"))
    K7, S_w = None, None
    if warping:
        EGamma, S_w, _ = _fit_warping_stiffness(blade, (x_sc + ref[0], y_sc + ref[1]), solver)
        K7 = np.zeros((7, 7))
        K7[:6, :6] = K6
        K7[6, 6] = EGamma
    return BeamSectionProperties((za, zb), ref, S, (float(xt + ref[0]), float(yt + ref[1])),
                                 (float(x_sc + ref[0]), float(y_sc + ref[1])), float(m), centre, K6, K7, S_w)


def _section_weights(mesh, z: float, nodes: Sequence[int]) -> List[float]:
    """Axial stiffness share of each section node: Et (= 1 / (A^-1)_11 along the beam axis)
    times its share of the element edges on the section (Simpson weights on quadratic edges)."""
    from laminate_constitutive_model.laminate_kinematics import integrate_abd

    weight = {int(n): 0.0 for n in nodes}
    scale = max(1.0, abs(z))
    cache: Dict[int, float] = {}
    for e in mesh.shell_elements:
        ids = list(e.node_ids)
        X = np.array([mesh.node_coordinates(n) for n in ids])
        on = np.abs(X[:, 2] - z) <= 1e-9 * scale
        if on.sum() < 2:
            continue
        key = id(e.laminate)
        if key not in cache:
            a = np.linalg.inv(integrate_abd(Laminate(e.laminate.plies))[:3, :3])  # membrane, wall not bending
            frame = np.asarray(e.frame)
            c2 = frame[0, 2] ** 2  # material axis 1 along the beam axis: c2 = 1
            cache[key] = 1.0 / (c2 * a[0, 0] + (1.0 - c2) * a[1, 1])
        Et = cache[key]
        nc = 4 if len(ids) in (4, 8) else 3
        for k in range(nc):
            a_, b_ = k, (k + 1) % nc
            if not (on[a_] and on[b_]):
                continue
            length = float(np.linalg.norm(X[b_] - X[a_]))
            if len(ids) == 2 * nc:
                for n, f in ((ids[a_], 1.0 / 6.0), (ids[nc + k], 4.0 / 6.0), (ids[b_], 1.0 / 6.0)):
                    if n in weight:
                        weight[n] += Et * length * f
            else:
                for n in (ids[a_], ids[b_]):
                    if n in weight:
                        weight[n] += 0.5 * Et * length
    return [weight[int(n)] for n in nodes]


def _fit_warping_stiffness(blade: BladeMesh, shear_centre, solver) -> Tuple[float, Optional[float], float]:
    """EΓ and Benscoter's S from the restrained twist of the shell with rigid sections.

    Every mesh section is tied rigid in its plane (free to warp) to a node on the shear
    centre, the root is clamped and a unit torque twists the tip. Benscoter's closed form
    for a cantilever, GJ theta(z) / T = (1 - rho) z + rho (z - (sinh k(z - L) + sinh kL)
    / (k cosh kL)) with rho = S / (GJ + S), k^2 = GJ S / (EΓ (GJ + S)), is fitted to the
    twist (a scan on k, with GJ and rho by least squares); rho near 1 means Vlasov's
    theory holds (no shear strain from warping; S is then returned as ``None``). Returns
    (EΓ, S, GJ of the rigid-section run).
    """
    from shell_fem.pre_processing.element_library.connectors import add_reference_node, rigid_link
    from shell_fem.pre_processing.containers.boundary_conditions import BoundaryConditions
    from shell_fem.pre_processing.load_library.load_case import LoadCase

    mesh = blade.mesh.copy()
    zs = np.asarray(blade.span_positions)
    z0 = float(zs[0])
    bcs = BoundaryConditions().clamp(blade.nodes("root"))
    refs = []
    for z in zs[1:]:
        r = add_reference_node(mesh, (shear_centre[0], shear_centre[1], float(z)))
        rigid_link(mesh, bcs, r, blade.section_nodes(float(z)), dofs=(0, 1))
        bcs.fix([r], [2, 3, 4])
        refs.append(r)
    case = LoadCase("unit_torque").add_nodal_load(refs[-1], 5, 1.0)
    sol = solver.solve_many(mesh, bcs, [case])[0]
    z = zs[1:] - z0
    Lb = float(zs[-1] - z0)
    twist = np.array([sol.node_displacement(r)[5] for r in refs])  # per unit torque
    best = None
    for kL in np.geomspace(0.5, 400.0, 800):
        k = kL / Lb
        ea, eb = np.exp(-2.0 * k * (Lb - z)), np.exp(-2.0 * kL)
        sh = -np.exp(-k * z) * (1.0 - ea) / (1.0 + eb)  # sinh(k(z - L)) / cosh(kL)
        F = z - (sh + (1.0 - eb) / (1.0 + eb)) / k
        # twist = alpha z + beta F with alpha = (1 - rho) / GJ, beta = rho / GJ
        M = np.column_stack([z, F])
        coef, *_ = np.linalg.lstsq(M, twist, rcond=None)
        r = float(np.sum((M @ coef - twist) ** 2))
        if best is None or r < best[0]:
            best = (r, k, coef)
    _, k, (alpha, beta) = best
    GJ_rigid = 1.0 / (alpha + beta)
    rho = min(max(beta * GJ_rigid, 0.0), 1.0)
    if rho >= 1.0 - 1e-6:
        return float(GJ_rigid / k**2), None, float(GJ_rigid)
    if rho <= 0.0:
        raise ValueError("the shell's twist shows no warping restraint: EΓ cannot be measured on this segment")
    S = rho * GJ_rigid / (1.0 - rho)
    EGamma = GJ_rigid * S / (k**2 * (GJ_rigid + S))
    return float(EGamma), float(S), float(GJ_rigid)


__all__ = ["BeamSectionProperties", "beam_section_properties", "transform_beam_stiffness"]
