"""Blade and wing structures: airfoil shell skins, spar caps and shear webs.

``BladeBuilder`` meshes a blade or wing segment between spanwise stations. Each
station has a chord, twist and airfoil (thickness can change along the span). The
skin is split chordwise into named regions (leading-edge panel, spar cap, trailing-
edge panel, ...) on both surfaces, and flat shear webs join the two surfaces at
chord positions that are region boundaries, so webs, caps and skins share nodes.

Axes: z runs along the span, x along the chord (leading edge to trailing edge) and
y through the thickness (suction side positive). The pitch axis sits at x = 0.
Skin normals point out of the section; web normals point to the trailing edge.

Element sets: ``<region>_ss`` (suction side), ``<region>_ps`` (pressure side) and
``<region>`` (both), ``<web>`` for each web, ``skin``, ``webs`` and ``all``.
Node sets: ``root`` and ``tip`` (end sections), ``le``, ``te``, ``<web>`` (web nodes),
``<web>_ss`` / ``<web>_ps`` (web-to-skin joint lines) and ``all``.

Laminates are given per region and web name; ``<region>_ss`` / ``<region>_ps``
override a side. Skins are laid up from the outer mould line by default
(``skin_reference="outer"``): the laminate grows inward from the airfoil surface, as
it does from a blade mould. The skins are then meshed on the laminates' mid-surface
(``mesh_surface="mid"``, the default), so flat facets keep the torsional stiffness
of thick sections; ``mesh_surface="outer"`` keeps the nodes on the mould line and
offsets the laminates instead. A ``Layup`` can be applied afterwards with the element
sets for ply-by-ply definitions (spar-cap ply drops along the span, for example);
call ``blade.to_mid_surface()`` after it so the mid-surface follows the new plies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from implicit_geometry_kernel import Airfoil, ShearWeb, SkinRegion, Station

from laminate_constitutive_model.laminate_kinematics import KinematicModel
from laminate_constitutive_model.laminate_mechanics import Laminate


@dataclass
class BladeMesh:
    """The mesh, its named sets and the section geometry used to build it."""

    mesh: "FEAMesh"  # noqa: F821
    node_sets: Dict[str, List[int]] = field(default_factory=dict)
    element_sets: Dict[str, List[int]] = field(default_factory=dict)
    stations: Tuple[Station, ...] = ()
    span_positions: np.ndarray = field(default_factory=lambda: np.zeros(0))  # element-edge sections
    # the same blade on the outer mould line, kept by a mid-surface mesh so that
    # ``to_mid_surface`` can redo the move after a ``Layup`` changes the plies
    mould_line: Optional["BladeMesh"] = field(default=None, repr=False, compare=False)
    _mid_laminates: Dict[int, Laminate] = field(default_factory=dict, repr=False, compare=False)
    _smoothing: Optional[float] = field(default=None, repr=False, compare=False)
    # the BladeGeometry a mesh from ``BladeBuilder.from_geometry`` was cut from
    geometry: Optional[object] = field(default=None, repr=False, compare=False)
    # the kernel ShellMesh a mesh from ``BladeBuilder.from_shell_mesh`` was built on
    shell_mesh: Optional[object] = field(default=None, repr=False, compare=False)
    # shell element ids created from each kernel element, in kernel order
    source_elements: Tuple[Tuple[int, ...], ...] = ()

    @property
    def surface(self) -> str:
        """``"mid"`` for skins meshed on their mid-surface, ``"outer"`` on the mould line."""
        on_mid = self.mould_line is not None or self.geometry is not None or self.shell_mesh is not None
        return "mid" if on_mid else "outer"

    def nodes(self, name: str) -> List[int]:
        try:
            return self.node_sets[name]
        except KeyError as exc:
            raise KeyError(f"No node set '{name}'. Available: {', '.join(self.node_sets)}") from exc

    def elements(self, name: str) -> List[int]:
        try:
            return self.element_sets[name]
        except KeyError as exc:
            raise KeyError(f"No element set '{name}'. Available: {', '.join(self.element_sets)}") from exc

    def section_nodes(self, z: float, tol: float = 1e-9) -> List[int]:
        """Nodes of the cross-section at span position ``z`` (a mesh section)."""
        return [n for n in self.node_sets["all"] if abs(self.mesh.node_coordinates(n)[2] - z) <= tol]

    def to_mid_surface(self, smoothing: Optional[float] = None) -> "BladeMesh":
        """The same blade meshed on the skin laminates' mid-surface instead of the mould line.

        Each skin node moves along the section normal by the offset of the laminates
        around it (inward by half the skin thickness for skins on the outer mould line),
        so the skin laminates need no offset. The shift is averaged over the elements at
        each node and then smoothed along the skin with a hat kernel ``smoothing`` long
        (two of the thickest skins by default), so the surface steps gently across
        thickness changes: region boundaries, spar-cap edges and ply drops applied with a
        ``Layup``. Elements in a blend keep the small residual offset that puts their
        plies where they were; elsewhere (and wherever it is under 5% of the
        thickness) the offset is zero. Web nodes are re-spaced
        between their joints, so the webs end on the skins' mid-surfaces.

        Nodes move within the cross-section plane (z is kept), so the root, tip and
        mesh sections stay where they were.

        On a blade that is already on its mid-surface (the ``BladeBuilder`` default),
        this redoes the move from the mould line: elements whose laminate was replaced
        since (by a ``Layup``, referenced to the mould line as laid in the mould) take
        their new plies, the others keep their original ones, and ``smoothing``
        defaults to the length used the first time. So after applying a ``Layup`` to
        a mid-surface blade, call ``blade = blade.to_mid_surface()``.
        """
        if self.geometry is not None:
            raise ValueError("a mesh built from a BladeGeometry is on its mid-surface already; change the geometry's "
                             "wall thicknesses and mesh it again instead")
        return _mid_surface(self, smoothing)


class BladeBuilder:
    """Shell model of a blade or wing segment: skins, spar caps and shear webs.

    Parameters
    ----------
    stations:
        Two or more ``Station`` objects in increasing ``z``. Chord, twist, pitch axis,
        offset and the airfoil surfaces are interpolated linearly between them.
    regions:
        ``SkinRegion`` objects that tile the chord from 0 (LE) to 1 (TE) without gaps;
        the same regions are used on both surfaces.
    webs:
        ``ShearWeb`` objects at region boundaries (chord fractions strictly inside).
    n_span:
        Elements along the span (an int), or increasing span fractions 0..1.
    element:
        ``"quad8_curved"`` (default), ``"quad8"`` or ``"quad4"``. Quadratic elements
        put their mid-side nodes on the airfoil surface. Flat facets (``"quad4"``,
        ``"quad8"``) under-predict the torsional stiffness of thick-walled sections
        if the skins stay on the mould line with offset laminates
        (``mesh_surface="outer"``); on the mid-surface (the default) they do not.
    trailing_edge:
        ``"auto"`` joins the surfaces at a sharp trailing edge and adds a flat
        ``te_web`` across a blunt one; ``"web"`` always adds it.
    material_axis:
        The 0-degree ply direction, projected onto each element: the span axis by
        default, so 0-degree plies run along the blade. Ply angles turn about each
        element's normal (outward on the skins), so the same angle on the suction and
        pressure sides is mirrored when seen from one side of the blade: equal angles
        give extension-twist coupling, opposite angles bend-twist coupling.
    mesh_surface:
        ``"mid"`` (default) meshes the skins on the laminates' mid-surface, smoothed
        across thickness changes over ``mid_surface_smoothing`` (see
        ``BladeMesh.to_mid_surface``), with the webs between the skins'
        mid-surfaces. Every element type then keeps the torsional stiffness of thick
        closed sections. ``"outer"`` meshes the skins on the outer mould line and
        offsets their laminates (see ``skin_reference``), which keeps the true outer
        geometry but makes flat facets (``"quad4"``, ``"quad8"``) too soft in torsion
        where the skins are thick. After a ``Layup``, call ``blade.to_mid_surface()``
        so the mid-surface follows the new plies.
    """

    def __init__(
        self,
        stations: Sequence[Station],
        regions: Sequence[SkinRegion],
        webs: Sequence[ShearWeb] = (),
        n_span: Union[int, Sequence[float]] = 10,
        element: str = "quad8_curved",
        skin_reference: str = "outer",
        trailing_edge: str = "auto",
        te_web_elements: int = 2,
        material_axis=(0.0, 0.0, 1.0),
        mesh_surface: str = "mid",
        mid_surface_smoothing: Optional[float] = None,
    ):
        self.material_axis = material_axis
        if mesh_surface not in ("outer", "mid"):
            raise ValueError("mesh_surface must be 'outer' or 'mid'")
        self.mesh_surface = mesh_surface
        self.mid_surface_smoothing = mid_surface_smoothing
        self.stations = tuple(sorted(stations, key=lambda s: s.z))
        if len(self.stations) < 2:
            raise ValueError("a blade needs at least two stations")
        if np.any(np.diff([s.z for s in self.stations]) <= 0.0):
            raise ValueError("stations need distinct span positions")
        self.regions = tuple(sorted(regions, key=lambda r: r.start))
        if not self.regions or abs(self.regions[0].start) > 1e-12 or abs(self.regions[-1].end - 1.0) > 1e-12:
            raise ValueError("skin regions must cover the chord from 0 to 1")
        for a, b in zip(self.regions[:-1], self.regions[1:]):
            if abs(a.end - b.start) > 1e-12:
                raise ValueError(f"regions '{a.name}' and '{b.name}' leave a gap or overlap")
        names = [r.name for r in self.regions]
        if len(set(names)) != len(names):
            raise ValueError("region names must be unique")
        bounds = {r.start for r in self.regions[1:]}
        self.webs = tuple(sorted(webs, key=lambda w: w.position))
        for w in self.webs:
            if not any(abs(w.position - b) < 1e-12 for b in bounds):
                raise ValueError(f"web '{w.name}' at {w.position} must sit on a region boundary {sorted(bounds)}")
        if element not in ("quad4", "quad8", "quad8_curved"):
            raise ValueError("element must be 'quad4', 'quad8' or 'quad8_curved'")
        if skin_reference not in ("outer", "mid", "inner"):
            raise ValueError("skin_reference must be 'outer', 'mid' or 'inner'")
        if trailing_edge not in ("auto", "web"):
            raise ValueError("trailing_edge must be 'auto' or 'web'")
        self.element = element
        self.skin_reference = skin_reference
        self.trailing_edge = trailing_edge
        self.te_web_elements = int(te_web_elements)
        if np.ndim(n_span) == 0:
            self.span_fractions = np.linspace(0.0, 1.0, int(n_span) + 1)
        else:
            self.span_fractions = np.asarray(n_span, dtype=float)
        f = self.span_fractions
        if f.size < 2 or abs(f[0]) > 1e-12 or abs(f[-1] - 1.0) > 1e-12 or np.any(np.diff(f) <= 0):
            raise ValueError("n_span must be >= 1 or increasing fractions from 0 to 1")

    @staticmethod
    def from_geometry(geometry, n_span: Union[int, Sequence[float]] = 10, element: str = "quad8_curved",
                      element_size: Optional[float] = None, wall_elements: Optional[Dict[str, int]] = None,
                      material_axis=(0.0, 0.0, 1.0)) -> "GeometryBladeBuilder":  # noqa: F821
        """Mesh a ``implicit_geometry_kernel.BladeGeometry`` instead of explicit regions and webs.

        See ``GeometryBladeBuilder``: the shells lie on the walls' mid-surfaces cut from
        the geometry along the span, with spar caps merged into their skins.
        """
        from shell_fem.pre_processing.mesh_library.blade_geometry import GeometryBladeBuilder

        return GeometryBladeBuilder(geometry, n_span, element, element_size, wall_elements, material_axis)

    @staticmethod
    def from_shell_mesh(shell_mesh, geometry=None, material_axis=(0.0, 0.0, 1.0)) -> "KernelMeshBladeBuilder":  # noqa: F821
        """Use the geometry kernel's own ``ShellMesh`` (``implicit_geometry_kernel.mesh.shell_mesh``).

        See ``KernelMeshBladeBuilder``: kernel quads become ``quad4`` elements and rib and
        tip polygons ``tri3`` fans, on the kernel's nodes, with no remeshing.
        ``build`` looks laminates up by label. ``build_pcomp`` takes one PCOMP per kernel
        element, including the cap offset ``eccentricity / thickness``.
        """
        from shell_fem.pre_processing.mesh_library.blade_kernel_mesh import KernelMeshBladeBuilder

        return KernelMeshBladeBuilder(shell_mesh, geometry, material_axis)

    # ------------------------------------------------------------ geometry
    def _station_at(self, z: float):
        zs = np.array([s.z for s in self.stations])
        k = int(np.clip(np.searchsorted(zs, z, side="right") - 1, 0, len(zs) - 2))
        a, b = self.stations[k], self.stations[k + 1]
        w = float(np.clip((z - a.z) / (b.z - a.z), 0.0, 1.0))
        lerp = lambda p, q: (1 - w) * p + w * q  # noqa: E731
        return a, b, w, lerp(a.chord, b.chord), lerp(a.twist, b.twist), lerp(a.pitch_axis, b.pitch_axis), \
            lerp(np.asarray(a.offset, float), np.asarray(b.offset, float))

    def point(self, side: str, x_c: float, z: float) -> np.ndarray:
        """Global position of the chord fraction ``x_c`` on ``side`` ("upper"/"lower") at span z."""
        a, b, w, chord, twist, pitch, offset = self._station_at(z)
        y_c = (1 - w) * a.airfoil.surface(side, x_c) + w * b.airfoil.surface(side, x_c)
        return self._place(x_c, float(y_c), chord, twist, pitch, offset, z)

    @staticmethod
    def _place(x_c, y_c, chord, twist, pitch, offset, z) -> np.ndarray:
        x, y = (x_c - pitch) * chord, y_c * chord
        t = np.radians(twist)
        c, s = np.cos(t), np.sin(t)  # positive twist is nose up: the LE (x < 0) moves to +y
        return np.array([c * x + s * y + offset[0], -s * x + c * y + offset[1], z])

    def te_gap(self, z: float) -> float:
        return float(np.linalg.norm(self.point("upper", 1.0, z) - self.point("lower", 1.0, z)))

    # ---------------------------------------------------------------- build
    def _chord_parameters(self, quadratic: bool):
        """Per chordwise node index: (region, s in [0, 1]); and each region's index span."""
        out = [(self.regions[0], 0.0)]
        spans = {}
        for r in self.regions:
            n = r.n * (2 if quadratic else 1)
            start = len(out) - 1
            out.extend((r, float(v)) for v in np.linspace(0.0, 1.0, n + 1)[1:])
            spans[r.name] = (start, start + n)
        return out, spans

    def _x_at(self, side: str, region: SkinRegion, s: float, z: float) -> float:
        """Chord fraction at arc-length fraction ``s`` of ``region`` on ``side`` (equal element lengths)."""
        if s <= 0.0:
            return region.start
        if s >= 1.0:
            return region.end
        key = (side, region.name, round(z, 12))
        table = self._arc_cache.get(key)
        if table is None:
            # sample densely, finer towards the LE where the surface turns fastest
            u = np.linspace(0.0, 1.0, 801)
            x = region.start + (region.end - region.start) * (1.0 - np.cos(0.5 * np.pi * u) if region.start == 0.0 else u)
            pts = np.array([self.point(side, float(v), z) for v in x])
            arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
            table = (arc / arc[-1], x)
            self._arc_cache[key] = table
        return float(np.interp(s, *table))

    def build(self, laminates, kinematics: KinematicModel, **element_options) -> BladeMesh:
        """Mesh the blade; ``laminates`` maps region / web names (or ``"<region>_ss"``) to laminates.

        One ``Laminate`` for everything is accepted too.
        """
        from shell_fem.pre_processing.containers.mesh import FEAMesh

        element_options = dict(element_options)
        element_options.setdefault("material_axis", self.material_axis)
        quadratic = self.element != "quad4"
        step = 2 if quadratic else 1
        etype = "quad8_curved" if self.element == "quad8_curved" else None
        self._arc_cache = {}
        params, spans = self._chord_parameters(quadratic)
        xc = np.array([r.start + v * (r.end - r.start) for r, v in params])  # nominal, for web lookup
        sf = self.span_fractions
        z0, z1 = self.stations[0].z, self.stations[-1].z
        zf = sf if not quadratic else np.concatenate([[sf[0]], np.column_stack([0.5 * (sf[:-1] + sf[1:]), sf[1:]]).ravel()])
        z_nodes = z0 + zf * (z1 - z0)
        nz = len(sf) - 1
        mesh = FEAMesh()
        node = 1
        grid: Dict[Tuple[str, int, int], int] = {}
        sharp = self.trailing_edge == "auto" and max(self.te_gap(z0), self.te_gap(z1)) < 1e-9 * max(
            s.chord for s in self.stations)
        imax = len(xc) - 1
        for j, z in enumerate(z_nodes):
            for i, (region, sv) in enumerate(params):
                if quadratic and i % 2 and j % 2:
                    continue
                for side in ("upper", "lower"):
                    if side == "lower" and (i == 0 or (sharp and i == imax)):
                        grid[(side, i, j)] = grid[("upper", i, j)]
                        continue
                    mesh.add_node(node, self.point(side, self._x_at(side, region, sv, float(z)), float(z)))
                    grid[(side, i, j)] = node
                    node += 1

        elements: Dict[str, List[int]] = {}
        eid = 1

        def lam_for(name: str, side: Optional[str], skin: bool) -> Laminate:
            if isinstance(laminates, Laminate):
                lam = laminates
            else:
                key = f"{name}_{side}" if side else name
                lam = laminates.get(key, laminates.get(name)) if hasattr(laminates, "get") else None
                if lam is None:
                    raise KeyError(f"No laminate for '{key}'. Give one for '{name}' or '{key}'")
            if skin and self.skin_reference != "mid":
                h = lam.thickness
                lam = Laminate(lam.plies, lam.name, offset=-0.5 * h if self.skin_reference == "outer" else 0.5 * h)
            return lam

        def quad(nodes_ij, side):
            """Corner + mid-side node ids of the cell with corner grid indices (i, j)."""
            i, j = nodes_ij
            g = lambda a, b: grid[(side, a, b)]  # noqa: E731
            d = step
            corners = [g(i, j), g(i + d, j), g(i + d, j + d), g(i, j + d)]
            if not quadratic:
                return corners
            return corners + [g(i + 1, j), g(i + d, j + 1), g(i + 1, j + d), g(i, j + 1)]

        def flip(conn):
            if len(conn) == 4:
                return [conn[0], conn[3], conn[2], conn[1]]
            return [conn[0], conn[3], conn[2], conn[1], conn[7], conn[6], conn[5], conn[4]]

        for r in self.regions:
            a, b = spans[r.name]
            for side, tag in (("upper", "ss"), ("lower", "ps")):
                lam = lam_for(r.name, tag, True)
                for jj in range(nz):
                    for ii in range(a, b, step):
                        conn = quad((ii, step * jj), side)
                        # chordwise x span: (+chord) x (+z) points into the section on the upper side
                        if side == "upper":
                            conn = flip(conn)
                        mesh.add_element(eid, conn, lam, kinematics, element_type=etype, **element_options)
                        elements.setdefault(f"{r.name}_{tag}", []).append(eid)
                        elements.setdefault(r.name, []).append(eid)
                        elements.setdefault("skin", []).append(eid)
                        eid += 1

        web_nodes: Dict[str, List[int]] = {}
        joint: Dict[str, List[int]] = {}
        web_defs = list(self.webs)
        if not sharp:
            web_defs.append(ShearWeb("te_web", 1.0, self.te_web_elements))
        for web in web_defs:
            i = int(np.argmin(np.abs(xc - web.position)))
            lam = lam_for(web.name, None, False)
            m = web.n * step
            col: Dict[Tuple[int, int], int] = {}
            for j, z in enumerate(z_nodes):
                top, bot = mesh.node_coordinates(grid[("upper", i, j)]), mesh.node_coordinates(grid[("lower", i, j)])
                for k in range(m + 1):
                    if quadratic and k % 2 and j % 2:
                        continue
                    if k == 0:
                        col[(k, j)] = grid[("lower", i, j)]
                    elif k == m:
                        col[(k, j)] = grid[("upper", i, j)]
                    else:
                        mesh.add_node(node, bot + (top - bot) * k / m)
                        col[(k, j)] = node
                        node += 1
            for jj in range(nz):
                for kk in range(0, m, step):
                    j = step * jj
                    g = lambda a, b: col[(a, b)]  # noqa: E731
                    d = step
                    conn = [g(kk, j), g(kk + d, j), g(kk + d, j + d), g(kk, j + d)]
                    if quadratic:
                        conn += [g(kk + 1, j), g(kk + d, j + 1), g(kk + 1, j + d), g(kk, j + 1)]
                    # lower -> upper (+y) x (+z) points to +x (the TE) for an untwisted section
                    mesh.add_element(eid, conn, lam, kinematics, element_type=etype, **element_options)
                    elements.setdefault(web.name, []).append(eid)
                    elements.setdefault("webs", []).append(eid)
                    eid += 1
            web_nodes[web.name] = list(dict.fromkeys(col.values()))
            joint[f"{web.name}_ss"] = [col[(m, j)] for j in range(len(z_nodes))]
            joint[f"{web.name}_ps"] = [col[(0, j)] for j in range(len(z_nodes))]

        all_nodes = mesh.node_ids
        sets: Dict[str, List[int]] = {
            "root": [n for n in all_nodes if abs(mesh.node_coordinates(n)[2] - z0) < 1e-12 * max(1.0, abs(z1))],
            "tip": [n for n in all_nodes if abs(mesh.node_coordinates(n)[2] - z1) < 1e-12 * max(1.0, abs(z1))],
            "le": list(dict.fromkeys(grid[("upper", 0, j)] for j in range(len(z_nodes)))),
            "te": list(dict.fromkeys(v for j in range(len(z_nodes)) for v in
                                     (grid[("upper", imax, j)], grid[("lower", imax, j)]))),
            "all": all_nodes,
        }
        sets.update(web_nodes)
        sets.update(joint)
        elements["all"] = mesh.element_ids
        blade = BladeMesh(mesh, sets, elements, self.stations, z0 + sf * (z1 - z0))
        if self.mesh_surface == "mid":
            blade = blade.to_mid_surface(self.mid_surface_smoothing)
        return blade


# ---------------------------------------------------------------------------
# Mid-surface meshes
# ---------------------------------------------------------------------------
def _node_normals(mesh, element_ids) -> Dict[int, np.ndarray]:
    """Area-weighted mean element normal at each node, projected into the section plane."""
    acc: Dict[int, np.ndarray] = {}
    for eid in element_ids:
        e = mesh.element(eid)
        n = np.asarray(e.frame)[2] * e.area
        for nid in e.node_ids:
            acc[nid] = acc.get(nid, 0.0) + n
    out = {}
    for nid, v in acc.items():
        v = np.array([v[0], v[1], 0.0])
        out[nid] = v / max(np.linalg.norm(v), 1e-300)
    return out


def _smooth_on_surface(mesh, element_ids, values: Dict[int, float], weights: Dict[int, float],
                       length: float) -> Dict[int, float]:
    """Hat-kernel average of nodal values over surface paths shorter than ``length``."""
    if length <= 0.0:
        return dict(values)
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra
    except ImportError:  # pragma: no cover - SciPy is an optional dependency
        return dict(values)
    ids = sorted(values)
    index = {n: k for k, n in enumerate(ids)}
    rows, cols, dist = [], [], []
    for eid in element_ids:
        nodes = [n for n in mesh.element(eid).node_ids if n in index]
        for a in nodes:
            for b in nodes:
                if a < b:
                    d = float(np.linalg.norm(mesh.node_coordinates(a) - mesh.node_coordinates(b)))
                    rows += [index[a], index[b]]
                    cols += [index[b], index[a]]
                    dist += [d, d]
    graph = coo_matrix((dist, (rows, cols)), shape=(len(ids), len(ids))).tocsr()
    w = np.array([weights[n] for n in ids])
    v = np.array([values[n] for n in ids])
    out = np.empty(len(ids))
    for start in range(0, len(ids), 256):  # in blocks: the distance matrix is dense
        block = np.arange(start, min(start + 256, len(ids)))
        D = dijkstra(graph, directed=False, indices=block, limit=length)
        W = np.where(np.isfinite(D), 1.0 - D / length, 0.0) * w[None, :]
        out[block] = (W @ v) / W.sum(axis=1)
    return dict(zip(ids, out))


RESIDUAL_OFFSET_TOL = 0.05  # residual offsets below this fraction of the thickness are dropped


def _mid_surface(blade: BladeMesh, smoothing: Optional[float]) -> BladeMesh:
    if blade.mould_line is None:
        return _move_to_mid_surface(blade, smoothing)
    # redo the move from the mould line, with the plies of any laminate replaced since
    source = blade.mould_line
    if smoothing is None:
        smoothing = blade._smoothing
    outer = _copy_mesh(source.mesh)
    for e in blade.mesh.elements:
        if blade._mid_laminates.get(e.element_id) is not e.laminate:
            outer.replace_element(outer.element(e.element_id).with_laminate(e.laminate))
    return _move_to_mid_surface(BladeMesh(outer, {k: list(v) for k, v in source.node_sets.items()},
                                          {k: list(v) for k, v in source.element_sets.items()},
                                          source.stations, source.span_positions), smoothing)


def _copy_mesh(mesh):
    from shell_fem.pre_processing.containers.mesh import FEAMesh

    new = FEAMesh(mesh.hygrothermal_engine, mesh.element_registry, mesh.solid_registry)
    for n in mesh.node_ids:
        new.add_node(n, mesh.node_coordinates(n).copy())
    for e in mesh.elements:
        new.insert_element(e)
    return new


def _move_to_mid_surface(blade: BladeMesh, smoothing: Optional[float]) -> BladeMesh:
    from shell_fem.pre_processing.containers.mesh import FEAMesh

    mesh = blade.mesh
    skin = list(blade.element_sets.get("skin", []))
    if not skin:
        return blade
    thick = max(mesh.element(e).laminate.thickness for e in skin)
    length = 2.0 * thick if smoothing is None else float(smoothing)

    # the shift of each skin node: the offset of the laminates around it, area-weighted
    normals = _node_normals(mesh, skin)
    total: Dict[int, float] = {}
    area: Dict[int, float] = {}
    for eid in skin:
        e = mesh.element(eid)
        w = e.area / len(e.node_ids)
        for nid in e.node_ids:
            total[nid] = total.get(nid, 0.0) + w * e.laminate.offset
            area[nid] = area.get(nid, 0.0) + w
    shift = {n: total[n] / area[n] for n in total}
    shift = _smooth_on_surface(mesh, skin, shift, area, length)

    xyz = {n: mesh.node_coordinates(n).copy() for n in mesh.node_ids}
    for n, d in shift.items():
        xyz[n] = xyz[n] + d * normals[n]

    # web nodes: same fraction of the way between their (moved) joints at the same z
    skin_nodes = set(shift)
    for name in blade.node_sets:
        if f"{name}_ss" not in blade.node_sets or f"{name}_ps" not in blade.node_sets:
            continue
        top = {round(float(mesh.node_coordinates(n)[2]), 9): n for n in blade.node_sets[f"{name}_ss"]}
        bot = {round(float(mesh.node_coordinates(n)[2]), 9): n for n in blade.node_sets[f"{name}_ps"]}
        for n in blade.node_sets[name]:
            if n in skin_nodes:
                continue
            p = mesh.node_coordinates(n)
            z = round(float(p[2]), 9)
            a, b = mesh.node_coordinates(bot[z]), mesh.node_coordinates(top[z])
            f = float(np.dot(p - a, b - a) / max(np.dot(b - a, b - a), 1e-300))
            xyz[n] = xyz[bot[z]] + f * (xyz[top[z]] - xyz[bot[z]])

    new = FEAMesh(mesh.hygrothermal_engine, mesh.element_registry, mesh.solid_registry)
    for n in mesh.node_ids:
        new.add_node(n, xyz[n])
    skin_set = set(skin)
    for e in mesh.elements:
        lam = e.laminate
        if e.element_id in skin_set:
            # the part of the node moves along this element's normal
            e3 = np.asarray(e.frame)[2]
            moved = float(np.mean([shift[n] * float(np.dot(normals[n], e3)) for n in e.node_ids]))
            residual = lam.offset - moved
            # residuals under 5% of the thickness are rounding of the smoothing: dropping
            # them keeps neighbouring elements on one shell section (CalculiX expands
            # shells with differing offsets at a node into knots, and runs out of nodes
            # when there are many)
            lam = lam.with_offset(0.0 if abs(residual) < RESIDUAL_OFFSET_TOL * lam.thickness else residual)
        coords = np.array([xyz[n] for n in e.node_ids])
        try:
            moved_element = type(e)(e.element_id, e.node_ids, coords, lam, e._kinematics, **e._options())
        except ValueError as exc:
            raise ValueError(f"Element {e.element_id} folds over on the mid-surface (a narrow element between "
                             f"skins of different thickness); smooth over a longer length than {length:g} "
                             "(mid_surface_smoothing) or widen the element") from exc
        new.insert_element(moved_element)
    return BladeMesh(new, {k: list(v) for k, v in blade.node_sets.items()},
                     {k: list(v) for k, v in blade.element_sets.items()}, blade.stations, blade.span_positions,
                     mould_line=BladeMesh(_copy_mesh(mesh), blade.node_sets, blade.element_sets, blade.stations,
                                          blade.span_positions),
                     _mid_laminates={e.element_id: e.laminate for e in new.elements}, _smoothing=smoothing)


__all__ = ["Airfoil", "Station", "SkinRegion", "ShearWeb", "BladeBuilder", "BladeMesh"]
