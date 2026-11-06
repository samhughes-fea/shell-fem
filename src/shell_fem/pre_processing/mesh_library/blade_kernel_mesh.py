"""Shell models of a blade from the geometry kernel's own shell mesh (``implicit_geometry_kernel.mesh``).

``KernelMeshBladeBuilder`` (made by ``BladeBuilder.from_shell_mesh``) takes a
``ShellMesh`` as it is, with no remeshing:

- skin and web quads become ``quad4`` elements on the kernel's nodes;
- each rib and tip-closure polygon (five or more nodes) becomes a fan of ``tri3``
  elements about a node at its centroid, so its perimeter keeps the skin and web
  nodes it shares;
- normals are the kernel's: out of the section on the skin, towards the trailing
  edge on webs, along +z on ribs and the tip closure.

Each element carries the partition the kernel cut it from. Its laminate is looked up
per layer by wall label, then the label before ``:``, then kind, as in
``BladeBuilder.from_geometry``: ``{"skin": ..., "cap": ..., "web": ..., "rib": ...}``
or ``{"skin_suction_surface:bay_0": ..., "shear_web_1": ..., "rib_2": ...}``. The
tip closure is skin: it looks up ``"tip_closure"`` and then ``"skin"``. A value may
be a ``Laminate`` or a function of z returning one (evaluated at the element's
centroid). Where a spar cap is bonded to the skin the element's layers are the cap
then the skin: their plies are stacked (cap inside) and the stack takes the offset
the kernel gives, ``-eccentricity``, so its mid-plane lies inside the skin
mid-surface where the nodes are. Skin, cap, rib and tip laminates must match the
mesh's layer thicknesses (within ``thickness_tol``).

``build_pcomp`` skips that lookup. It takes one :class:`PcompAssignment` per kernel
element: the plies already chosen for that partition, bottom to top, and
``offset = eccentricity / thickness`` of the full stack. The nodes stay where the
mesh put them. ``solve_kernel_pcomp`` (``simulation_runner.static.kernel_pcomp``)
runs the native static solver on that model.

Element sets: every region (``cap_su``, ``web_0``, ``rib_1:cell_0``, ...), partition
and layer label, the label before ``:``, ``skin`` (skins, including the capped
parts), ``caps``, ``webs``, ``ribs``, ``tip_closure`` and ``all``. Node sets:
``root``, ``tip`` (the outermost station), one per web partition, and ``all``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from laminate_constitutive_model.laminate_kinematics import KinematicModel
from laminate_constitutive_model.laminate_mechanics import Laminate, Ply


def _keys(surface: str, label: str, position: int, n_layers: int) -> Tuple[str, ...]:
    if surface == "tip_closure":
        return ("tip_closure", "skin")
    if surface == "skin":
        kind = "cap" if position < n_layers - 1 else "skin"
    else:
        kind = surface
    return tuple(dict.fromkeys((label, label.split(":")[0], kind)))


@dataclass(frozen=True)
class PcompAssignment:
    """One Nastran PCOMP on a kernel shell element.

    ``plies`` are bottom to top along the element normal, the PCOMP order.
    ``offset`` is how far the node surface sits above the laminate mid-plane,
    in units of the laminate thickness: 0 puts the nodes on the mid-plane.
    A spar cap whose nodes stay on the skin mid-surface uses
    ``eccentricity / thickness`` of the full stack (skin plus cap). The
    constitutive mid-plane is then ``-offset * thickness`` (inward when the
    fraction is positive and the normal points outward), and a Nastran deck
    writes ``Z0 = -(0.5 + offset) * thickness``.
    """

    plies: Tuple[Ply, ...]
    offset: float = 0.0
    name: str = ""

    def __post_init__(self) -> None:
        plies = tuple(self.plies)
        if not plies or not all(isinstance(p, Ply) for p in plies):
            raise TypeError("PcompAssignment.plies must be one or more Ply instances")
        if not np.isfinite(self.offset):
            raise ValueError("PcompAssignment.offset must be finite")
        object.__setattr__(self, "plies", plies)

    def laminate(self) -> Laminate:
        """The stack with its mid-plane at ``-offset * thickness``."""
        thickness = float(sum(p.thickness for p in self.plies))
        return Laminate(self.plies, self.name, offset=-float(self.offset) * thickness)


class KernelMeshBladeBuilder:
    """Shell model of a blade on the geometry kernel's ``ShellMesh``.

    Parameters
    ----------
    shell_mesh:
        An ``implicit_geometry_kernel.mesh.ShellMesh`` (``shell_mesh(geo, h=...)``).
    geometry:
        The ``BladeGeometry`` it was meshed from (optional; kept on the result).
    material_axis:
        The 0-degree ply direction, projected onto each element (the span axis by
        default).
    """

    def __init__(self, shell_mesh, geometry=None, material_axis=(0.0, 0.0, 1.0)):
        if not len(shell_mesh.elements):
            raise ValueError("the shell mesh has no elements")
        self.shell_mesh = shell_mesh
        self.geometry = geometry
        self.material_axis = material_axis

    def build(self, laminates, kinematics: KinematicModel, thickness_tol: float = 0.01, **element_options):
        """The ``BladeMesh``. ``laminates`` maps labels, regions or kinds to laminates (or functions of z)."""
        sm = self.shell_mesh

        def laminate_of(keys: Sequence[str], z: float) -> Laminate:
            for key in keys:
                if hasattr(laminates, "get") and key in laminates:
                    lam = laminates[key]
                    break
            else:
                if isinstance(laminates, Laminate):
                    lam = laminates
                else:
                    raise KeyError(f"no laminate for '{keys[0]}' (tried {', '.join(map(repr, keys))})")
            if callable(lam) and not isinstance(lam, Laminate):
                lam = lam(z)
            return lam

        stacks: Dict[Tuple, Laminate] = {}

        def laminate_for(i: int, z: float) -> Laminate:
            surface, layers = sm.surface[i], tuple(sm.layers[i])
            parts = []
            for k, (label, t) in enumerate(zip(layers, sm.layer_thicknesses[i])):
                lam = laminate_of(_keys(surface, label, k, len(layers)), z)
                if surface != "web" and abs(lam.thickness - t) > thickness_tol * t:
                    raise ValueError(f"'{label}' is {t * 1e3:.4g} mm thick at z = {z:g} in the mesh "
                                     f"but its laminate is {lam.thickness * 1e3:.4g} mm")
                parts.append(lam)
            if len(parts) == 1:
                return parts[0]
            offset = parts[-1].offset - float(sm.eccentricity[i])
            key = (tuple(id(p) for p in parts), round(offset, 15))
            if key not in stacks:
                stacks[key] = Laminate([ply for p in parts for ply in p.plies], "+".join(p.name for p in parts),
                                       offset=offset)
            return stacks[key]

        caps = [sm.surface[i] == "skin" and len(sm.layers[i]) > 1 for i in range(len(sm.elements))]
        return self._assemble(laminate_for, kinematics, caps, None, element_options)

    def build_pcomp(self, assignments: Sequence[PcompAssignment], kinematics: KinematicModel,
                    material_axes=None, **element_options):
        """The ``BladeMesh`` from one PCOMP per kernel element.

        ``assignments`` follows ``shell_mesh.elements``. ``material_axes`` is one
        0-degree direction, or one direction per kernel element; each is projected
        onto the element. The assignment's ``offset`` is used as given: the kernel
        eccentricity is not applied a second time.
        """
        sm = self.shell_mesh
        assignments = tuple(assignments)
        if len(assignments) != len(sm.elements):
            raise ValueError(f"{len(assignments)} PCOMP assignments for {len(sm.elements)} kernel elements")
        if material_axes is not None:
            axes = np.asarray(material_axes, dtype=float)
            if axes.shape == (3,):
                material_axes = np.tile(axes, (len(assignments), 1))
            elif axes.shape != (len(assignments), 3):
                raise ValueError("material_axes must be one direction or one per kernel element")
            else:
                material_axes = axes

        def laminate_for(i: int, _z: float) -> Laminate:
            return assignments[i].laminate()

        caps = [sm.surface[i] == "skin" and assignments[i].offset != 0.0
                for i in range(len(sm.elements))]
        return self._assemble(laminate_for, kinematics, caps, material_axes, element_options)

    def _assemble(self, laminate_for, kinematics, caps, material_axes, element_options):
        from shell_fem.pre_processing.containers.mesh import FEAMesh
        from shell_fem.pre_processing.mesh_library.blade import BladeMesh


        sm = self.shell_mesh
        xyz = np.asarray(sm.nodes, dtype=float)
        mesh = FEAMesh()
        for k, p in enumerate(xyz):
            mesh.add_node(k + 1, p)
        next_node = len(xyz) + 1
        elements: Dict[str, List[int]] = {}
        source: List[Tuple[int, ...]] = []
        eid = 1
        for i, conn in enumerate(sm.elements):
            pts = xyz[list(conn)]
            lam = laminate_for(i, float(pts[:, 2].mean()))
            surface = sm.surface[i]
            layers = tuple(sm.layers[i])
            # the kernel mesh calls this ``name``; tidalshell's copy of the same mesh calls it ``region``
            region = sm.name[i] if hasattr(sm, "name") else sm.region[i]
            sets = [region, sm.partition[i]] + [v for label in layers for v in (label, label.split(":")[0])]
            sets.append({"skin": "skin", "web": "webs", "rib": "ribs"}.get(surface, surface))
            if caps[i]:
                sets.append("caps")
            sets = list(dict.fromkeys(sets))
            options = dict(element_options)
            if material_axes is None:
                options.setdefault("material_axis", self.material_axis)
            else:
                options["material_axis"] = material_axes[i]
            tri_options = {k: v for k, v in options.items() if k != "membrane"}
            ids = [k + 1 for k in conn]
            if len(ids) == 4:
                new = [eid]
                mesh.add_element(eid, ids, lam, kinematics, element_type="quad4", **options)
                eid += 1
            else:
                centre = next_node
                mesh.add_node(centre, pts.mean(axis=0))
                next_node += 1
                new = []
                for a, b in zip(ids, ids[1:] + ids[:1]):
                    mesh.add_element(eid, [centre, a, b], lam, kinematics, element_type="tri3", **tri_options)
                    new.append(eid)
                    eid += 1
            source.append(tuple(new))
            for name in sets:
                elements.setdefault(name, []).extend(new)

        all_nodes = mesh.node_ids
        zs = xyz[:, 2]
        z0, z1 = float(zs.min()), float(zs.max())
        tol = 1e-9 * max(1.0, abs(z0), abs(z1))
        node_sets: Dict[str, List[int]] = {
            "root": [k + 1 for k in np.flatnonzero(np.abs(zs - z0) <= tol)],
            "tip": [k + 1 for k in np.flatnonzero(np.abs(zs - z1) <= tol)],
            "all": all_nodes,
        }
        for i, surface in enumerate(sm.surface):
            if surface == "web":
                node_sets.setdefault(sm.partition[i], []).extend(k + 1 for k in sm.elements[i])
        node_sets = {k: list(dict.fromkeys(v)) for k, v in node_sets.items()}
        elements["all"] = mesh.element_ids
        stations = tuple(self.geometry.stations) if self.geometry is not None else ()
        return BladeMesh(mesh, node_sets, elements, stations, np.asarray(sm.stations, dtype=float),
                         geometry=self.geometry, shell_mesh=sm, source_elements=tuple(source))

    def __repr__(self) -> str:
        sm = self.shell_mesh
        return f"KernelMeshBladeBuilder({len(sm.elements)} kernel elements, {len(sm.stations)} stations)"


__all__ = ["KernelMeshBladeBuilder", "PcompAssignment"]
