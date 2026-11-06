"""The finite element mesh and its degree-of-freedom numbering.

Every node has the six shell DOFs [u, v, w, theta_x, theta_y, theta_z]; nodes of
enriched elements (full RZT, full layerwise) carry their extra in-plane amplitudes
after them. ``FEAMesh`` holds nodes, shell and solid elements, element and node sets,
and maps (node, DOF) pairs to global equation numbers.
"""

from __future__ import annotations

from enum import Enum, IntEnum
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from shell_fem.pre_processing.element_library.shell import DEFAULT_ELEMENTS, STANDARD_DOFS, ElementRegistry, ShellElement
from shell_fem.pre_processing.element_library import curved_shell as _curved  # noqa: F401  (registers the curved elements)
from shell_fem.pre_processing.element_library import solid as _solid
from laminate_constitutive_model.laminate_hygrothermal import HygrothermalEngine
from laminate_constitutive_model.laminate_kinematics import KinematicModel
from laminate_constitutive_model.laminate_mechanics import Laminate


class DOF(IntEnum):
    UX = 0
    UY = 1
    UZ = 2
    RX = 3
    RY = 4
    RZ = 5


class EnrichedDOF(str, Enum):
    """Symbolic enrichment DOFs, expanded per node when constraints are applied.

    ``X`` / ``Y`` select the x / y components of every enrichment amplitude at the
    node (zig-zag or interface displacements along local x / y); ``ALL`` selects
    both. Nodes without enrichment simply have none, so these can be used on any mesh.
    On a simply supported edge, fix the component tangential to the edge.
    """

    X = "enriched_x"
    Y = "enriched_y"
    ALL = "enriched_all"


ALL_DOFS = tuple(DOF)
DofLike = Union[int, EnrichedDOF]
DofKey = Tuple[int, DofLike]


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------
class FEAMesh:
    """Nodes, shell elements and layered solid elements with a global DOF map.

    Shells are created by an ``ElementRegistry`` (Quad4, Tri3 and Quad8 by default,
    chosen by node count; curved Quad8 and Tri6 by name), so new element types plug in
    without touching the mesh. Layered solids (Hex8, Hex20) are added with
    ``add_solid``. Every node carries the six shell DOFs; the rotations of nodes used
    only by solids have no stiffness and are held at zero by the solver.
    """

    def __init__(
        self,
        hygrothermal_engine: Optional[HygrothermalEngine] = None,
        element_registry: Optional[ElementRegistry] = None,
        solid_registry=None,
    ):
        self._nodes: Dict[int, np.ndarray] = {}
        self._elements: Dict[int, ShellElement] = {}
        self.hygrothermal_engine = hygrothermal_engine or HygrothermalEngine()
        self.element_registry = element_registry or DEFAULT_ELEMENTS
        self.solid_registry = solid_registry or _solid.DEFAULT_SOLIDS
        self._layout: Optional[Tuple[Dict[int, int], Dict[int, int], int]] = None

    # ---------------------------------------------------------------- nodes
    def add_node(self, node_id: int, xyz: Sequence[float]) -> None:
        node_id = int(node_id)
        if node_id in self._nodes:
            raise ValueError(f"Node {node_id} already exists")
        xyz = np.asarray(xyz, dtype=float).ravel()
        if xyz.size == 2:
            xyz = np.append(xyz, 0.0)
        if xyz.size != 3:
            raise ValueError("Node coordinates need 2 or 3 components")
        self._nodes[node_id] = xyz
        self._layout = None

    def remove_nodes(self, node_ids: Sequence[int]) -> None:
        """Remove nodes that no element uses and renumber the DOF map."""
        drop = {int(n) for n in node_ids}
        used = {n for element in self._elements.values() for n in element.node_ids} & drop
        if used:
            raise ValueError(f"Nodes {sorted(used)} are used by elements and cannot be removed")
        self._nodes = {n: xyz for n, xyz in self._nodes.items() if n not in drop}
        self._layout = None

    @property
    def node_ids(self) -> List[int]:
        return list(self._nodes)

    def node_coordinates(self, node_id: int) -> np.ndarray:
        return self._nodes[node_id]

    def coordinates_array(self) -> np.ndarray:
        return np.array([self._nodes[n] for n in self._nodes])

    @property
    def n_nodes(self) -> int:
        return len(self._nodes)

    # ----------------------------------------------------------- DOF layout
    def _dof_layout(self) -> Tuple[Dict[int, int], Dict[int, int], int]:
        if self._layout is None:
            count = {n: STANDARD_DOFS for n in self._nodes}
            signature: Dict[int, tuple] = {}
            for element in self._elements.values():
                d = element.dofs_per_node
                if d == STANDARD_DOFS:
                    continue
                sig = element.enrichment_signature
                for n in element.node_ids:
                    if n in signature and signature[n] != sig:
                        raise ValueError(
                            f"Node {n} is shared by elements with different enrichments ({signature[n]} and {sig}); "
                            "full layerwise elements must have the same number of layers across shared nodes"
                        )
                    signature[n] = sig
                    count[n] = d
            offsets, total = {}, 0
            for n in self._nodes:
                offsets[n] = total
                total += count[n]
            self._layout = (offsets, count, total)
        return self._layout

    @property
    def n_dof(self) -> int:
        return self._dof_layout()[2]

    def node_dof_count(self, node_id: int) -> int:
        return self._dof_layout()[1][int(node_id)]

    def dof_index(self, node_id: int, dof: int) -> int:
        offsets, count, _ = self._dof_layout()
        node_id, dof = int(node_id), int(dof)
        if node_id not in offsets:
            raise KeyError(f"Node {node_id} is not in the mesh")
        if not 0 <= dof < count[node_id]:
            raise KeyError(f"Node {node_id} has {count[node_id]} DOFs; DOF {dof} does not exist")
        return offsets[node_id] + dof

    def expand_dof(self, node_id: int, dof: DofLike) -> List[int]:
        """Global indices selected by a numeric or symbolic (``EnrichedDOF``) DOF at a node."""
        if isinstance(dof, EnrichedDOF):
            offsets, count, _ = self._dof_layout()
            base, n_extra = offsets[int(node_id)], count[int(node_id)] - STANDARD_DOFS
            start = {EnrichedDOF.X: 0, EnrichedDOF.Y: 1, EnrichedDOF.ALL: 0}[dof]
            step = 1 if dof is EnrichedDOF.ALL else 2
            return [base + STANDARD_DOFS + k for k in range(start, n_extra, step)]
        return [self.dof_index(node_id, dof)]

    def standard_dof_indices(self) -> np.ndarray:
        """(n_nodes, 6) global indices of the six shell DOFs of every node, in node order."""
        offsets, _, _ = self._dof_layout()
        return np.array([[offsets[n] + d for d in range(STANDARD_DOFS)] for n in self._nodes], dtype=int).reshape(-1, 6)

    def standard_array(self, vector: np.ndarray) -> np.ndarray:
        """(n_nodes, 6) view of a global vector restricted to the six shell DOFs."""
        return np.asarray(vector)[self.standard_dof_indices()]

    # --------------------------------------------------------- node queries
    def nodes_where(self, predicate: Callable[[np.ndarray], bool]) -> List[int]:
        return [n for n, xyz in self._nodes.items() if predicate(xyz)]

    def nodes_on_line(self, axis: int, value: float, tol: float = 1e-9) -> List[int]:
        """Nodes whose coordinate ``axis`` (0=x, 1=y, 2=z) equals ``value``."""
        return self.nodes_where(lambda p: abs(p[axis] - value) <= tol)

    def closest_node(self, point: Sequence[float]) -> int:
        point = np.asarray(point, dtype=float)
        if point.size == 2:
            point = np.append(point, 0.0)
        ids = self.node_ids
        distances = np.linalg.norm(self.coordinates_array() - point, axis=1)
        return ids[int(np.argmin(distances))]

    # ------------------------------------------------------------- elements
    def add_element(
        self,
        element_id: int,
        node_ids: Sequence[int],
        laminate: Laminate,
        kinematics: KinematicModel,
        element_type: Optional[str] = None,
        **element_options,
    ) -> ShellElement:
        """Create an element through the registry (type inferred from the node count)."""
        element_id = int(element_id)
        if element_id in self._elements:
            raise ValueError(f"Element {element_id} already exists")
        missing = [n for n in node_ids if n not in self._nodes]
        if missing:
            raise KeyError(f"Element {element_id} references unknown nodes {missing}")
        coords = np.array([self._nodes[n] for n in node_ids])
        element_options.setdefault("hygrothermal_engine", self.hygrothermal_engine)
        element = self.element_registry.create(element_id, node_ids, coords, laminate, kinematics, element_type, **element_options)
        self._elements[element_id] = element
        self._layout = None
        return element

    def add_solid(
        self,
        element_id: int,
        node_ids: Sequence[int],
        laminate: Laminate,
        element_type: Optional[str] = None,
        material_axis=None,
        gradient_origin: Optional[float] = None,
    ):
        """Add a layered solid ("hex8" or "hex20", inferred from the node count).

        The plies of ``laminate`` stack from the bottom face (nodes 0-3) to the top face
        (nodes 4-7); ply angles are measured from ``material_axis`` (global x by default)
        about the stack direction. ``gradient_origin`` is the height (on the laminate's z
        axis) about which through-thickness temperature gradients act, for elements that
        hold part of a thicker laminate; it defaults to the element laminate mid-plane.
        """
        element_id = int(element_id)
        if element_id in self._elements:
            raise ValueError(f"Element {element_id} already exists")
        missing = [n for n in node_ids if n not in self._nodes]
        if missing:
            raise KeyError(f"Element {element_id} references unknown nodes {missing}")
        cls = self.solid_registry.resolve(len(node_ids), element_type)
        coords = np.array([self._nodes[n] for n in node_ids])
        element = cls(element_id, node_ids, coords, laminate, material_axis=material_axis,
                      hygrothermal_engine=self.hygrothermal_engine, gradient_origin=gradient_origin)
        self._elements[element_id] = element
        self._layout = None
        return element

    @property
    def shell_elements(self) -> List[ShellElement]:
        return [e for e in self._elements.values() if not getattr(e, "IS_SOLID", False)]

    @property
    def solid_elements(self) -> list:
        return [e for e in self._elements.values() if getattr(e, "IS_SOLID", False)]

    @property
    def has_solids(self) -> bool:
        return any(getattr(e, "IS_SOLID", False) for e in self._elements.values())

    def covered_solids(self) -> set:
        """Ids of solids whose top face is the bottom face of another solid.

        A mesh-wide pressure acts on shells and on the exposed top faces of solids only.
        """
        bottoms = {frozenset(e.node_ids[i] for i in e.BOTTOM_FACE) for e in self.solid_elements}
        return {e.element_id for e in self.solid_elements if frozenset(e.node_ids[i] for i in e.TOP_FACE) in bottoms}

    def element_free_nodes(self) -> List[int]:
        """Nodes no element uses (reference nodes of connections)."""
        used = {n for e in self._elements.values() for n in e.node_ids}
        return [n for n in self._nodes if n not in used]

    def solid_only_nodes(self) -> List[int]:
        """Nodes used by solids and by no shell: their rotations have no stiffness."""
        solid, shell = set(), set()
        for e in self._elements.values():
            (solid if getattr(e, "IS_SOLID", False) else shell).update(e.node_ids)
        return sorted(solid - shell)

    def insert_element(self, element: ShellElement) -> ShellElement:
        """Add an element built elsewhere (its nodes must already exist)."""
        if element.element_id in self._elements:
            raise ValueError(f"Element {element.element_id} already exists")
        missing = [n for n in element.node_ids if n not in self._nodes]
        if missing:
            raise KeyError(f"Element {element.element_id} references unknown nodes {missing}")
        self._elements[element.element_id] = element
        self._layout = None
        return element

    def replace_element(self, element: ShellElement) -> None:
        if element.element_id not in self._elements:
            raise KeyError(f"Element {element.element_id} is not in the mesh")
        self._elements[element.element_id] = element
        self._layout = None

    def set_element_laminate(self, element_id: int, laminate: Laminate) -> None:
        self.replace_element(self._elements[element_id].with_laminate(laminate))

    def set_kinematics(self, kinematics: KinematicModel) -> None:
        for element in list(self._elements.values()):
            self._elements[element.element_id] = element.with_kinematics(kinematics)
        self._layout = None

    def set_material_axis(self, material_axis, element_ids: Optional[Iterable[int]] = None) -> None:
        """Set the 0-degree reference direction (vector or orientation object) of elements."""
        ids = self.element_ids if element_ids is None else [int(e) for e in element_ids]
        for eid in ids:
            self._elements[eid] = self._elements[eid].with_material_axis(material_axis)

    def element(self, element_id: int) -> ShellElement:
        return self._elements[element_id]

    @property
    def elements(self) -> List[ShellElement]:
        return list(self._elements.values())

    @property
    def element_ids(self) -> List[int]:
        return list(self._elements)

    @property
    def n_elements(self) -> int:
        return len(self._elements)

    def element_dofs(self, element: Union[int, ShellElement]) -> np.ndarray:
        if not hasattr(element, "node_ids"):
            element = self._elements[int(element)]
        offsets, _, _ = self._dof_layout()
        d = element.dofs_per_node
        return np.array([offsets[n] + k for n in element.node_ids for k in range(d)], dtype=int)

    def copy(self) -> "FEAMesh":
        clone = FEAMesh(self.hygrothermal_engine, self.element_registry, self.solid_registry)
        clone._nodes = {k: v.copy() for k, v in self._nodes.items()}
        clone._elements = dict(self._elements)
        return clone

    # -------------------------------------------------------------- factory
    @classmethod
    def rectangular_plate(
        cls,
        length_x: float,
        length_y: float,
        nx: int,
        ny: int,
        laminate: Laminate,
        kinematics: KinematicModel,
        origin: Sequence[float] = (0.0, 0.0, 0.0),
        **element_options,
    ) -> "FEAMesh":
        """Structured nx-by-ny Quad4 mesh in the xy-plane (see ``RectangularPlateBuilder``)."""
        from shell_fem.pre_processing.mesh_library.builders import RectangularPlateBuilder

        return RectangularPlateBuilder(length_x, length_y, nx, ny, origin=origin, element="quad4").build(
            laminate, kinematics, **element_options)
