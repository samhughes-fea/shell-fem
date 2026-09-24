"""Immutable shell formulation layouts and descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

from blade_structure_utilities.geometry_utilities.coordinate_frames import (
    FRAME_C,
    FRAME_E,
    FRAME_M,
    FrameTag,
    require_frame_tag,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.analysis_dof_layout import (
    AnalysisDofLayout,
)

ComponentCapability = Literal[
    "modelled",
    "constitutive_recovered",
    "postprocessed_reconstruction",
    "theory_constrained_zero",
    "not_modelled_zero",
]

CANONICAL_RESULTANT_NAMES: tuple[str, ...] = (
    "N11",
    "N22",
    "N12",
    "M11",
    "M22",
    "M12",
    "Q1",
    "Q2",
)
CANONICAL_STRESS_NAMES: tuple[str, ...] = (
    "sigma11",
    "sigma22",
    "sigma33",
    "tau23",
    "tau13",
    "tau12",
)
MITC4_DOF_NAMES: tuple[str, ...] = (
    "u_x",
    "u_s",
    "w",
    "beta_s",
    "beta_x",
    "beta_z",
)
MITC4_GENERALIZED_STRAIN_NAMES: tuple[str, ...] = (
    "eps11",
    "eps22",
    "gamma12",
    "kappa11",
    "kappa22",
    "kappa12",
    "gamma13",
    "gamma23",
)


def _require_unique(names: Sequence[str], *, kind: str) -> tuple[str, ...]:
    out = tuple(str(n) for n in names)
    if len(set(out)) != len(out) or not out:
        raise ValueError(f"{kind} names must be unique and non-empty")
    return out


@dataclass(frozen=True, slots=True)
class ShellDofLayout:
    """Named nodal DOF layout in a structural frame."""

    layout_id: str
    names: tuple[str, ...]
    frame: FrameTag
    n_nodes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "names", _require_unique(self.names, kind="DOF"))
        object.__setattr__(self, "frame", require_frame_tag(self.frame))
        if int(self.n_nodes) < 1:
            raise ValueError("n_nodes must be >= 1")

    @property
    def n_dof_node(self) -> int:
        return len(self.names)

    @property
    def n_elem_dof(self) -> int:
        return self.n_nodes * self.n_dof_node


@dataclass(frozen=True, slots=True)
class ShellTopologyLayout:
    """Element topology (node count and connectivity convention)."""

    topology_id: str
    n_nodes: int
    node_order: tuple[int, ...]

    def __post_init__(self) -> None:
        order = tuple(int(i) for i in self.node_order)
        if len(order) != int(self.n_nodes) or sorted(order) != list(range(self.n_nodes)):
            raise ValueError("node_order must be a permutation of 0..n_nodes-1")
        object.__setattr__(self, "node_order", order)


@dataclass(frozen=True, slots=True)
class GeneralizedComponentLayout:
    """Work-conjugate generalized strain / resultant layout."""

    layout_id: str
    strain_names: tuple[str, ...]
    resultant_names: tuple[str, ...]
    frame: FrameTag
    capabilities: Mapping[str, ComponentCapability]

    def __post_init__(self) -> None:
        strains = _require_unique(self.strain_names, kind="strain")
        resultants = _require_unique(self.resultant_names, kind="resultant")
        if len(strains) != len(resultants):
            raise ValueError("strain and resultant layouts must be work-conjugate")
        object.__setattr__(self, "strain_names", strains)
        object.__setattr__(self, "resultant_names", resultants)
        object.__setattr__(self, "frame", require_frame_tag(self.frame))
        caps = {str(k): v for k, v in dict(self.capabilities).items()}
        missing = [name for name in resultants if name not in caps]
        if missing:
            raise ValueError(f"missing capability declarations: {missing}")

    @property
    def n_components(self) -> int:
        return len(self.resultant_names)


@dataclass(frozen=True, slots=True)
class ShellFormulationDescriptor:
    """Immutable discrete-formulation contract."""

    formulation_id: str
    continuum_theory: str
    topology: ShellTopologyLayout
    dof_layout: ShellDofLayout
    generalized: GeneralizedComponentLayout
    stress_names: tuple[str, ...]
    stress_frame: FrameTag
    stress_capabilities: Mapping[str, ComponentCapability]
    analysis_dof_layout: AnalysisDofLayout

    def __post_init__(self) -> None:
        if self.topology.n_nodes != self.dof_layout.n_nodes:
            raise ValueError("topology and DOF layout node counts must match")
        if len(self.analysis_dof_layout.formulation_dof_to_analysis) != self.dof_layout.n_dof_node:
            raise ValueError("analysis embedding must map every formulation nodal DOF")
        object.__setattr__(
            self, "stress_names", _require_unique(self.stress_names, kind="stress")
        )
        object.__setattr__(self, "stress_frame", require_frame_tag(self.stress_frame))
        caps = {str(k): v for k, v in dict(self.stress_capabilities).items()}
        missing = [name for name in self.stress_names if name not in caps]
        if missing:
            raise ValueError(f"missing stress capability declarations: {missing}")
        object.__setattr__(self, "stress_capabilities", caps)


def mitc4_fsdt_descriptor() -> ShellFormulationDescriptor:
    """Production MITC4-FSDT descriptor (Q4, six DOFs, eight generalized components)."""
    topology = ShellTopologyLayout(
        topology_id="Q4",
        n_nodes=4,
        node_order=(0, 1, 2, 3),
    )
    dof = ShellDofLayout(
        layout_id="mitc4_c6",
        names=MITC4_DOF_NAMES,
        frame=FRAME_C,
        n_nodes=4,
    )
    generalized = GeneralizedComponentLayout(
        layout_id="fsdt_nmq8",
        strain_names=MITC4_GENERALIZED_STRAIN_NAMES,
        resultant_names=CANONICAL_RESULTANT_NAMES,
        frame=FRAME_E,
        capabilities={
            name: "modelled" for name in CANONICAL_RESULTANT_NAMES
        },
    )
    analysis = AnalysisDofLayout.station_c6()
    return ShellFormulationDescriptor(
        formulation_id="MITC4-FSDT",
        continuum_theory="FSDT",
        topology=topology,
        dof_layout=dof,
        generalized=generalized,
        stress_names=CANONICAL_STRESS_NAMES,
        stress_frame=FRAME_M,
        stress_capabilities={
            "sigma11": "constitutive_recovered",
            "sigma22": "constitutive_recovered",
            "sigma33": "not_modelled_zero",
            "tau23": "postprocessed_reconstruction",
            "tau13": "postprocessed_reconstruction",
            "tau12": "constitutive_recovered",
        },
        analysis_dof_layout=analysis,
    )


__all__ = [
    "AnalysisDofLayout",
    "CANONICAL_RESULTANT_NAMES",
    "CANONICAL_STRESS_NAMES",
    "ComponentCapability",
    "GeneralizedComponentLayout",
    "MITC4_DOF_NAMES",
    "MITC4_GENERALIZED_STRAIN_NAMES",
    "ShellDofLayout",
    "ShellFormulationDescriptor",
    "ShellTopologyLayout",
    "mitc4_fsdt_descriptor",
]
