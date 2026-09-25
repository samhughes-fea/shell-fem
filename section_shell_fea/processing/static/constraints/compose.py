"""Sole deterministic composer of a station constraint set.

Order is fixed:

1. junction equations from topology/metadata;
2. essential supports;
3. cap-tie equations, expanding any junction-slave masters;
4. validate/coalesce;
5. freeze sorted equations, fixed DOFs, and report.

Assembly never builds these equations independently.
"""

from __future__ import annotations

from typing import Any

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.global_numbering import (
    DEFAULT_BC_MODE,
    attach_cluster_node_meta,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.constraints.cap_tie.geometry import (
    CapTieReport,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.constraints.cap_tie.mpcs import (
    build_cap_tie_constraints,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.constraints.junction.mpcs import (
    build_cluster_basis_constraints,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.constraints.junction.supports import (
    build_clamp_supports,
    build_cluster_basis_supports,
    build_hybrid_collinear_supports,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.constraints.types import (
    Constraints,
    SectionConstraintReport,
    SectionConstraintSet,
    constraint_equations_to_solver_dict,
    solver_dict_to_constraint_equations,
)

_TB_LIKE_MODES = frozenset(
    {"transformed_basis", "hybrid_collinear", "hybrid_kind_crossing"}
)


def _mpc_clusters(topology: Any, *, interface_constraint_mode: str) -> list:
    from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.global_numbering import (
        cluster_is_kind_crossing,
    )

    mode = str(interface_constraint_mode or "shared")
    clusters = list(getattr(topology, "clusters", None) or [])
    if mode not in {"hybrid_collinear", "hybrid_kind_crossing"}:
        return clusters
    flags = getattr(topology, "cluster_is_collinear_flags", None) or {}
    panel_maps = list(getattr(topology, "panel_maps", None) or [])
    mpc_clusters = []
    for cid, cluster in enumerate(clusters):
        if bool(flags.get(cid, False)):
            continue
        if mode == "hybrid_kind_crossing" and not cluster_is_kind_crossing(cluster, panel_maps):
            continue
        mpc_clusters.append(cluster)
    return mpc_clusters


def _junction_equations(assembly: Any, *, interface_constraint_mode: str) -> Constraints:
    topology = assembly.topology
    if topology is None:
        return {}
    mode = str(interface_constraint_mode or "shared")
    if mode not in _TB_LIKE_MODES:
        return {}
    ndof = int(assembly.cache.n_dof_node)
    maps = {
        "endpoint_cluster_node": getattr(topology, "endpoint_cluster_node", None) or {},
        "endpoint_cluster_rot_node": getattr(topology, "endpoint_cluster_rot_node", None) or {},
        "cluster_layer_node": getattr(topology, "cluster_layer_node", None) or {},
        "cluster_layer_rot_node": getattr(topology, "cluster_layer_rot_node", None) or {},
    }
    if not maps["endpoint_cluster_node"]:
        return {}
    zones = list(getattr(assembly, "zones", None) or [])
    return build_cluster_basis_constraints(
        list(assembly.panel_maps),
        zones,
        _mpc_clusters(topology, interface_constraint_mode=mode),
        maps["endpoint_cluster_node"],
        maps["endpoint_cluster_rot_node"],
        maps["cluster_layer_node"],
        maps["cluster_layer_rot_node"],
        ndof=ndof,
    )


def _support_dofs(
    assembly: Any,
    *,
    interface_constraint_mode: str,
    bc_mode: str,
) -> set[int]:
    ndof = int(assembly.cache.n_dof_node)
    topology = assembly.topology
    mode = str(interface_constraint_mode or "shared")
    if mode == "transformed_basis" and topology is not None:
        return build_cluster_basis_supports(
            list(assembly.panel_maps),
            dict(topology.endpoint_cluster_index),
            dict(getattr(topology, "cluster_layer_node", None) or {}),
            dict(getattr(topology, "cluster_layer_rot_node", None) or {}),
            ndof=ndof,
            bc_mode=bc_mode,
        )
    if mode in {"hybrid_collinear", "hybrid_kind_crossing"} and topology is not None:
        return build_hybrid_collinear_supports(
            list(assembly.panel_maps),
            dict(topology.endpoint_cluster_index),
            dict(getattr(topology, "cluster_layer_node", None) or {}),
            dict(getattr(topology, "cluster_layer_rot_node", None) or {}),
            ndof=ndof,
            bc_mode=bc_mode,
        )
    return build_clamp_supports(
        list(assembly.panel_maps),
        ndof=ndof,
        bc_mode=bc_mode,
    )


def build_section_constraint_set(
    assembly: Any,
    *,
    interface_constraint_mode: str = "shared",
    cap_tie_mode: str = "none",
    bc_mode: str = DEFAULT_BC_MODE,
    shell_midlines: Any | None = None,
) -> SectionConstraintSet:
    """Compose one immutable constraint set from an assembled section.

    ``shell_midlines`` carries SG-017 bonds for ``cap_tie_mode='skin_offset'``
    (FEA-CT-001). When omitted, midlines are taken from
    ``assembly.shell_midlines`` if present.
    """
    iface = str(interface_constraint_mode or "shared")
    ties = str(cap_tie_mode or "none")
    supports_mode = str(bc_mode or DEFAULT_BC_MODE)
    ndof = int(assembly.cache.n_dof_node)
    midlines = shell_midlines
    if midlines is None:
        midlines = getattr(assembly, "shell_midlines", None)

    junction = _junction_equations(assembly, interface_constraint_mode=iface)
    topology = assembly.topology
    if topology is not None and getattr(topology, "cluster_layer_node", None):
        attach_cluster_node_meta(
            assembly.node_meta,
            list(topology.clusters),
            list(getattr(assembly, "zones", None) or []),
            dict(topology.cluster_layer_node or {}),
            dict(topology.cluster_layer_rot_node or {}),
            junction,
        )

    fixed = _support_dofs(
        assembly,
        interface_constraint_mode=iface,
        bc_mode=supports_mode,
    )
    cap_dict, cap_report = build_cap_tie_constraints(
        list(assembly.panel_maps),
        assembly.node_meta,
        ndof=ndof,
        mode=ties,
        existing=junction,
        shell_midlines=midlines,
    )
    if not isinstance(cap_report, CapTieReport):
        cap_report = CapTieReport(mode=ties)

    overlap = set(junction) & set(cap_dict)
    if overlap:
        raise ValueError(
            "constraint composer: cap-tie slaves overlap junction slaves: "
            f"{sorted(overlap)[:8]}"
        )
    merged: Constraints = {**junction, **cap_dict}
    # Supports take precedence (same as constrain/reduce): drop MPC rows whose
    # slave is already an essential BC so the frozen product stays disjoint.
    dropped_fixed = sorted(int(s) for s in set(merged) & set(fixed))
    for slave in dropped_fixed:
        merged.pop(slave, None)
    notes = [str(n) for n in cap_report.notes]
    if dropped_fixed:
        notes.append(
            f"dropped {len(dropped_fixed)} MPC slave(s) already in fixed_dofs "
            "(supports take precedence)"
        )
    equations = solver_dict_to_constraint_equations(merged)
    # Round-trip through the adapter so freeze/sort is the production form.
    merged = constraint_equations_to_solver_dict(equations)

    # Deprecated compatibility view: node_meta still carries the solver dict.
    assembly.node_meta.constraints = dict(merged)

    remaining_junction = set(junction) - set(dropped_fixed)
    remaining_cap = set(cap_dict) - set(dropped_fixed)
    tie_slaves = frozenset(int(lk.slave_node) for lk in cap_report.links)
    report = SectionConstraintReport(
        interface_constraint_mode=iface,
        cap_tie_mode=ties,
        bc_mode=supports_mode,
        n_junction_equations=len(remaining_junction),
        n_cap_tie_equations=len(remaining_cap),
        n_fixed_dofs=len(fixed),
        n_cap_links=int(cap_report.n_links),
        tie_slave_nodes=tie_slaves,
        junction_slave_dofs=frozenset(int(s) for s in remaining_junction),
        notes=tuple(notes),
        cap_tie=cap_report,
    )
    return SectionConstraintSet(
        equations=equations,
        fixed_dofs=frozenset(int(d) for d in fixed),
        report=report,
    )


__all__ = ["build_section_constraint_set"]
