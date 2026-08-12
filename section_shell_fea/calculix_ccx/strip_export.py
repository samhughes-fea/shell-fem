"""Export strip meshes to CalculiX input decks for validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from blade_structure_utilities.mesh_utilities.section_mesh.engine.fixtures.discretise import discretise_strip
from .orientation import quad_coords_from_strip


def export_rectangular_strip_inp(
    path: str | Path,
    *,
    length_s: float = 2.0,
    span_x: float = 1.0,
    n_elements: int = 4,
    thickness: float = 0.01,
    e_modulus: float = 70e9,
    nu: float = 0.33,
    nx_traction: float = 1000.0,
) -> Path:
    """Write a single flat rectangular strip S4R model to a CalculiX ``.inp`` file.

    Fixture-only: uses ``discretise_strip`` for a one-panel rectangle, not the
    full-section dual mesher path (BL-005).
    """

    out = Path(path)
    s_panel = np.array([0.0, length_s])
    s_nodes, elements = discretise_strip(s_panel, n_elements, panel_label="skin:test")
    yz_nodes = np.column_stack([s_nodes, np.zeros_like(s_nodes)])

    lines: list[str] = [
        "*HEADING",
        "FastBlade S4R strip validation model",
        "*NODE",
    ]
    node_id = 1
    local_to_ccx: dict[int, int] = {}
    for li in range(2 * len(s_nodes)):
        si = li if li < len(s_nodes) else li - len(s_nodes)
        x_val = 0.0 if li < len(s_nodes) else span_x
        lines.append(f"{node_id}, {x_val}, {yz_nodes[si, 0]}, {yz_nodes[si, 1]}")
        local_to_ccx[li] = node_id
        node_id += 1

    lines.append("*ELEMENT, TYPE=S4R, ELSET=STRIP")
    for ei, elem in enumerate(elements, start=1):
        nids = [local_to_ccx[i] for i in elem]
        lines.append(f"{ei}, {', '.join(str(n) for n in nids)}")

    lines.extend([
        "*MATERIAL, NAME=MAT1",
        f"*ELASTIC",
        f"{e_modulus}, {nu}",
        "*SHELL SECTION, ELSET=STRIP, MATERIAL=MAT1",
        f"{thickness}",
        "*BOUNDARY",
        "1, 1, 6",
        f"{local_to_ccx[len(s_nodes) - 1]}, 1, 6",
        f"{local_to_ccx[len(s_nodes)]}, 1, 6",
        f"{local_to_ccx[2 * len(s_nodes) - 1]}, 1, 6",
    ])
    top_nodes = [local_to_ccx[li] for li in range(len(s_nodes), 2 * len(s_nodes))]
    force_per_node = nx_traction * length_s / len(top_nodes)

    # *CLOAD must be inside the *STEP for CalculiX.
    lines.extend(["*STEP", "*STATIC", "*CLOAD"])
    for nid in top_nodes:
        lines.append(f"{nid}, 1, {force_per_node}")
    lines.extend([
        "*EL FILE",
        "S, E",
        "*NODE FILE",
        "U",
        "*END STEP",
    ])
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def strip_coords_for_panel(
    length_s: float,
    n_elements: int,
    *,
    span_x: float = 1.0,
) -> tuple[np.ndarray, list[list[int]], np.ndarray]:
    """Return ``s_nodes``, ``elements``, and ``yz_nodes`` for a flat strip."""

    s_panel = np.array([0.0, length_s])
    s_nodes, elements = discretise_strip(s_panel, n_elements, panel_label="skin:test")
    yz_nodes = np.column_stack([s_nodes, np.zeros_like(s_nodes)])
    return s_nodes, elements, yz_nodes


__all__ = ["export_rectangular_strip_inp", "strip_coords_for_panel"]
