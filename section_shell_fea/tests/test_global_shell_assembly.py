"""Direct regression on the single global shell assembly (``engine/global_shell_assembly.py``).

This 1,000+-line assembly spine had no direct test; here we lock its structural
contract (K_global shape/symmetry, load-vector shape, per-panel maps) and the MPC
interface-constraint reduction on a real multi-cell section.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.api.shell_panel_topology import build_shell_panel_topology
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.fem_config import MITC4_DEFAULTS
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.global_shell_assembly import (
    assemble_global_shell,
)
from blade_structure_utilities.mesh_utilities.section_mesh import build_section_strip_mesh

_NDOF = MITC4_DEFAULTS.n_dof_node


def _panels_and_mesh(*, n_elements_per_panel: int = 4):
    panel_topology = build_shell_panel_topology("2412", [0.35], chord_m=1.0)
    zones = list(panel_topology.zones)
    zeros = [np.zeros(len(np.asarray(p.s, dtype=float))) for p in zones]
    mesh = build_section_strip_mesh(
        panel_topology.shell_inputs, n_elements_per_panel=n_elements_per_panel
    )
    return zones, zeros, mesh


def test_assemble_global_shell_structural_contract() -> None:
    zones, zeros, mesh = _panels_and_mesh(n_elements_per_panel=4)
    K, f, node_meta, panel_maps, abd_p, t_p, g_p, lab_p = assemble_global_shell(
        zones, zeros, zeros, section_strip_mesh=mesh,
    )

    ndof = node_meta.n_nodes * _NDOF
    assert sp.issparse(K)
    assert K.shape == (ndof, ndof)
    assert np.asarray(f).shape == (ndof,)
    assert len(panel_maps) == len(zones)
    assert abd_p.ndim == 3 and abd_p.shape[1:] == (6, 6)
    assert len(lab_p) == abd_p.shape[0]
    assert np.all(np.isfinite(K.data))
    assert np.all(np.isfinite(np.asarray(f)))


def test_assemble_global_shell_is_symmetric() -> None:
    zones, zeros, mesh = _panels_and_mesh(n_elements_per_panel=4)
    K, _f, _nm, _pm, *_abd = assemble_global_shell(
        zones, zeros, zeros, section_strip_mesh=mesh
    )

    K = K.tocsr()
    asym = (K - K.T)
    scale = max(float(abs(K).max()), 1.0)
    assert float(abs(asym).max()) / scale < 1e-9


def test_assemble_global_shell_element_count_scales_dofs() -> None:
    """More elements per panel ⇒ strictly more global DOFs (mesh refinement wiring)."""
    zones, zeros, mesh4 = _panels_and_mesh(n_elements_per_panel=4)
    _panels2, zeros2, mesh8 = _panels_and_mesh(n_elements_per_panel=8)
    _K4, _f4, nm4, _, *_abd4 = assemble_global_shell(
        zones, zeros, zeros, section_strip_mesh=mesh4
    )
    _K8, _f8, nm8, _, *_abd8 = assemble_global_shell(
        zones, zeros2, zeros2, section_strip_mesh=mesh8
    )

    assert nm8.n_nodes > nm4.n_nodes


def test_assemble_global_shell_interface_modes_change_junction_topology() -> None:
    """``shared`` merges junction nodes; ``transformed_basis`` keeps them separate + constrained."""
    zones, zeros, mesh = _panels_and_mesh(n_elements_per_panel=4)
    _Ks, _fs, nm_shared, _, *_abd = assemble_global_shell(
        zones, zeros, zeros, section_strip_mesh=mesh, interface_constraint_mode="shared",
    )
    _Kt, _ft, nm_tb, _, *_abd2 = assemble_global_shell(
        zones, zeros, zeros, section_strip_mesh=mesh,
        interface_constraint_mode="transformed_basis",
    )
    # Shared physically collapses coincident junction nodes into one, so it has
    # strictly fewer global nodes than the constraint-based transformed basis.
    assert nm_shared.n_nodes < nm_tb.n_nodes
    assert nm_tb.constraints is not None and len(nm_tb.constraints) > 0
    assert not nm_shared.constraints


def test_assembly_only_fast_path_is_deterministic() -> None:
    """Assembly-only path is the sole K_global source for periodic homogenisation.

    Locks perf finding #1: ``assemble_global_shell`` (via ``assemble_section_from_zones``)
    is idempotent so cache hits / double assembly agree to numerical noise.
    """
    zones, zeros, mesh = _panels_and_mesh(n_elements_per_panel=4)
    K_a, _f_a, nm_a, _pm_a, abd_a, *_rest_a = assemble_global_shell(
        zones, zeros, zeros, section_strip_mesh=mesh,
    )
    K_b, _f_b, nm_b, _pm_b, abd_b, *_rest_b = assemble_global_shell(
        zones, zeros, zeros, section_strip_mesh=mesh,
    )

    assert nm_a.n_nodes == nm_b.n_nodes
    assert K_a.shape == K_b.shape
    assert abd_a.shape == abd_b.shape
    diff = (K_a.tocsr() - K_b.tocsr())
    scale = max(float(abs(K_a).max()), 1.0)
    assert float(abs(diff).max()) / scale < 1e-12
