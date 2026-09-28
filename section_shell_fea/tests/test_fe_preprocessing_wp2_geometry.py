"""WP2 — unified element geometry: assembly / recovery / buckling agree on L_s.

WP0 pinned the disagreement (recovery and buckling recomputed plain inboard
contour length while assembly used a blended outboard length). WP2 persists the
authoritative assembly length on the cache DTOs so all three consumers read the
*same* ``L_s`` when they honour the cache.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import (
    ShellTheory,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.element_library.formulation_factory import (
    _element_L_s,
    build_formulation_cache,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.element_library.kernel import (
    resolve_kernel_by_id,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.buckling.operations.assemble_geometric import (
    assemble_panel_geometric_stiffness,
    assemble_section_geometric_stiffness,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.compute_secondary.secondary_results_orchestrator import (
    _compute_secondary_resultants,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.tests._panel_topology_factory import (
    build_shell_panel_topology,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.shell_mesh_map import (
    panel_maps_from_section_strip_mesh,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.global_shell_assembly import (
    _NDOF_NODE,
    _PanelGlobalMap,
)
from blade_structure_utilities.mesh_utilities.section_mesh import build_section_strip_mesh


def _panel_maps_and_loads(*, n_elements_per_panel: int = 3):
    panel_topology = build_shell_panel_topology("2412", [0.35], chord_m=1.0)
    zones = list(panel_topology.zones)
    nx = [np.linspace(0.0, 1000.0, len(np.asarray(p.s, dtype=float))) for p in zones]
    nxy = [np.zeros_like(n) for n in nx]
    strip = build_section_strip_mesh(
        panel_topology.shell_inputs, n_elements_per_panel=n_elements_per_panel
    )
    raw_maps, _ = panel_maps_from_section_strip_mesh(strip, zones)
    # Contiguous global numbering (connected-assembly style) so scatter indices
    # are valid non-negative DOFs.
    panel_maps = []
    next_g = 0
    for pm in raw_maps:
        if len(pm.s_nodes) < 2:
            panel_maps.append(pm)
            continue
        gn = list(pm.global_nodes)
        for i in range(len(gn)):
            gn[i] = next_g
            next_g += 1
        panel_maps.append(
            _PanelGlobalMap(pm.s_nodes, pm.elements, gn, pm.panel_label, pm.panel_index)
        )
    return panel_maps, zones, nx, nxy


def _scaled_outboard_zones(zones: list, scale: float = 1.25) -> list:
    out: list = []
    for z in zones:
        s = np.asarray(z.s, dtype=float)
        s0 = float(s[0])
        out.append(
            SimpleNamespace(
                s=(s0 + (s - s0) * float(scale)).copy(),
                nodes=np.asarray(getattr(z, "nodes"), dtype=float).copy(),
                lam=z.lam,
            )
        )
    return out


def _cache_with_scaled_outboard(scale: float = 1.3):
    panel_maps, zones, nx, nxy = _panel_maps_and_loads()
    zones_out = _scaled_outboard_zones(zones, scale=scale)
    cache = build_formulation_cache(
        panel_maps,
        zones,
        nx,
        nxy,
        shell_theory=ShellTheory.FSDT,
        L_x=1.0,
        zones_outboard=zones_out,
    )
    return cache, panel_maps, zones, zones_out


def test_cache_persists_blended_assembly_L_s():
    """Persisted ``L_s`` equals the factory blended length and differs from plain."""
    cache, panel_maps, zones, zones_out = _cache_with_scaled_outboard()
    assert cache.stiffness_objects
    blended_seen = 0
    for stiff, force in zip(cache.stiffness_objects, cache.force_objects):
        pi = int(stiff.panel_index)
        lei = int(stiff.local_element_index)
        pm = panel_maps[pi]
        i0, i1, _i2, _i3 = pm.elements[lei]
        L_s_blended, _ = _element_L_s(pi, pm.s_nodes, i0, i1, zones, zones_out, 1.0)
        L_s_plain = float(abs(pm.s_nodes[i1] - pm.s_nodes[i0]))
        assert np.isfinite(stiff.L_s)
        assert stiff.L_s == L_s_blended
        assert force.L_s == L_s_blended
        assert stiff.formulation_id == cache.formulation_id
        if abs(L_s_blended - L_s_plain) > 1e-12:
            blended_seen += 1
    assert blended_seen > 0, "scaled outboard must produce blended != plain L_s"


def test_recovery_uses_cached_blended_L_s():
    """Secondary recovery honours ``stiff.L_s`` (blended), not recomputed plain."""
    cache, panel_maps, zones, zones_out = _cache_with_scaled_outboard()
    kernel = resolve_kernel_by_id(cache.formulation_id)

    rng = np.random.default_rng(1234)
    U_e = [
        rng.standard_normal(int(s.K_e.shape[0])) * 1e-4 for s in cache.stiffness_objects
    ]

    result = _compute_secondary_resultants(
        cache, U_e, L_x=1.0, panel_maps=panel_maps, zones=zones
    )
    rows = result.elemental_results.resultant_rows

    # Independently recover using the persisted blended L_s; must match bitwise.
    ri = 0
    checked_blended = 0
    for stiff, u_elem in zip(cache.stiffness_objects, U_e):
        if not np.isfinite(stiff.L_s) or stiff.L_s < 1e-30:
            continue
        pi = int(stiff.panel_index)
        spec = cache.panel_constitutive_specs[pi]
        assert spec is not None
        expected = kernel.recover_generalized(u_elem, stiff.L_s, 1.0, spec.fsdt)
        row = rows[ri]
        assert row.N11 == expected["N11"]
        assert row.M12 == expected["M12"]
        # Recovering with the plain length would differ for blended elements.
        pm = panel_maps[pi]
        i0, i1, _i2, _i3 = pm.elements[int(stiff.local_element_index)]
        L_s_plain = float(abs(pm.s_nodes[i1] - pm.s_nodes[i0]))
        if abs(stiff.L_s - L_s_plain) > 1e-9:
            plain = kernel.recover_generalized(u_elem, L_s_plain, 1.0, spec.fsdt)
            assert row.N11 != plain["N11"]
            checked_blended += 1
        ri += 1
    assert checked_blended > 0


def test_buckling_uses_cached_blended_L_s():
    """Geometric-stiffness assembly scatters element ``Kg`` at ``stiff.L_s``."""
    cache, panel_maps, zones, zones_out = _cache_with_scaled_outboard()

    # Choose one element whose blended L_s differs from plain contour length.
    target = None
    for stiff in cache.stiffness_objects:
        pi = int(stiff.panel_index)
        lei = int(stiff.local_element_index)
        pm = panel_maps[pi]
        i0, i1, _i2, _i3 = pm.elements[lei]
        L_s_plain = float(abs(pm.s_nodes[i1] - pm.s_nodes[i0]))
        if abs(stiff.L_s - L_s_plain) > 1e-9:
            target = (stiff, pi, lei, L_s_plain)
            break
    assert target is not None
    stiff, pi, lei, L_s_plain = target

    n_gdof = int(max(int(s.global_dof_indices.max()) for s in cache.stiffness_objects) + 1)
    prestress = {(pi, lei): (-5.0e4, 1.0e3, -2.0e3)}
    Kg = assemble_section_geometric_stiffness(
        cache, panel_maps, prestress, n_gdof, L_x=1.0, shell_theory=ShellTheory.FSDT
    )

    g = stiff.global_dof_indices
    Kg_blended = assemble_panel_geometric_stiffness(
        L_s=stiff.L_s, L_x=1.0, N11=-5.0e4, N22=1.0e3, N12=-2.0e3, n_gauss=2
    )
    Kg_plain = assemble_panel_geometric_stiffness(
        L_s=L_s_plain, L_x=1.0, N11=-5.0e4, N22=1.0e3, N12=-2.0e3, n_gauss=2
    )
    sub = np.asarray(Kg[np.ix_(g, g)].todense())
    np.testing.assert_allclose(sub, Kg_blended, rtol=0.0, atol=0.0)
    assert not np.allclose(sub, Kg_plain)
