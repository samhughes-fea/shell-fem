"""BladeBuilder.from_shell_mesh: shell models on the geometry kernel's own shell mesh."""

import numpy as np
import pytest

from shell_fem import BladeBuilder, FSDTModel, Laminate, beam_section_properties, create_default_library
from implicit_geometry_kernel import Airfoil, BladeGeometry, Station

kernel_mesh = pytest.importorskip("implicit_geometry_kernel.mesh")

AIRFOIL = Airfoil.naca4("0024")


def _geometry(key="2D-F-LC-2R", length=4.0):
    stations = [Station(0.0, 1.0, AIRFOIL, pitch_axis=0.3), Station(length, 1.0, AIRFOIL, pitch_axis=0.3)]
    return BladeGeometry(stations, key, 0.003, pivot_fraction=0.3, spar_cap_thickness=0.008,
                         shear_web_thickness=0.002, web_positions=(0.18, 0.48))


@pytest.fixture(scope="module")
def laminates():
    lib = create_default_library()
    glass, carbon = lib.get("E-glass/epoxy"), lib.get("T300/5208")
    return {
        "skin": Laminate.from_angles(glass, [0, 45, -45, -45, 45, 0], 0.5e-3),
        "web": Laminate.from_angles(glass, [45, -45, -45, 45], 0.5e-3),
        "cap": Laminate.from_angles(carbon, [0] * 16, 0.5e-3),
    }


@pytest.fixture(scope="module")
def ribbed(laminates):
    geo = _geometry()
    sm = kernel_mesh.shell_mesh(geo, h=0.1)
    rib_t = {t for s, ts in zip(sm.surface, sm.layer_thicknesses) if s == "rib" for t in ts}
    assert len(rib_t) == 1
    lib = create_default_library()
    rib = Laminate.from_angles(lib.get("E-glass/epoxy"), [0, 90] * 12, rib_t.pop() / 24)
    blade = BladeBuilder.from_shell_mesh(sm, geo).build(dict(laminates, rib=rib), FSDTModel())
    return sm, blade


def _normal(mesh, element):
    p = np.array([mesh.node_coordinates(n) for n in element.node_ids])
    a = 0.5 * np.sum(np.cross(p, np.roll(p, -1, axis=0)), axis=0)
    return a / np.linalg.norm(a), p.mean(axis=0)


def test_kernel_nodes_quads_and_polygon_fans(ribbed):
    sm, blade = ribbed
    mesh = blade.mesh
    n_poly = len(sm.polygons)
    assert len(mesh.node_ids) == len(sm.nodes) + n_poly  # one centre node per polygon
    for k in (0, len(sm.nodes) // 2, len(sm.nodes) - 1):
        assert np.allclose(mesh.node_coordinates(k + 1), sm.nodes[k])
    kinds = [len(e.node_ids) for e in mesh.shell_elements]
    assert kinds.count(4) == len(sm.quads)
    assert kinds.count(3) == sum(len(sm.elements[i]) for i in sm.polygons)
    assert blade.surface == "mid" and blade.shell_mesh is sm
    assert np.allclose(blade.span_positions, sm.stations)


def test_sets_and_normals(ribbed):
    _sm, blade = ribbed
    for name in ("skin", "caps", "webs", "ribs", "tip_closure", "all", "spar_cap_suction_surface", "shear_web_1",
                 "skin_suction_surface", "skin_pressure_surface:leading_edge", "cap_su", "web_0"):
        assert blade.elements(name), name
    assert set(blade.elements("caps")) <= set(blade.elements("skin"))
    mesh = blade.mesh
    by_id = {e.element_id: e for e in mesh.shell_elements}
    flux = 0.0
    for eid in blade.elements("skin"):
        n, x = _normal(mesh, by_id[eid])
        flux += float(n[:2] @ x[:2])
        assert np.allclose(by_id[eid].frame[2], n, atol=1e-6)  # the element's normal is the kernel's
    assert flux > 0.0  # out of the section
    for eid in blade.elements("webs"):
        assert _normal(mesh, by_id[eid])[0][0] > 0.99  # towards the trailing edge
    for name in ("ribs", "tip_closure"):
        for eid in blade.elements(name):
            assert _normal(mesh, by_id[eid])[0][2] > 0.99
    assert np.allclose([mesh.node_coordinates(n)[2] for n in blade.nodes("tip")], 4.0)
    assert np.allclose([mesh.node_coordinates(n)[2] for n in blade.nodes("root")], 0.0)
    assert blade.nodes("shear_web_0")


def test_capped_skin_is_one_offset_stack_and_the_tip_is_skin(ribbed, laminates):
    _sm, blade = ribbed
    by_id = {e.element_id: e for e in blade.mesh.shell_elements}
    caps = set(blade.elements("caps"))
    for eid in caps:
        lam = by_id[eid].laminate
        assert lam.thickness == pytest.approx(0.011)
        assert lam.offset == pytest.approx(-0.004)  # half the 8 mm cap, inside the skin mid-surface
    for eid in set(blade.elements("skin")) - caps:
        assert by_id[eid].laminate.offset == 0.0
    for eid in blade.elements("tip_closure"):
        assert by_id[eid].laminate is laminates["skin"]


def test_errors(laminates):
    geo = _geometry("2D-F-LC-0R")
    sm = kernel_mesh.shell_mesh(geo, h=0.25)
    with pytest.raises(KeyError):
        BladeBuilder.from_shell_mesh(sm).build({"skin": laminates["skin"]}, FSDTModel())
    thick = Laminate(list(laminates["skin"].plies) * 2, "thick")
    with pytest.raises(ValueError):
        BladeBuilder.from_shell_mesh(sm).build(dict(laminates, skin=thick), FSDTModel())


def test_beam_properties_match_the_geometry_builder(laminates):
    """The kernel mesh and from_geometry's own mesh give the same beam (h = c/10 against 24 quad8 bays)."""
    geo = _geometry("2D-F-LC-0R", length=6.0)
    kernel = beam_section_properties(BladeBuilder.from_shell_mesh(kernel_mesh.shell_mesh(geo, h=0.1), geo)
                                     .build(laminates, FSDTModel()))
    own = beam_section_properties(BladeBuilder.from_geometry(geo, n_span=24).build(laminates, FSDTModel()))
    for name in ("EA", "EI_x", "EI_y", "GJ"):
        assert getattr(kernel, name) == pytest.approx(getattr(own, name), rel=0.01), name
