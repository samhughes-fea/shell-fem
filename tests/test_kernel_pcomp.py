"""PCOMP assignment on the kernel shell mesh, including the cap offset."""

import numpy as np
import pytest

from implicit_geometry_kernel.mesh.shell_mesh import ShellMesh
from shell_fem import (
    AnalysisModel,
    BladeBuilder,
    BoundaryConditions,
    DOF,
    FSDTModel,
    Laminate,
    LoadCase,
    NastranDeckWriter,
    NastranFieldFormatter,
    PcompAssignment,
    solve_kernel_pcomp,
)
from laminate_constitutive_model.laminate_mechanics import Ply


def _quad_mesh(n: int, length: float, width: float, eccentricity: float = 0.0) -> ShellMesh:
    """A flat strip of ``n`` quads in the x-z plane, span along z."""
    xs = (0.0, width)
    zs = np.linspace(0.0, length, n + 1)
    nodes = np.array([[x, 0.0, z] for z in zs for x in xs])
    elements = []
    for i in range(n):
        a = 2 * i
        elements.append((a, a + 1, a + 3, a + 2))
    blank = ("panel",) * n
    return ShellMesh(
        nodes, tuple(elements), blank, ("skin",) * n, blank,
        np.full(n, 0.01), np.full(n, eccentricity),
        tuple(("skin",) for _ in range(n)), tuple((0.01,) for _ in range(n)),
        stations=(0.0, length))


def _assignment(material, thickness: float, offset: float) -> PcompAssignment:
    return PcompAssignment((Ply(material, thickness, 0.0),), offset, "panel")


def test_offset_fraction_is_the_cap_eccentricity_over_thickness(aluminium):
    """offset = eccentricity / thickness puts the mid-plane on -eccentricity, and Z0 follows."""
    ecc, thickness = 0.004, 0.011
    mesh = _quad_mesh(1, 1.0, 0.1, eccentricity=ecc)
    # the kernel still reports the eccentricity; the assignment is what is used
    assigned = _assignment(aluminium, thickness, ecc / thickness)
    blade = BladeBuilder.from_shell_mesh(mesh).build_pcomp([assigned], FSDTModel())
    lam = blade.mesh.element(blade.elements("caps")[0]).laminate
    assert lam.offset == pytest.approx(-ecc)
    assert lam.thickness == pytest.approx(thickness)
    z0 = -(0.5 + assigned.offset) * thickness
    assert lam.offset - 0.5 * lam.thickness == pytest.approx(z0)

    ignored = _assignment(aluminium, thickness, 0.0)
    plain = BladeBuilder.from_shell_mesh(mesh).build_pcomp([ignored], FSDTModel())
    assert plain.mesh.element(1).laminate.offset == 0.0
    assert "caps" not in plain.element_sets


def test_nastran_deck_writes_that_z0(aluminium, tmp_path):
    ecc, thickness = 0.004, 0.011
    mesh = _quad_mesh(1, 1.0, 0.1)
    assigned = _assignment(aluminium, thickness, ecc / thickness)
    blade = BladeBuilder.from_shell_mesh(mesh).build_pcomp([assigned], FSDTModel())
    model = AnalysisModel(blade.mesh, BoundaryConditions().clamp(blade.nodes("root")), LoadCase())
    deck = NastranDeckWriter().write(model, tmp_path, "cap")
    z0 = -(0.5 + assigned.offset) * thickness
    text = deck.path.read_text()
    assert "PCOMP" in text
    assert NastranFieldFormatter.field(z0) in text


def test_pcomp_matches_the_lookup_offset(aluminium):
    """The fraction recovered from a lookup stack rebuilds the same mid-plane."""
    from implicit_geometry_kernel import Airfoil, BladeGeometry, Station

    airfoil = Airfoil.naca4("0024")
    stations = [Station(0.0, 1.0, airfoil, pitch_axis=0.3), Station(2.0, 1.0, airfoil, pitch_axis=0.3)]
    geo = BladeGeometry(stations, "2D-F-LC-0R", 0.003, pivot_fraction=0.3, spar_cap_thickness=0.008,
                        shear_web_thickness=0.002, web_positions=(0.18, 0.48))
    import implicit_geometry_kernel.mesh as kernel_mesh

    sm = kernel_mesh.shell_mesh(geo, h=0.5)
    skin = Laminate.from_angles(aluminium, [0, 90], 1.5e-3)
    cap = Laminate.from_angles(aluminium, [0] * 8, 1.0e-3)
    web = Laminate.from_angles(aluminium, [45, -45], 1.0e-3)
    laminates = {"skin": skin, "cap": cap, "web": web}
    looked = BladeBuilder.from_shell_mesh(sm, geo).build(laminates, FSDTModel())
    by_id = {e.element_id: e for e in looked.mesh.shell_elements}
    assignments = []
    for i, eids in enumerate(looked.source_elements):
        lam = by_id[eids[0]].laminate
        fraction = 0.0 if lam.thickness == 0.0 else -lam.offset / lam.thickness
        assignments.append(PcompAssignment(tuple(lam.plies), fraction, lam.name))
    again = BladeBuilder.from_shell_mesh(sm, geo).build_pcomp(assignments, FSDTModel())
    for eid in again.mesh.element_ids:
        a, b = again.mesh.element(eid).laminate, looked.mesh.element(eid).laminate
        assert a.offset == pytest.approx(b.offset)
        assert a.thickness == pytest.approx(b.thickness)
    assert set(again.elements("caps")) == set(looked.elements("caps"))


def test_cantilever_strip_matches_beam_theory(aluminium):
    length, width, thickness, n = 1.0, 0.1, 1.0e-3, 16
    mesh = _quad_mesh(n, length, width)
    assigned = [_assignment(aluminium, thickness, 0.0)] * n
    bcs = BoundaryConditions()
    load = LoadCase("tip")
    # clamp and load are filled after the node sets exist; build via the solver with them
    # constructed from the same mesh stations: root z = 0, tip z = length, two nodes each
    root = [1, 2]
    tip = [2 * n + 1, 2 * n + 2]
    bcs.clamp(root)
    force = 1.0
    for node in tip:
        load.add_nodal_load(node, DOF.UY, 0.5 * force)
    blade, (result,) = solve_kernel_pcomp(mesh, assigned, bcs, load)
    assert result.ok
    e = aluminium.elastic.E1
    ei = e * width * thickness**3 / 12.0
    expected = force * length**3 / (3.0 * ei)
    tip_y = np.mean([result.displacements[node][DOF.UY] for node in blade.nodes("tip")])
    assert tip_y == pytest.approx(expected, rel=0.02)
    assert max(abs(result.displacements[node][DOF.UY]) for node in blade.nodes("root")) == 0.0
