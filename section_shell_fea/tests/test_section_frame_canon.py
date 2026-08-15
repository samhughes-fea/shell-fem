"""The coordinate-frames canon, asserted on the FE worker.

Companion to ``mesh_utilities/section_mesh/tests/test_section_plane_canon.py``,
which pins the geometry→mesh handoff. This file pins the FE side of the same
seam, for the same reason that file gives: every orientation test in the package
is a dot product *within* one frame, and so is blind to a consistent
relabelling. These are cross-frame checks.

Covers, per ``geometry_utilities/coordinate_frames/README.md``:

* section arrays reaching the FE worker are canonical **B** — chord-long in
  :math:`Y_B`, thin in :math:`X_B`;
* the B→NACA adapter used for XFOIL untwists and reads the chord off the right
  column;
* Hashin FI is evaluated in **M**, never in **E**;
* the canon transforms are not reimplemented inside the package;
* section figures use the horiz-:math:`Y_B` / vert-:math:`X_B` / LE-on-RHS recipe.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from blade_structure_utilities.geometry_utilities.coordinate_frames import (
    SECTION_PLOT_XLABEL,
    SECTION_PLOT_YLABEL,
    rotate_S_to_B,
    rotate_ply_stress_E_to_M,
    section_normal_2d,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.loading.builders.outline_frames import (
    blade_outline_to_unit_chord,
    blade_to_naca_rotation,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.api.shell_panel_topology import (
    build_shell_panel_topology,
)
from blade_structure_utilities.load_utilities.section_loads import (
    SectionOutlineGeometry,
)

CHORD = 1.0
_PKG = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def panels_B():
    """Panel nodes as the FE solver sees them, for a twisted section."""
    pt = build_shell_panel_topology("2412", [0.35], chord_m=CHORD, twist_rad=0.15)
    return [np.asarray(p.nodes, dtype=float) for p in pt.zones]


class TestPanelArraysAreCanonicalB:
    def test_chord_spans_Y_and_thickness_spans_X(self, panels_B):
        """The give-away a swapped frame would fail: which axis is long.

        In canonical B the section is chord-long in ``Y`` and thickness-thin in
        ``X``. The rotated-NACA frame the FE worker used to receive had these the
        other way round.
        """
        pts = np.vstack(panels_B)
        x_extent = float(np.ptp(pts[:, 0]))
        y_extent = float(np.ptp(pts[:, 1]))
        assert y_extent > 3.0 * x_extent, (
            f"expected chord-long in Y: X extent {x_extent:.4f}, Y extent {y_extent:.4f}"
        )

    def test_pivot_is_inside_the_section(self, panels_B):
        """Origin on the pivot chordline means Y_LE < 0 < Y_TE."""
        pts = np.vstack(panels_B)
        assert float(np.min(pts[:, 1])) < 0.0 < float(np.max(pts[:, 1]))


class TestXfoilChordAdapter:
    """``outline_frames`` is the one B→NACA hop; it must untwist, not re-twist."""

    def _ring_B(self) -> np.ndarray:
        pivot = 0.333
        return np.array(
            [
                [0.0, 1.0 - pivot],
                [0.1, 0.5 - pivot],
                [0.0, 0.0 - pivot],
                [-0.1, 0.5 - pivot],
            ],
            dtype=float,
        )

    def test_unit_chord_comes_off_the_chord_axis(self):
        ring = self._ring_B()
        geom = SectionOutlineGeometry.from_closed_contour(ring, ensure_clockwise=True)
        naca, _ = blade_outline_to_unit_chord(geom, 0.0)
        coords = naca.coordinates
        assert float(coords[:, 0].min()) == pytest.approx(0.0, abs=1e-9)
        assert float(coords[:, 0].max()) == pytest.approx(1.0, abs=1e-6)
        # Reading the chord off the canonical-B thickness column (the pre-canon
        # bug) normalises by ~t/c and blows the thickness column up to ~1.
        assert float(np.abs(coords[:, 1]).max()) < 0.3

    @pytest.mark.parametrize("twist", [0.0, 0.2, -0.35])
    def test_untwists_rather_than_double_twisting(self, twist):
        ring = self._ring_B()
        twisted = rotate_S_to_B(ring, twist)
        geom = SectionOutlineGeometry.from_closed_contour(twisted, ensure_clockwise=True)
        naca, _ = blade_outline_to_unit_chord(geom, twist)
        coords = naca.coordinates
        i_thick = int(np.argmax(np.abs(coords[:, 1])))
        assert float(coords[i_thick, 0]) == pytest.approx(0.5, abs=1e-6)
        assert float(np.abs(coords[i_thick, 1])) == pytest.approx(0.1, abs=1e-6)

    def test_rotation_matrix_is_the_canonical_one(self):
        """The adapter matrix must be ``rotate_B_to_S`` + swap, not a local 2×2."""
        twist = 0.27
        pts_B = np.array([[0.3, -0.4], [-0.2, 0.9], [0.0, 0.0]], dtype=float)
        got = pts_B @ blade_to_naca_rotation(twist)
        expect_S = rotate_S_to_B(pts_B, -twist)  # == rotate_B_to_S(pts_B, twist)
        np.testing.assert_allclose(got, expect_S[:, ::-1], atol=1e-12)


class TestFailureIndexIsEvaluatedInM:
    """Canon: 'Feeding E-frame stress to hashin_fi returns a wrong index.'"""

    def _lam(self, angle_deg: float):
        from blade_structure_utilities.material_utilities.laminate import LaminateDefinition
        from blade_structure_utilities.material_utilities.orthotropic import OrthotropicPly

        ply = OrthotropicPly(
            name="gfrp", E1=30e9, E2=8e9, G12=3e9, nu12=0.3, rho=1.5e3, t_ply=2e-4,
            Xt=500e6, Xc=300e6, Yt=50e6, Yc=200e6, S12=40e6, Zt=1e6, S13=1e6, S23=1e6,
        )
        return LaminateDefinition(plies=[(ply, angle_deg), (ply, -angle_deg)])

    def _row(self):
        from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import (
            ShellPanelResultants,
        )

        return ShellPanelResultants(
            N11=8.0e3, N22=-4.0e2, N12=6.0e2,
            M11=3.0, M22=-12.0, M12=1.5,
            Q1=1.0, Q2=2.0, thickness_m=8e-4,
            panel_index=0, station_index=0,
        )

    def test_off_axis_plies_use_the_material_frame(self):
        from blade_structure_utilities.material_utilities.failure import hashin_fi
        from blade_structure_utilities.material_utilities.laminate import ply_mid_stresses_laminate
        from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.api.fi_recovery import (
            hashin_fi_per_ply_from_rows_batch,
        )

        lam = self._lam(45.0)
        rows = [self._row()]
        fi_out, needs_review = hashin_fi_per_ply_from_rows_batch(rows, lam, [])
        assert not needs_review[0]

        abd = lam.stiffness_bundle().ABD
        nm = np.concatenate([rows[0].to_N_vec()[:3], rows[0].to_M_vec()[:3]])
        sol = np.linalg.solve(abd, nm)
        sig_lam = ply_mid_stresses_laminate(lam, sol[:3], sol[3:6])

        def _fi(sig3, pi):
            ply = lam.plies[pi][0]
            return float(
                hashin_fi(
                    np.asarray(sig3, dtype=float).reshape(1, 3),
                    np.array([ply.Xt]), np.array([ply.Xc]),
                    np.array([ply.Yt]), np.array([ply.Yc]), np.array([ply.S12]),
                )[0]
            )

        for pi, sig in enumerate(sig_lam):
            sig_E = np.asarray(sig, dtype=float).ravel()[:3]
            sig_M = rotate_ply_stress_E_to_M(sig_E, lam.plies[pi][1])
            assert fi_out[0, pi] == pytest.approx(_fi(sig_M, pi))
            # ...and the E-frame answer is genuinely different, so this is not a
            # tautology on a stack where the two frames happen to coincide.
            assert fi_out[0, pi] != pytest.approx(_fi(sig_E, pi), rel=1e-3)

    def test_zero_deg_plies_are_frame_agnostic(self):
        """0° plies: E ≡ M, so the rotation must be a no-op."""
        from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.api.fi_recovery import (
            hashin_fi_per_ply_from_rows_batch,
        )

        lam = self._lam(0.0)
        fi_out, _ = hashin_fi_per_ply_from_rows_batch([self._row()], lam, [])
        assert np.all(np.isfinite(fi_out))
        assert float(np.max(fi_out)) > 0.0


class TestTransformsAreNotReimplemented:
    """Import-boundary guard: the canon owns twist, +x3, and the ply transforms."""

    # Guard applies to package sources; the tests themselves quote the banned
    # patterns in their own assertion messages.
    _SKIP_DIRS = ("benchmarks", "output", "__pycache__", "tests")

    def _package_sources(self):
        for path in _PKG.rglob("*.py"):
            rel = path.relative_to(_PKG)
            if any(part in self._SKIP_DIRS for part in rel.parts):
                continue
            yield rel, path.read_text(encoding="utf-8")

    def test_no_local_ply_transform_matrices(self):
        """A hand-built ``[[c*c, s*s, 2*s*c], ...]`` is T_sigma or T_epsilon."""
        offenders = [
            str(rel)
            for rel, src in self._package_sources()
            if "c * c, s * s" in src or "c*c, s*s" in src
        ]
        assert offenders == [], (
            "ply stress/strain transforms must come from coordinate_frames "
            f"(ply_stress_transform_E_to_M / ply_strain_transform_E_to_M): {offenders}"
        )

    def test_no_local_section_normal(self):
        """``[-t[1], t[0]]`` is the +x3 convention owned by section_normal_2d."""
        allowed = {
            # Vectorised over a whole polyline; asserted equivalent below.
            Path("processing/static/diagnostics/thin_wall_plots.py"),
        }
        offenders = [
            str(rel)
            for rel, src in self._package_sources()
            if ("-float(t[1]), float(t[0])" in src or "-t[:, 1], t[:, 0]" in src)
            and rel not in allowed
        ]
        assert offenders == [], (
            f"+x3 normals must delegate to coordinate_frames.section_normal_2d: {offenders}"
        )

    def test_vectorised_outward_normals_match_the_canon(self):
        from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.thin_wall_plots import (
            outward_normals,
        )

        pts = np.array([[0.0, 0.0], [0.1, 0.4], [0.05, 0.9]], dtype=float)
        got = outward_normals(pts)
        # End vertices use the one-sided tangent, which is what section_normal_2d
        # is given here; interior vertices use the central difference.
        np.testing.assert_allclose(got[0], section_normal_2d(pts[1] - pts[0]), atol=1e-12)
        np.testing.assert_allclose(got[-1], section_normal_2d(pts[-1] - pts[-2]), atol=1e-12)
        np.testing.assert_allclose(got[1], section_normal_2d(pts[2] - pts[0]), atol=1e-12)

    def test_mesh_map_normal_delegates(self):
        from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.shell_mesh_map import (
            normal_2d,
        )

        t = np.array([0.3, -0.7])
        np.testing.assert_allclose(normal_2d(t), section_normal_2d(t), atol=1e-15)


class TestSectionPlotContract:
    """Horizontal Y_B, vertical X_B, LE on the RHS, data never negated."""

    def _rendered_axes(self, mesh, tmp_path):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics import (
            example_plots,
        )

        captured = {}
        original = plt.subplots

        def _spy(*args, **kwargs):
            fig, ax = original(*args, **kwargs)
            captured.setdefault("ax", ax)
            return fig, ax

        plt.subplots = _spy
        try:
            example_plots.save_mitc4_v2_section_mesh_figure(tmp_path / "mesh.png", mesh)
        finally:
            plt.subplots = original
        return captured["ax"]

    def test_axes_are_labelled_and_inverted(self, tmp_path):
        pytest.importorskip("matplotlib")
        from blade_structure_utilities.mesh_utilities.section_mesh.strip.strip_mesh import (
            build_section_strip_mesh,
        )

        pt = build_shell_panel_topology("2412", [0.35], chord_m=CHORD, twist_rad=0.15)
        mesh = build_section_strip_mesh(pt.shell_inputs, n_elements_per_panel=4)
        ax = self._rendered_axes(mesh, tmp_path)

        assert ax.get_xlabel() == SECTION_PLOT_XLABEL
        assert ax.get_ylabel() == SECTION_PLOT_YLABEL
        assert ax.xaxis_inverted(), "leading edge must render on the RHS"

    def test_drawn_extents_are_chord_horizontal(self, tmp_path):
        """The 90°-rotated rendering would put the chord on the vertical axis."""
        pytest.importorskip("matplotlib")
        from blade_structure_utilities.mesh_utilities.section_mesh.strip.strip_mesh import (
            build_section_strip_mesh,
        )

        pt = build_shell_panel_topology("2412", [0.35], chord_m=CHORD, twist_rad=0.15)
        mesh = build_section_strip_mesh(pt.shell_inputs, n_elements_per_panel=4)
        ax = self._rendered_axes(mesh, tmp_path)

        drawn = np.vstack([np.asarray(ln.get_xydata(), dtype=float) for ln in ax.get_lines()])
        assert float(np.ptp(drawn[:, 0])) > 3.0 * float(np.ptp(drawn[:, 1]))

    def test_stored_geometry_is_not_negated_for_display(self):
        """The LE-on-RHS flip is display-only; Y_LE stays negative in the data."""
        pt = build_shell_panel_topology("2412", [0.35], chord_m=CHORD, twist_rad=0.15)
        pts = np.vstack([np.asarray(p.nodes, dtype=float) for p in pt.zones])
        assert float(np.min(pts[:, 1])) < 0.0 < float(np.max(pts[:, 1]))


class TestFlapEdgeMapReachesTheRightAxis:
    """The load channels must bend the section about the axis they name.

    ``blade_resultants_to_shell_channels`` maps ``Mz <- flap`` / ``My <- edge``,
    and ``beam_resultants_to_nodal_load`` turns those into axial couples about
    the stored columns. Whether that lands on the flapwise or the edgewise axis
    depends entirely on the section frame — and no existing load test can see
    it, because each is self-consistent within whatever frame it is handed.

    This is the assertion that was missing when the strips carried the swapped
    rotated-NACA frame: the adapter said ``Mz <- flap`` while the couple bent
    the section edgewise.
    """

    def _axial_node_loads(self, **named_resultants):
        """Nodal axial load per node, and the canonical-B node positions."""
        from blade_structure_utilities.geometry_utilities.coordinate_frames import (
            blade_resultants_to_shell_channels,
        )
        from blade_structure_utilities.geometry_utilities.section_contract.beam_resultant_vector import (
            BeamResultantVector,
        )
        from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.loading.beam_resultants_to_nodal_load import (
            beam_resultants_to_nodal_load,
        )
        from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.connected_assembly_core import (
            U_X,
            gdof,
        )
        from blade_structure_utilities.fem_utilities.section_shell_fea.simulation_runner.section_linear_static_runner import (
            SectionLinearStaticRunner,
        )
        from blade_structure_utilities.mesh_utilities.section_mesh.strip.strip_mesh import (
            build_section_strip_mesh,
        )

        pt = build_shell_panel_topology("2412", [0.35], chord_m=CHORD, twist_rad=0.0)
        mesh = build_section_strip_mesh(pt.shell_inputs, n_elements_per_panel=6)
        runner = SectionLinearStaticRunner(
            panel_topology=pt, section_strip_mesh=mesh,
            My=0.0, Mz=0.0, load_channel="equipollent",
        ).execute()
        node_meta = runner.assembly.node_meta

        ch = blade_resultants_to_shell_channels(**named_resultants)
        vec = BeamResultantVector.from_named(
            N=0.0, Vy=ch.Vy, Vz=ch.Vz, My=ch.My, Mz=ch.Mz, T=0.0,
        )
        f = beam_resultants_to_nodal_load(node_meta, vec)
        fx = np.array(
            [f[gdof(i, U_X, 6)] for i in range(node_meta.n_nodes)], dtype=float
        )
        return np.asarray(node_meta.yz, dtype=float), fx

    @staticmethod
    def _abs_corr(pos: np.ndarray, fx: np.ndarray) -> tuple[float, float]:
        loaded = np.isfinite(fx) & np.isfinite(pos).all(axis=1) & (np.abs(fx) > 0.0)
        r_x = abs(float(np.corrcoef(pos[loaded, 0], fx[loaded])[0, 1]))
        r_y = abs(float(np.corrcoef(pos[loaded, 1], fx[loaded])[0, 1]))
        return r_x, r_y

    def test_flapwise_moment_produces_flapwise_bending(self):
        """M_flapwise: axial load varies with X_B, and not at all with Y_B."""
        pos, fx = self._axial_node_loads(M_flapwise=1.0e5)
        r_x, r_y = self._abs_corr(pos, fx)
        assert r_y == pytest.approx(0.0, abs=1e-9), (
            f"a flapwise moment leaked into edgewise bending: |corr(Y_B)|={r_y:.4f}"
        )
        assert r_x > 0.5, f"expected axial load to track X_B; |corr(X_B)|={r_x:.4f}"

    def test_edgewise_moment_produces_edgewise_bending(self):
        """M_edgewise: the mirror image — axial load varies with Y_B only."""
        pos, fx = self._axial_node_loads(M_edgewise=1.0e5)
        r_x, r_y = self._abs_corr(pos, fx)
        assert r_x == pytest.approx(0.0, abs=1e-9), (
            f"an edgewise moment leaked into flapwise bending: |corr(X_B)|={r_x:.4f}"
        )
        assert r_y > 0.5, f"expected axial load to track Y_B; |corr(Y_B)|={r_y:.4f}"
