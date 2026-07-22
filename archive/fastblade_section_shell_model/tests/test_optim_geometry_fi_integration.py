"""Real-geometry FI integration: OptimBladeGeometry → shell_fi_batch."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from blade_precompute.materials.laminate import LaminateDefinition
from blade_precompute.materials.ply import OrthotropicPly
from blade_precompute.section_definition.types import SectionDefinition, SubcomponentGeometry
from blade_precompute.section_optimisation.core.types import OptimBladeGeometry
from blade_precompute.section_optimisation.engine.shell_fi_eval import shell_fi_batch
from blade_precompute.section_shell_model.api.optim_geometry_builder import (
    pipeline_b_section_from_optim_geometry,
)

from blade_precompute.section_shell_model.api.pipeline_b import (
    build_pipeline_b_section,
    build_tapered_pipeline_b_section,
)


def _ply(name: str = "g") -> OrthotropicPly:
    return OrthotropicPly(
        name=name,
        E1=30e9,
        E2=8e9,
        G12=3e9,
        nu12=0.3,
        rho=1.5e3,
        t_ply=2e-4,
        Xt=500e6,
        Xc=300e6,
        Yt=50e6,
        Yc=200e6,
        S12=40e6,
        Zt=1e6,
        S13=12e6,
        S23=1e6,
    )


def _section(z: float) -> SectionDefinition:
    skin_lam = LaminateDefinition(plies=[(_ply("skin"), 0.0), (_ply("skin"), 90.0)])
    line = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    subs = [
        SubcomponentGeometry(name="skin", midsurface_coords=line, material=skin_lam, thickness=0.002),
    ]
    return SectionDefinition(station_z=z, subcomponents=subs)


def _naca_profile(chord: float) -> np.ndarray:
    from blade_utilities.sdf.airfoil import AirfoilSDF

    af = AirfoilSDF.from_naca("2412", n_points=80, chord=float(chord))
    return np.asarray(af.vertices, dtype=np.float64)[:, :2]


def _multi_station_bg() -> OptimBladeGeometry:
    z = np.array([0.0, 0.45, 0.9], dtype=np.float64)
    chords = np.array([1.05, 1.0, 0.94], dtype=np.float64)
    twist = np.array([2.0, 5.0, 8.0], dtype=np.float64)
    profiles = [_naca_profile(c) for c in chords]
    return OptimBladeGeometry(
        z_stations=z,
        r_ref=np.zeros((len(z), 3), dtype=np.float64),
        kappa0=np.zeros((len(z), 3), dtype=np.float64),
        chord=chords,
        twist=twist,
        airfoil_profiles=profiles,
        web_positions=np.array([0.35], dtype=np.float64),
        subcomponent_materials={},
    )


def _shear_resultants(n_s: int) -> np.ndarray:
    r = np.zeros((n_s, 7), dtype=np.float64)
    for i in range(n_s):
        r[i, 4] = 8.0e3 * (1.0 + 0.1 * i)
        r[i, 5] = 4.0e3 * (1.0 + 0.05 * i)
        r[i, 1] = 1.2e4
    return r


@patch(
    "blade_precompute.section_shell_model.api.optim_geometry_builder.pipeline_b_section_from_optim_geometry"
)
@patch(
    "blade_precompute.section_shell_model.api.optim_geometry_builder.build_tapered_pipeline_b_from_optim_stations"
)
def test_optim_geometry_fi_finite_and_trusted(mock_taper, mock_pb):
    """NACA2412 via optim dispatch: Hashin FI finite; tapered fi_il > 0 under shear."""
    pb = build_pipeline_b_section("2412", [0.35], chord_m=1.0)
    tapered = build_tapered_pipeline_b_section(
        "2412", [0.35], chord_inboard_m=1.0, chord_outboard_m=0.96, L_x=0.45,
    )
    mock_pb.return_value = pb
    mock_taper.return_value = tapered

    bg = _multi_station_bg()
    section_defs = [_section(float(z)) for z in bg.z_stations]
    r_sec = _shear_resultants(len(section_defs))

    _, fi_h, msk, _, _, fi_il, nr = shell_fi_batch(
        section_defs,
        r_sec,
        bg,
        4,
        ["skin"],
        n_elements_per_panel=4,
        use_tapered_il=True,
    )

    assert np.any(msk), "expected trusted Hashin rows"
    assert np.all(np.isfinite(fi_h[msk]))
    assert not np.any(nr)
    assert float(np.max(fi_il)) > 0.0


def test_optim_geometry_real_pipeline_b_builder() -> None:
    """Real pipeline_b builder on resampled spanwise NACA (no mocks)."""
    from blade_precompute.section_geometry.api.factory import resolve_system_type
    from blade_precompute.orchestration.component_materials import ComponentMaterialsMap
    from blade_precompute.orchestration.precompute import (
        build_optim_blade_geometry_from_spanwise,
        load_inputs,
        load_material_library_dat,
        resample_precompute_inputs,
    )
    from blade_precompute.orchestration.precompute.grid import LinspaceSpec, linspace_from_spec
    from blade_precompute.orchestration.precompute.stages_pkg import default_dv0
    from blade_precompute.section_optimisation.engine.section_builder import SectionBuilder

    repo = Path(__file__).resolve().parents[3]
    data_dir = repo / "data_library"
    if not data_dir.is_dir():
        pytest.skip("data_library not present")

    inp_full = load_inputs(data_dir)
    z0, z1 = float(inp_full.span_r_z_m[0]), float(inp_full.span_r_z_m[-1])
    z1 = z0 + 0.85 * (z1 - z0)
    z_struct = linspace_from_spec(LinspaceSpec(z0, z1, 2))
    inp = resample_precompute_inputs(inp_full, z_struct)
    mat_table = load_material_library_dat(data_dir / "material_library.dat")
    logical = ComponentMaterialsMap(skin=0, spar_cap=0, shear_web=0)
    layout = resolve_system_type("2D-CN")
    bg = build_optim_blade_geometry_from_spanwise(
        inp,
        mat_table=mat_table,
        logical=logical.to_dict(),
        system_layout=layout,
        naca_airfoil_n_points=60,
    )
    dv0 = default_dv0(2)
    section_defs = SectionBuilder.build(dv0, bg)

    for i, sd in enumerate(section_defs):
        pb = pipeline_b_section_from_optim_geometry(bg, sd, i)
        assert len(pb.panels) >= 4

    from blade_precompute.section_optimisation.engine.shell_fi_eval import _composite_subcomponents

    cnames: list[str] = []
    for sd in section_defs:
        for sub in _composite_subcomponents(sd):
            if sub.name not in cnames:
                cnames.append(sub.name)

    r_sec = _shear_resultants(2)
    _, fi_h, msk, _, _, fi_il, nr = shell_fi_batch(
        section_defs,
        r_sec,
        bg,
        4,
        cnames,
        n_elements_per_panel=4,
        use_tapered_il=True,
    )
    assert np.all(np.isfinite(fi_h))
    assert float(np.max(fi_il)) > 0.0
    if np.any(msk):
        assert not np.any(nr[msk])


def test_optim_geometry_web_orientation_propagates_to_pipeline() -> None:
    """``4D-F`` layout must build flapwise webs in ``pipeline_b_section_from_optim_geometry``."""
    from blade_precompute.section_geometry.api.factory import resolve_system_type
    from blade_precompute.orchestration.component_materials import ComponentMaterialsMap
    from blade_precompute.orchestration.precompute import (
        build_optim_blade_geometry_from_spanwise,
        load_inputs,
        load_material_library_dat,
        resample_precompute_inputs,
    )
    from blade_precompute.orchestration.precompute.grid import LinspaceSpec, linspace_from_spec
    from blade_precompute.orchestration.precompute.stages_pkg import default_dv0
    from blade_precompute.section_optimisation.engine.section_builder import SectionBuilder

    repo = Path(__file__).resolve().parents[3]
    data_dir = repo / "data_library"
    if not data_dir.is_dir():
        pytest.skip("data_library not present")

    inp_full = load_inputs(data_dir)
    z0 = float(inp_full.span_r_z_m[0])
    z_struct = linspace_from_spec(LinspaceSpec(z0, z0, 1))
    inp = resample_precompute_inputs(inp_full, z_struct)
    mat_table = load_material_library_dat(data_dir / "material_library.dat")
    logical = ComponentMaterialsMap(skin=0, spar_cap=0, shear_web=0).to_dict()

    layout_f = resolve_system_type("4D-F")
    layout_cn = resolve_system_type("4D-CN")
    bg_f = build_optim_blade_geometry_from_spanwise(
        inp, mat_table=mat_table, logical=logical, system_layout=layout_f, naca_airfoil_n_points=60,
    )
    bg_cn = build_optim_blade_geometry_from_spanwise(
        inp, mat_table=mat_table, logical=logical, system_layout=layout_cn, naca_airfoil_n_points=60,
    )
    assert bg_f.web_orientation == "flapwise"
    assert bg_cn.web_orientation == "chord_normal"

    # Flapwise placement uses B-frame lookup only when twist is non-zero.
    bg_f.twist = np.asarray(bg_f.twist, dtype=np.float64).copy()
    bg_cn.twist = np.asarray(bg_cn.twist, dtype=np.float64).copy()
    bg_f.twist[0] = 12.0
    bg_cn.twist[0] = 12.0

    dv0 = default_dv0(1)
    sd_f = SectionBuilder.build(dv0, bg_f)[0]
    sd_cn = SectionBuilder.build(dv0, bg_cn)[0]
    pb_f = pipeline_b_section_from_optim_geometry(bg_f, sd_f, 0)
    pb_cn = pipeline_b_section_from_optim_geometry(bg_cn, sd_cn, 0)

    web_f = next(p for p in pb_f.panels if "web" in str(getattr(p, "label", "")).lower())
    web_cn = next(p for p in pb_cn.panels if "web" in str(getattr(p, "label", "")).lower())
    assert web_f.nodes.shape == web_cn.nodes.shape
    assert not np.allclose(web_f.nodes, web_cn.nodes, atol=1e-9)
