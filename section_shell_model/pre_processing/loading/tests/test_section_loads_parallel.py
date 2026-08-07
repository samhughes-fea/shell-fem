"""Serial vs parallel parity for section_loads_impl (synthetic aero)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from blade_precompute.orchestration import PrecomputeOrchestrationContext
from blade_precompute.orchestration.precompute.containers import PrecomputeInputs
from blade_precompute.orchestration.precompute.stages_pkg.loads import section_loads_impl
from blade_structure_utilities.geometry_utilities.section_geometry.core.system_layout import resolve_system_type
from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.loading.containers.types import SectionLoadsJobConfig
from blade_precompute.orchestration.component_materials import ComponentMaterialsMap


def _inputs(tmp_path: Path) -> PrecomputeInputs:
    z = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    return PrecomputeInputs(
        spanwise_path=tmp_path / "span.dat",
        extreme_loads_path=tmp_path / "loads.dat",
        span_r_z_m=z,
        radial_r_m=z + 10.0,
        chord_m=np.array([1.0, 0.98, 0.95], dtype=np.float64),
        twist_deg=np.zeros(3, dtype=np.float64),
        kappa0_x=np.zeros(3, dtype=np.float64),
        kappa0_y=np.zeros(3, dtype=np.float64),
        kappa0_z=np.zeros(3, dtype=np.float64),
        naca_m=np.full(3, 2.0, dtype=np.float64),
        naca_p=np.full(3, 4.0, dtype=np.float64),
        naca_xx=np.full(3, 12.0, dtype=np.float64),
        naca_series=np.full(3, 4, dtype=np.int64),
        loads_r_z_m=z,
        q_y_Npm=np.array([1000.0, 1100.0, 1200.0], dtype=np.float64),
        q_z_Npm=np.array([50.0, 55.0, 60.0], dtype=np.float64),
        m_x_Nmpm=np.zeros(3, dtype=np.float64),
        log_dump_level="summary",
    )


def _orch() -> PrecomputeOrchestrationContext:
    return PrecomputeOrchestrationContext(
        system_type_key="2D-CN",
        layout=resolve_system_type("2D-CN"),
        component_materials=ComponentMaterialsMap(skin=0, spar_cap=0, shear_web=0),
    )


def _job_config() -> SectionLoadsJobConfig:
    return SectionLoadsJobConfig(
        u_inf_m_s=10.0,
        tip_speed_ratio=5.0,
        use_direct_xfoil_scalars=True,
        direct_reynolds_number=1e6,
        direct_mach_number=0.1,
        direct_alpha_deg=5.0,
    )


def _traction_arrays(out) -> dict[int, np.ndarray]:
    return {
        int(i): np.asarray(f.traction_field.total_forces, dtype=np.float64)
        for i, f in out.station_load_fields.items()
    }


@pytest.mark.parametrize("persist_pngs", [False, True])
def test_section_loads_serial_parallel_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    persist_pngs: bool,
) -> None:
    monkeypatch.setenv("ADVANCED_BLADE_SECTION_LOADS_SYNTHETIC_AERO", "1")
    inp = _inputs(tmp_path)
    orch = _orch()
    job_config = _job_config()
    common = dict(
        job_config=job_config,
        section_plot_station_spec="all",
        orchestration=orch,
        persist_pngs=persist_pngs,
    )
    serial_dir = tmp_path / "serial"
    parallel_dir = tmp_path / "parallel"
    serial = section_loads_impl(
        inp,
        serial_dir,
        section_solve_n_workers=1,
        **common,
    )
    parallel = section_loads_impl(
        inp,
        parallel_dir,
        section_solve_n_workers=2,
        **common,
    )

    assert serial.station_indices == parallel.station_indices
    serial_tr = _traction_arrays(serial)
    parallel_tr = _traction_arrays(parallel)
    assert serial_tr.keys() == parallel_tr.keys()
    for i in serial_tr:
        np.testing.assert_allclose(serial_tr[i], parallel_tr[i], rtol=1e-6, atol=1e-3)


def test_section_station_load_field_pickle_roundtrip() -> None:
    from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.loading.tests.test_load_field_serialization import (
        _example_load_field,
    )

    field = _example_load_field()
    restored = pickle.loads(pickle.dumps(field))
    np.testing.assert_allclose(
        restored.traction_field.total_forces,
        field.traction_field.total_forces,
    )
