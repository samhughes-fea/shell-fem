"""Per-station surface traction field construction."""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..containers.types import SectionLoadsJobConfig
from blade_structure_utilities.section_geometry.interface.shell_strip_topology import concat_skin_ring_vertices
from ..containers.types import (
    DistributedStationLoadSpec,
    LoadFieldProvenance,
    SectionStationLoadField,
    StationAeroContext,
)
from .incident_aero import (
    build_xfoil_run_config,
    derive_incident_aero,
)
from .outline_frames import (
    bframe_outline_to_sframe_chord,
    xfoil_te_arc_offset,
)
from blade_structure_utilities.section_shell_model.pre_processing.mesh.shell_inputs_from_section import ShellMeshInputs
from blade_structure_utilities.point_to_surftraction.distributions import CfDistribution, CpDistribution
from blade_structure_utilities.point_to_surftraction.geometry import SectionOutlineGeometry
from blade_structure_utilities.point_to_surftraction.integration import ForceResultantIntegrator
from blade_structure_utilities.point_to_surftraction.matching import LoadMatcher
from blade_structure_utilities.point_to_surftraction.tractions import (
    PressureTractionBuilder,
    ShearTractionBuilder,
    TractionField,
)
from blade_structure_utilities.point_to_surftraction.xfoil.xfoil_adapter import XfoilAdapter, XfoilRunResult

_SYNTHETIC_AERO_ENV = "ADVANCED_BLADE_SECTION_LOADS_SYNTHETIC_AERO"
_UNIFORM_CP = -1.0
_UNIFORM_CF = 0.01
# Shaped fallback gives full-rank pressure/shear resultants for LoadMatcher.
_SYNTHETIC_CP_OFFSET = -0.8
_SYNTHETIC_CP_AMP = 0.6
_SYNTHETIC_CF_BASE = 0.01
_SYNTHETIC_CF_AMP = 0.005


def _synthetic_aero_enabled(job_config: SectionLoadsJobConfig) -> bool:
    if job_config.use_direct_xfoil_scalars:
        return True
    return os.environ.get(_SYNTHETIC_AERO_ENV, "0").strip() in {"1", "true", "True", "yes"}


def _xfoil_available(job_config: SectionLoadsJobConfig) -> bool:
    if job_config.xfoil_executable:
        return Path(job_config.xfoil_executable).is_file()
    return shutil.which("xfoil") is not None


def _skin_ring_from_shell_inputs(shell_inputs: ShellMeshInputs) -> NDArray[np.float64]:
    strips = tuple(shell_inputs.midlines)
    ring = concat_skin_ring_vertices(strips)
    if ring.shape[0] < 3:
        raise ValueError(
            "ShellMeshInputs must contain at least three skin-ring vertices; "
            f"got shape {ring.shape}."
        )
    return np.asarray(ring, dtype=np.float64)


def _panel_centre_stations(geometry: SectionOutlineGeometry) -> NDArray[np.float64]:
    if geometry.perimeter <= 0.0:
        raise ValueError("outline perimeter must be strictly positive.")
    return np.asarray(geometry.arc_length / geometry.perimeter, dtype=float)


def _synthetic_cp_cf(
    geometry: SectionOutlineGeometry,
    *,
    uniform: bool = False,
) -> tuple[CpDistribution, CfDistribution]:
    stations = _panel_centre_stations(geometry)
    n = int(stations.size)
    if uniform:
        cp_vals = np.full(n, _UNIFORM_CP, dtype=float)
        cf_vals = np.full(n, _UNIFORM_CF, dtype=float)
    else:
        phase = 2.0 * np.pi * stations
        x_norm = geometry.panel_centres[:, 0] / max(
            float(np.max(geometry.panel_centres[:, 0])), 1e-12
        )
        chord_bias = x_norm - 0.5
        cp_vals = _SYNTHETIC_CP_OFFSET + _SYNTHETIC_CP_AMP * np.cos(phase) + 0.35 * chord_bias
        cf_vals = _SYNTHETIC_CF_BASE + _SYNTHETIC_CF_AMP * np.sin(phase) + 0.012 * chord_bias
    return CpDistribution(stations=stations, values=cp_vals), CfDistribution(
        stations=stations, values=cf_vals
    )


def _remap_arc_distribution(dist: Any, s_offset: float, cls: type) -> Any:
    """Shift a normalized-arc distribution by ``s_offset`` (mod 1) and re-sort."""
    stations = (np.asarray(dist.stations, dtype=float) + s_offset) % 1.0
    order = np.argsort(stations, kind="mergesort")
    stations = stations[order]
    values = np.asarray(dist.values, dtype=float)[order]
    keep = np.concatenate([[True], np.diff(stations) > 1e-12])
    return cls(stations=stations[keep], values=values[keep])


def _xfoil_cache_key(
    xfoil_geometry: Any,
    xfoil_cfg: Any,
) -> str:
    coords = np.asarray(getattr(xfoil_geometry, "coordinates", xfoil_geometry), dtype=np.float64)
    rounded = np.round(coords, 8).tobytes()
    payload = (
        rounded,
        float(xfoil_cfg.reynolds_number),
        float(xfoil_cfg.mach_number),
        float(xfoil_cfg.alpha_deg),
        int(xfoil_cfg.n_iterations),
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()[:16]


def _try_xfoil_cache_hit(
    cache_dir: Path,
    cache_key: str,
) -> XfoilRunResult | None:
    sub = cache_dir / cache_key
    cp_path = sub / "cp_output.dat"
    if not cp_path.is_file():
        return None
    from blade_structure_utilities.point_to_surftraction.xfoil.xfoil_io import (
        parse_xfoil_cf_file,
        parse_xfoil_cp_file,
    )

    cp = parse_xfoil_cp_file(cp_path)
    cf_path = sub / "cf_output.dat"
    cf = parse_xfoil_cf_file(cf_path) if cf_path.is_file() else None
    return XfoilRunResult(
        cp_distribution=cp,
        cf_distribution=cf,
        cp_output_file=cp_path,
        cf_output_file=cf_path if cf_path.is_file() else None,
        coordinate_file=sub / "airfoil.dat",
        command_file=sub / "xfoil_commands.in",
        polar_output_file=sub / "polar_output.dat",
        stdout="",
        stderr="",
        cp_available=True,
        cf_available=cf is not None,
    )


def _store_xfoil_cache_artifacts(cache_dir: Path, cache_key: str, result: XfoilRunResult) -> None:
    sub = cache_dir / cache_key
    sub.mkdir(parents=True, exist_ok=True)
    for src_attr, name in (
        ("cp_output_file", "cp_output.dat"),
        ("cf_output_file", "cf_output.dat"),
        ("coordinate_file", "airfoil.dat"),
        ("command_file", "xfoil_commands.in"),
        ("polar_output_file", "polar_output.dat"),
    ):
        src = getattr(result, src_attr, None)
        if src is None:
            continue
        src_path = Path(src)
        if src_path.is_file():
            dst = sub / name
            if src_path.resolve() != dst.resolve():
                shutil.copy2(src_path, dst)


def _run_xfoil_or_synthetic(
    geometry: SectionOutlineGeometry,
    job_config: SectionLoadsJobConfig,
    incident: Any,
    *,
    twist_rad: float,
    cache_dir: Path | None,
) -> tuple[CpDistribution, CfDistribution, dict[str, Any], bool, bool]:
    xfoil_cfg = build_xfoil_run_config(job_config, incident)
    artifacts: dict[str, Any] = {}
    use_synthetic = _synthetic_aero_enabled(job_config) or not _xfoil_available(job_config)

    if use_synthetic:
        cp, cf = _synthetic_cp_cf(geometry)
        artifacts["mode"] = "synthetic_shaped_aero"
        return cp, cf, artifacts, False, True

    adapter = XfoilAdapter()
    workdir = cache_dir
    if workdir is not None:
        workdir.mkdir(parents=True, exist_ok=True)
        xfoil_cfg = replace(
            xfoil_cfg,
            keep_temporary_files=True,
            working_directory=str(workdir),
        )

    xfoil_geometry, _ = bframe_outline_to_sframe_chord(geometry, twist_rad, te_first=True)
    s_te = xfoil_te_arc_offset(geometry, twist_rad)

    cache_key: str | None = None
    if workdir is not None:
        cache_key = _xfoil_cache_key(xfoil_geometry, xfoil_cfg)
        cached = _try_xfoil_cache_hit(workdir, cache_key)
        if cached is not None:
            result = cached
        else:
            result = None
    else:
        result = None

    if result is None:
        try:
            result = adapter.run_cp_analysis(xfoil_geometry, xfoil_cfg)
        except (FileNotFoundError, RuntimeError, OSError) as exc:
            if not _synthetic_aero_enabled(job_config):
                raise
            cp, cf = _synthetic_cp_cf(geometry)
            artifacts["mode"] = "synthetic_shaped_aero"
            artifacts["xfoil_error"] = str(exc)
            return cp, cf, artifacts, False, True
        if workdir is not None and cache_key is not None:
            _store_xfoil_cache_artifacts(workdir, cache_key, result)

    # XFOIL distributions are parameterized by normalized arc length along the
    # TE-first rolled ring; shift them back onto the original outline's arc.
    cp = _remap_arc_distribution(result.cp_distribution, s_te, CpDistribution)
    cf = (
        _remap_arc_distribution(result.cf_distribution, s_te, CfDistribution)
        if result.cf_distribution is not None
        else None
    )
    if cf is None:
        _, cf = _synthetic_cp_cf(geometry)
    artifacts.update(
        {
            "mode": "xfoil",
            "cp_output_file": str(result.cp_output_file),
            "coordinate_file": str(result.coordinate_file),
            "command_file": str(result.command_file),
        }
    )
    if result.cf_output_file is not None:
        artifacts["cf_output_file"] = str(result.cf_output_file)
    return cp, cf, artifacts, True, False


def _build_traction_field(
    geometry: SectionOutlineGeometry,
    distributed: DistributedStationLoadSpec,
    cp: CpDistribution,
    cf: CfDistribution,
) -> tuple[TractionField, int]:
    stations = _panel_centre_stations(geometry)
    cp_values = cp.interpolate(stations)
    cf_values = cf.interpolate(stations)

    pressure_builder = PressureTractionBuilder()
    shear_builder = ShearTractionBuilder()
    integrator = ForceResultantIntegrator()
    matcher = LoadMatcher()

    pressure_forces = pressure_builder.build(geometry, cp_values)
    shear_forces = shear_builder.build(geometry, cf_values)
    pressure_resultant = integrator.total_pressure_force(pressure_forces)
    shear_resultant = integrator.total_shear_force(shear_forces)

    # B-frame: component 0 = flapwise (y), component 1 = edgewise (z).
    target_force = np.array([distributed.q_y_Npm, distributed.q_z_Npm], dtype=float)
    match = matcher.match(pressure_resultant, shear_resultant, target_force)

    scaled_pressure = match.lambda_pressure * pressure_forces
    scaled_shear = match.lambda_shear * shear_forces
    total_forces = scaled_pressure + scaled_shear
    traction = TractionField(
        pressure_forces=scaled_pressure,
        shear_forces=scaled_shear,
        total_forces=total_forces,
    )
    return traction, int(match.rank)


def build_station_load_field(
    shell_inputs: ShellMeshInputs,
    *,
    distributed: DistributedStationLoadSpec,
    job_config: SectionLoadsJobConfig,
    station_context: StationAeroContext,
    cache_dir: Path | None = None,
    station_id: str | None = None,
) -> SectionStationLoadField:
    """Build a scaled surface traction field for one structural station."""
    outline_yz = _skin_ring_from_shell_inputs(shell_inputs)
    geometry = SectionOutlineGeometry.from_closed_contour(outline_yz, ensure_clockwise=True)
    incident = derive_incident_aero(job_config, station_context)
    cp, cf, xfoil_artifacts, xfoil_used, synthetic = _run_xfoil_or_synthetic(
        geometry,
        job_config,
        incident,
        twist_rad=float(shell_inputs.twist_rad),
        cache_dir=cache_dir,
    )
    traction, match_rank = _build_traction_field(geometry, distributed, cp, cf)

    sid = station_id if station_id is not None else f"i{station_context.station_index:04d}"
    derivation = "direct_xfoil_scalars" if job_config.use_direct_xfoil_scalars else "tsr_relative_wind_v1"
    provenance = LoadFieldProvenance(
        derivation=derivation,
        xfoil_used=xfoil_used,
        synthetic_aero=synthetic,
        direct_scalar_override=bool(job_config.use_direct_xfoil_scalars),
        load_matcher_rank=match_rank,
        notes=str(xfoil_artifacts.get("mode", "")),
    )

    return SectionStationLoadField(
        station_id=sid,
        z_station=float(station_context.z_station),
        chord_m=float(shell_inputs.chord_m),
        twist_rad=float(shell_inputs.twist_rad),
        outline_yz=outline_yz,
        traction_field=traction,
        distributed_source=distributed,
        incident=incident,
        xfoil_artifacts=xfoil_artifacts,
        provenance=provenance,
    )
