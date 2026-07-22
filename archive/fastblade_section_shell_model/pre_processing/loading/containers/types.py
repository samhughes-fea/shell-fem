"""DTOs for per-station surface traction load fields."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from blade_utilities.point_to_surftraction.tractions import TractionField


@dataclass(frozen=True)
class StationAeroContext:
    """Per-station geometry context for incident-aero derivation."""

    station_index: int
    z_station: float
    radial_pos_m: float
    chord_m: float
    twist_deg: float
    r_tip_m: float


@dataclass(frozen=True)
class IncidentAeroSpec:
    """Resolved operating point passed to XFOIL (or synthetic fallback)."""

    u_rel_m_s: float
    reynolds_number: float
    mach_number: float
    alpha_deg: float
    omega_rad_s: float
    u_tan_m_s: float


@dataclass(frozen=True)
class DistributedStationLoadSpec:
    """Authoritative distributed line loads at one structural station."""

    z_station: float
    q_y_Npm: float
    q_z_Npm: float
    m_x_Nmpm: float


@dataclass(frozen=True)
class LoadFieldProvenance:
    """Audit trail for how a station load field was produced."""

    derivation: str
    xfoil_used: bool
    synthetic_aero: bool
    direct_scalar_override: bool
    load_matcher_rank: int | None = None
    notes: str = ""


@dataclass(frozen=True)
class SectionLoadsJobConfig:
    """Job-level aero / XFOIL settings for the ``section_loads`` stage."""

    u_inf_m_s: float
    tip_speed_ratio: float
    rho_kg_m3: float = 1.225
    dynamic_viscosity_pa_s: float = 1.81e-5
    speed_of_sound_m_s: float = 340.0
    pitch_deg: float = 0.0
    inflow_phi_deg: float = 0.0
    xfoil_executable: str | None = None
    xfoil_n_iterations: int = 100
    xfoil_timeout_seconds: float = 90.0
    use_direct_xfoil_scalars: bool = False
    direct_reynolds_number: float | None = None
    direct_mach_number: float | None = None
    direct_alpha_deg: float | None = None


@dataclass(frozen=True)
class SectionStationLoadField:
    """Handoff contract from ``section_loads`` to ``section_shell_model``."""

    station_id: str
    z_station: float
    chord_m: float
    twist_rad: float
    outline_yz: NDArray[np.float64]
    traction_field: TractionField
    distributed_source: DistributedStationLoadSpec
    incident: IncidentAeroSpec
    xfoil_artifacts: Mapping[str, Any] = field(default_factory=dict)
    provenance: LoadFieldProvenance = field(
        default_factory=lambda: LoadFieldProvenance(
            derivation="unknown",
            xfoil_used=False,
            synthetic_aero=False,
            direct_scalar_override=False,
        )
    )

    def to_json_dict(self) -> dict[str, Any]:
        tf = self.traction_field
        return {
            "schema_version": "section_load_field_v1",
            "station_id": str(self.station_id),
            "z_station": float(self.z_station),
            "chord_m": float(self.chord_m),
            "twist_rad": float(self.twist_rad),
            "outline_yz": np.asarray(self.outline_yz, dtype=float).tolist(),
            "distributed_source": {
                "z_station": float(self.distributed_source.z_station),
                "q_y_Npm": float(self.distributed_source.q_y_Npm),
                "q_z_Npm": float(self.distributed_source.q_z_Npm),
                "m_x_Nmpm": float(self.distributed_source.m_x_Nmpm),
            },
            "incident": {
                "u_rel_m_s": float(self.incident.u_rel_m_s),
                "reynolds_number": float(self.incident.reynolds_number),
                "mach_number": float(self.incident.mach_number),
                "alpha_deg": float(self.incident.alpha_deg),
                "omega_rad_s": float(self.incident.omega_rad_s),
                "u_tan_m_s": float(self.incident.u_tan_m_s),
            },
            "provenance": {
                "derivation": str(self.provenance.derivation),
                "xfoil_used": bool(self.provenance.xfoil_used),
                "synthetic_aero": bool(self.provenance.synthetic_aero),
                "direct_scalar_override": bool(self.provenance.direct_scalar_override),
                "load_matcher_rank": self.provenance.load_matcher_rank,
                "notes": str(self.provenance.notes),
            },
            "xfoil_artifacts": dict(self.xfoil_artifacts),
            "traction_field": {
                "pressure_forces": np.asarray(tf.pressure_forces, dtype=float).tolist(),
                "shear_forces": np.asarray(tf.shear_forces, dtype=float).tolist(),
                "total_forces": np.asarray(tf.total_forces, dtype=float).tolist(),
            },
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> SectionStationLoadField:
        schema = payload.get("schema_version", payload.get("schema"))
        if schema != "section_load_field_v1":
            raise ValueError(f"unsupported section load field schema: {schema!r}")
        ds = payload["distributed_source"]
        inc = payload["incident"]
        prov = payload["provenance"]
        tf = payload["traction_field"]
        return cls(
            station_id=str(payload["station_id"]),
            z_station=float(payload["z_station"]),
            chord_m=float(payload["chord_m"]),
            twist_rad=float(payload["twist_rad"]),
            outline_yz=np.asarray(payload["outline_yz"], dtype=np.float64),
            traction_field=TractionField(
                pressure_forces=np.asarray(tf["pressure_forces"], dtype=np.float64),
                shear_forces=np.asarray(tf["shear_forces"], dtype=np.float64),
                total_forces=np.asarray(tf["total_forces"], dtype=np.float64),
            ),
            distributed_source=DistributedStationLoadSpec(
                z_station=float(ds["z_station"]),
                q_y_Npm=float(ds["q_y_Npm"]),
                q_z_Npm=float(ds["q_z_Npm"]),
                m_x_Nmpm=float(ds["m_x_Nmpm"]),
            ),
            incident=IncidentAeroSpec(
                u_rel_m_s=float(inc["u_rel_m_s"]),
                reynolds_number=float(inc["reynolds_number"]),
                mach_number=float(inc["mach_number"]),
                alpha_deg=float(inc["alpha_deg"]),
                omega_rad_s=float(inc["omega_rad_s"]),
                u_tan_m_s=float(inc["u_tan_m_s"]),
            ),
            xfoil_artifacts=dict(payload.get("xfoil_artifacts") or {}),
            provenance=LoadFieldProvenance(
                derivation=str(prov["derivation"]),
                xfoil_used=bool(prov["xfoil_used"]),
                synthetic_aero=bool(prov["synthetic_aero"]),
                direct_scalar_override=bool(prov["direct_scalar_override"]),
                load_matcher_rank=prov.get("load_matcher_rank"),
                notes=str(prov.get("notes", "")),
            ),
        )
