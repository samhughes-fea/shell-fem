"""Derive XFOIL operating scalars from job config and station context."""

from __future__ import annotations

import math

from ..containers.types import SectionLoadsJobConfig
from ..containers.types import IncidentAeroSpec, StationAeroContext
from blade_structure_utilities.load_utilities.beam_to_shell_loads.surface_traction.xfoil.xfoil_adapter import XfoilRunConfig


def derive_incident_aero(
    job_config: SectionLoadsJobConfig,
    station: StationAeroContext,
) -> IncidentAeroSpec:
    """Return ``Re``, ``Ma``, and effective ``alpha`` for one station."""
    r_tip = max(float(station.r_tip_m), 1e-12)
    r_station = float(station.radial_pos_m)
    chord = max(float(station.chord_m), 1e-12)

    if job_config.use_direct_xfoil_scalars:
        re = float(job_config.direct_reynolds_number or 1.0e6)
        ma = float(job_config.direct_mach_number or 0.0)
        alpha = float(job_config.direct_alpha_deg or 0.0)
        u_rel = max(re * job_config.dynamic_viscosity_pa_s / (job_config.rho_kg_m3 * chord), 0.0)
        omega = job_config.u_inf_m_s * job_config.tip_speed_ratio / r_tip
        u_tan = omega * r_station
        return IncidentAeroSpec(
            u_rel_m_s=u_rel,
            reynolds_number=re,
            mach_number=ma,
            alpha_deg=alpha,
            omega_rad_s=omega,
            u_tan_m_s=u_tan,
        )

    omega = job_config.u_inf_m_s * job_config.tip_speed_ratio / r_tip
    u_tan = omega * r_station
    u_inf = float(job_config.u_inf_m_s)
    u_rel = math.sqrt(u_inf * u_inf + u_tan * u_tan)
    rho = float(job_config.rho_kg_m3)
    mu = float(job_config.dynamic_viscosity_pa_s)
    a_sound = max(float(job_config.speed_of_sound_m_s), 1e-12)
    re = rho * u_rel * chord / mu
    ma = u_rel / a_sound
    # Relative-wind velocity triangle: the local inflow angle follows from the
    # axial (u_inf) and tangential (omega*r) components; ``inflow_phi_deg`` is
    # an additional job-level offset on top of the derived angle.
    phi_local_deg = math.degrees(math.atan2(u_inf, u_tan))
    alpha = float(
        job_config.pitch_deg + job_config.inflow_phi_deg + phi_local_deg - station.twist_deg
    )
    return IncidentAeroSpec(
        u_rel_m_s=u_rel,
        reynolds_number=re,
        mach_number=ma,
        alpha_deg=alpha,
        omega_rad_s=omega,
        u_tan_m_s=u_tan,
    )


def build_xfoil_run_config(
    job_config: SectionLoadsJobConfig,
    incident: IncidentAeroSpec,
) -> XfoilRunConfig:
    """Map job + derived incident scalars to an ``XfoilRunConfig``."""
    return XfoilRunConfig(
        reynolds_number=float(incident.reynolds_number),
        mach_number=float(incident.mach_number),
        alpha_deg=float(incident.alpha_deg),
        n_iterations=int(job_config.xfoil_n_iterations),
        timeout_seconds=float(job_config.xfoil_timeout_seconds),
        executable_path=job_config.xfoil_executable,
    )
