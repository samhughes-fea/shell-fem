"""Map station load fields to exclusive beam vs face-traction shell loads.

Production path: shape resultants / traction via the helpers below, then
:class:`~blade_structure_utilities.fem_utilities.section_shell_model.simulation_runner.section_linear_static_runner.SectionLinearStaticRunner`.
"""
from __future__ import annotations

import warnings
from typing import Any, Mapping

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.loading.containers.types import (
    SectionStationLoadField,
)


def traction_points_from_load_field(load_field: SectionStationLoadField) -> np.ndarray | None:
    """Panel-centre ``(n_panel, 2)`` locations aligned with ``traction_field.total_forces``."""
    tf = load_field.traction_field
    outline = np.asarray(load_field.outline_yz, dtype=float)
    if tf is None or outline.ndim != 2 or outline.shape[0] < 2:
        return None
    centres = 0.5 * (outline[:-1] + outline[1:])
    if np.asarray(tf.total_forces).shape[0] != centres.shape[0]:
        return None
    return centres


def beam_resultants_from_load_field(
    load_field: SectionStationLoadField,
    *,
    prefer_traction_integration: bool = False,
) -> tuple[float, float, float, float, float, float]:
    """Return ``(N, Vy, Vz, My, Mz, T)`` from the distributed load source or traction integration."""
    return section_resultants_from_load_field(
        load_field,
        prefer_traction_integration=prefer_traction_integration,
    )


def section_resultants_from_load_field(
    load_field: SectionStationLoadField,
    *,
    prefer_traction_integration: bool = False,
) -> tuple[float, float, float, float, float, float]:
    """Return ``(N, Vy, Vz, My, Mz, T)`` for Channel A of a load-field handoff.

    Load-provenance policy (mutually exclusive channels):

    - When the load field has a usable ``TractionField`` (panel centres align with
      ``total_forces``), Channel B owns strip force: ``Vy = Vz = 0``. Distributed
      torque ``m_x`` remains on Channel A as ``T`` (LoadMatcher never matches torque).
    - When there is no usable traction field, Channel A carries distributed intensities
      as shears (``Vy = q_y``, ``Vz = q_z``) so the load is applied once.

    ``prefer_traction_integration`` only applies on the no-traction path: it replaces
    ``Vy``/``Vz`` with ``traction_field.get_total_force()`` when a field exists but is
    not panel-aligned for overlay.
    """
    ds = load_field.distributed_source
    T = float(ds.m_x_Nmpm)
    N = 0.0
    My = 0.0
    Mz = 0.0

    if traction_points_from_load_field(load_field) is not None:
        # Traction owns strip force — never also put the same q into Vy/Vz.
        return (N, 0.0, 0.0, My, Mz, T)

    Vy = float(ds.q_y_Npm)
    Vz = float(ds.q_z_Npm)
    if prefer_traction_integration and load_field.traction_field is not None:
        total = load_field.traction_field.get_total_force()
        Vy = float(total[0])
        Vz = float(total[1])

    return (N, Vy, Vz, My, Mz, T)


def resolve_shell_station_resultants(
    station_index: int,
    *,
    station_resultants: Mapping[int, tuple[float, float, float, float, float, float]] | None,
    station_load_fields: Mapping[int, SectionStationLoadField] | None,
    prefer_traction_integration: bool = False,
) -> tuple[float, float, float, float, float, float]:
    """Resolve per-station Channel A loads; explicit ``station_resultants`` take precedence.

    When both ``station_resultants`` and ``station_load_fields`` are supplied for the
    same station, a ``RuntimeWarning`` is emitted and the tip-integrated / explicit
    resultants win (production orchestration should pass only one source). Callers
    that forward a load field into dispatch must also call
    :func:`load_field_for_dispatch` so the losing load field is not also applied.
    """
    has_resultants = (
        station_resultants is not None and station_index in station_resultants
    )
    has_fields = (
        station_load_fields is not None and station_index in station_load_fields
    )
    if has_resultants and has_fields:
        warnings.warn(
            f"resolve_shell_station_resultants: station {station_index} has both "
            "station_resultants and station_load_fields; using station_resultants "
            "(callers must pick one Channel A source).",
            RuntimeWarning,
            stacklevel=2,
        )
    if has_resultants:
        sr = station_resultants[station_index]
        return (
            float(sr[0]),
            float(sr[1]),
            float(sr[2]),
            float(sr[3]),
            float(sr[4]),
            float(sr[5]),
        )
    if has_fields:
        return beam_resultants_from_load_field(
            station_load_fields[station_index],
            prefer_traction_integration=prefer_traction_integration,
        )
    raise ValueError(
        f"resolve_shell_station_resultants: no load data for station {station_index}; "
        "supply station_resultants or station_load_fields "
        "(unit placeholder resultants are not allowed on production paths)."
    )


def load_field_for_dispatch(
    station_index: int,
    *,
    station_resultants: Mapping[int, tuple[float, float, float, float, float, float]] | None,
    station_load_fields: Mapping[int, SectionStationLoadField] | None,
) -> SectionStationLoadField | None:
    """Return the load field to forward into dispatch, or ``None`` if suppressed.

    When explicit ``station_resultants`` win over a simultaneous load field, the
    load field must **not** be forwarded — otherwise Channel B would double-apply
    the same strip force already present in tip-integrated ``Vy``/``Vz``.
    """
    if station_load_fields is None or station_index not in station_load_fields:
        return None
    if station_resultants is not None and station_index in station_resultants:
        return None
    return station_load_fields[station_index]


__all__ = [
    "beam_resultants_from_load_field",
    "load_field_for_dispatch",
    "resolve_shell_station_resultants",
    "section_resultants_from_load_field",
    "traction_points_from_load_field",
]
