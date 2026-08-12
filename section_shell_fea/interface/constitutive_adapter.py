"""Shell-owned constitutive builders for beam-facing station handoff."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import BeamSectionConstitutiveInput, ShellStationProvenance


def build_shell_constitutive_input(
    *,
    z_station: float,
    K6: np.ndarray | None = None,
    mass_per_length: float | None = None,
    centres: dict[str, float] | None = None,
    inertia_terms: dict[str, float] | None = None,
    warping_metadata: dict[str, Any] | None = None,
    provenance: ShellStationProvenance | None = None,
) -> BeamSectionConstitutiveInput:
    """Create a validated shell-owned beam constitutive DTO.

    ``K6`` may be omitted (``None``) — Saint-Venant periodic homogenisation is
    deprecated; main-track linear-static station analysis does not supply it.
    """

    k6 = None if K6 is None else np.asarray(K6, dtype=float).copy()
    return BeamSectionConstitutiveInput(
        z_station=float(z_station),
        K6=k6,
        mass_per_length=mass_per_length,
        centres=dict(centres or {}),
        inertia_terms=dict(inertia_terms or {}),
        warping_metadata=dict(warping_metadata or {}),
        provenance=provenance or ShellStationProvenance(),
    )


def build_shell_constitutive_stack(
    *,
    station_z: Sequence[float],
    K6: np.ndarray,
    provenance: ShellStationProvenance | None = None,
    centres_by_station: Sequence[dict[str, float]] | None = None,
    inertia_by_station: Sequence[dict[str, float]] | None = None,
    warping_metadata_by_station: Sequence[dict[str, Any]] | None = None,
    mass_per_length: Sequence[float | None] | None = None,
) -> list[BeamSectionConstitutiveInput]:
    """Build a shell-owned constitutive DTO list from parallel tabulated arrays."""

    z = np.asarray(station_z, dtype=float).ravel()
    k6 = np.asarray(K6, dtype=float)
    if k6.shape != (z.size, 6, 6):
        raise ValueError(f"K6 must have shape (n, 6, 6); got {k6.shape!r}")
    if z.size >= 2 and np.any(np.diff(z) <= 0.0):
        raise ValueError("station_z must be strictly increasing")

    prov = provenance or ShellStationProvenance()
    centres_seq = _normalise_metadata_sequence(centres_by_station, z.size, "centres_by_station")
    inertia_seq = _normalise_metadata_sequence(inertia_by_station, z.size, "inertia_by_station")
    warping_seq = _normalise_metadata_sequence(
        warping_metadata_by_station,
        z.size,
        "warping_metadata_by_station",
    )
    mass_seq = _normalise_mass_sequence(mass_per_length, z.size)
    return [
        build_shell_constitutive_input(
            z_station=float(z[i]),
            K6=k6[i],
            mass_per_length=mass_seq[i],
            centres=centres_seq[i],
            inertia_terms=inertia_seq[i],
            warping_metadata=warping_seq[i],
            provenance=prov,
        )
        for i in range(z.size)
    ]


def build_shell_constitutive_stack_from_legacy_section_properties(
    *,
    station_z: Sequence[float],
    K6: np.ndarray,
    provenance: ShellStationProvenance | None = None,
    section_results: Sequence[Any] | None = None,
) -> list[BeamSectionConstitutiveInput]:
    """Build shell constitutive DTOs from tabulated K6 and optional section property metadata."""

    k6 = np.asarray(K6, dtype=float)
    if k6.ndim != 3 or k6.shape[1:] != (6, 6):
        raise ValueError(f"K6 must have shape (n, 6, 6); got {k6.shape!r}")

    n = int(np.asarray(station_z, dtype=float).ravel().size)
    centres_by_station: list[dict[str, float]] = []
    inertia_by_station: list[dict[str, float]] = []
    mass_per_length: list[float | None] = []
    warping_metadata_by_station: list[dict[str, Any]] = []

    results = list(section_results) if section_results is not None else [None] * n
    if len(results) != n:
        raise ValueError("section_results length must match number of stations")

    for res in results:
        centres: dict[str, float] = {}
        inertia: dict[str, float] = {}
        warping: dict[str, Any] = {}
        mu_line: float | None = None
        if res is not None:
            y_ec = _maybe_float(getattr(res, "y_ec", None))
            z_ec = _maybe_float(getattr(res, "z_ec", None))
            y_e = _maybe_float(getattr(res, "y_e", None))
            z_e = _maybe_float(getattr(res, "z_e", None))
            y_sc = _maybe_float(getattr(res, "y_sc", None))
            z_sc = _maybe_float(getattr(res, "z_sc", None))
            if y_ec is not None and z_ec is not None:
                centres["elastic_center_y"] = y_ec
                centres["elastic_center_z"] = z_ec
            elif y_e is not None and z_e is not None:
                centres["elastic_center_y"] = y_e
                centres["elastic_center_z"] = z_e
            elif y_sc is not None and z_sc is not None:
                # Deprecated alias of elastic centroid (not shear centre).
                centres["elastic_center_y"] = y_sc
                centres["elastic_center_z"] = z_sc
            # Prefer an explicit shear-centre attribute when present.
            y_s = _maybe_float(getattr(res, "y_shear", None))
            z_s = _maybe_float(getattr(res, "z_shear", None))
            if y_s is not None and z_s is not None:
                centres["shear_center_y"] = y_s
                centres["shear_center_z"] = z_s
            mu_line = _maybe_float(getattr(res, "mass_per_length", None))
            for key in ("Ixx", "Iyy", "Ixy", "J", "A"):
                value = _maybe_float(getattr(res, key, None))
                if value is not None:
                    inertia[key] = value
            for key in ("I_omega", "gamma_y", "gamma_z", "GA_y", "GA_z"):
                value = _maybe_float(getattr(res, key, None))
                if value is not None:
                    warping[key] = value
        centres_by_station.append(centres)
        inertia_by_station.append(inertia)
        mass_per_length.append(mu_line)
        warping_metadata_by_station.append({**warping, "source": "legacy_section_properties"})

    return build_shell_constitutive_stack(
        station_z=station_z,
        K6=K6,
        provenance=provenance,
        centres_by_station=centres_by_station,
        inertia_by_station=inertia_by_station,
        warping_metadata_by_station=warping_metadata_by_station,
        mass_per_length=mass_per_length,
    )


def _normalise_metadata_sequence(
    values: Sequence[dict[str, float]] | Sequence[dict[str, Any]] | None,
    n: int,
    name: str,
) -> list[dict[str, Any]]:
    if values is None:
        return [{} for _ in range(n)]
    seq = [dict(item) for item in values]
    if len(seq) != n:
        raise ValueError(f"{name} length must match number of stations ({n})")
    return seq


def _normalise_mass_sequence(values: Sequence[float | None] | None, n: int) -> list[float | None]:
    if values is None:
        return [None] * n
    seq = list(values)
    if len(seq) != n:
        raise ValueError(f"mass_per_length length must match number of stations ({n})")
    return [None if value is None else float(value) for value in seq]


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


__all__ = [
    "build_shell_constitutive_input",
    "build_shell_constitutive_stack",
    "build_shell_constitutive_stack_from_legacy_section_properties",
]
