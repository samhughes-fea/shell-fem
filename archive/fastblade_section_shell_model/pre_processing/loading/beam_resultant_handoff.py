"""Resolve :class:`BeamResultantVector` (or legacy scalar kwargs) into shell K6 RHS.

Transport about the elastic centroid uses the full equipollent map on the
vector contract; projection uses :data:`SHELL_K6_MODE_ORDER`.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray

from blade_precompute.contract.beam_resultant_vector import (
    FRAME_SECTION_YZ,
    BeamResultantVector,
)


def resolve_beam_resultants_for_shell_k6(
    *,
    y_e: float,
    z_e: float,
    resultants: BeamResultantVector | None = None,
    N: float = 0.0,
    Vy: float = 0.0,
    Vz: float = 0.0,
    My: float = 0.0,
    Mz: float = 0.0,
    T: float = 0.0,
    loading_axis_yz: tuple[float, float] | None = None,
    transport_to_elastic_centroid: bool = True,
) -> tuple[BeamResultantVector, NDArray[np.float64]]:
    """Return (vector about elastic centroid, ``f_beam`` in shell K6 order).

    Prefer ``resultants``. Scalar kwargs build a shim vector referenced at the
    elastic centroid (bit-for-bit with historical no-offset behaviour). Passing
    ``loading_axis_yz`` without ``resultants`` is deprecated — set
    ``resultants.about_yz`` instead.

    When ``transport_to_elastic_centroid`` is False, ``about_yz`` is retagged to
    ``(y_e, z_e)`` without the ``r×F`` map (legacy Channel-A ``T_ec = T``).
    """
    ye, ze = float(y_e), float(z_e)

    def _to_ec(vec: BeamResultantVector) -> BeamResultantVector:
        if transport_to_elastic_centroid:
            return vec.transported_to(ye, ze)
        return vec.with_about_yz(ye, ze)

    if resultants is not None:
        if loading_axis_yz is not None:
            warnings.warn(
                "loading_axis_yz is ignored when resultants=BeamResultantVector is "
                "supplied; use resultants.about_yz / transported_to instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        at_ec = _to_ec(resultants)
        return at_ec, at_ec.to_shell_k6_conjugate()

    if loading_axis_yz is not None:
        warnings.warn(
            "loading_axis_yz is deprecated; pass resultants=BeamResultantVector("
            "..., about_yz=loading_axis) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        vec = BeamResultantVector.from_named(
            N=N, Vy=Vy, Vz=Vz, My=My, Mz=Mz, T=T,
            frame=FRAME_SECTION_YZ,
            about_yz=(float(loading_axis_yz[0]), float(loading_axis_yz[1])),
        )
        at_ec = _to_ec(vec)
        return at_ec, at_ec.to_shell_k6_conjugate()

    # Scalar shim: already about the elastic centroid → no transport.
    vec = BeamResultantVector.from_named(
        N=N, Vy=Vy, Vz=Vz, My=My, Mz=Mz, T=T,
        frame=FRAME_SECTION_YZ,
        about_yz=(ye, ze),
    )
    return vec, vec.to_shell_k6_conjugate()


def beam_resultant_vector_from_any(
    loads: BeamResultantVector | dict[str, Any] | None,
    *,
    about_yz: tuple[float, float] = (0.0, 0.0),
    frame: str = FRAME_SECTION_YZ,
) -> BeamResultantVector | None:
    """Coerce a loads dict or vector; ``None`` stays ``None``."""
    if loads is None:
        return None
    if isinstance(loads, BeamResultantVector):
        return loads
    return BeamResultantVector.from_mapping(
        {str(k): float(v) for k, v in loads.items()},
        frame=frame,
        about_yz=about_yz,
    )


__all__ = [
    "resolve_beam_resultants_for_shell_k6",
    "beam_resultant_vector_from_any",
]
