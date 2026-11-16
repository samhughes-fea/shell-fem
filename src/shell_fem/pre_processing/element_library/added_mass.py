"""Added mass in water as a labelled strip-heave model on shells.

This is a potential-flow model, not a measured frequency in water. No still-water
frequency, hydrodynamic coefficient, or empirical factor is implied or fitted.

Labelled model
    ``strip_heave``: two-dimensional unbounded-fluid added mass of a flat plate
    (thin-airfoil heave), applied as a consistent mass on the local-normal
    translation only. In-plane translations, rotations, and enrichment amplitudes
    carry none. Layered solids are skipped: strip theory is a surface model.

Formula
    Per unit span, ``m_a' = rho_fluid * pi * (chord / 2)**2``.
    Spread over the chord as an areal density on the shell,
    ``rho_A = m_a' / chord = rho_fluid * pi * chord / 4``.

Source
    Newman, J. N. (1977). *Marine Hydrodynamics*. MIT Press, Cambridge, MA.
    Two-dimensional added mass of a flat plate of chord ``c = 2b`` for motion
    normal to the plate is ``rho * pi * b**2`` per unit span (the circumscribed
    cylinder). Theodorsen, T. (1935). *General Theory of Aerodynamic Instability
    and the Mechanism of Flutter*, NACA Report 496: the apparent mass of a thin
    airfoil of semi-chord ``b`` is ``rho * pi * b**2`` per unit span.

The default is off: ``vibration_modes`` with no ``added_mass`` leaves the
structural mass unchanged. ``rho_fluid`` and ``chord`` are supplied by the
caller; this module does not choose seawater density or a blade chord.

Public API
----------
STRIP_HEAVE_FORMULA, STRIP_HEAVE_SOURCE, StripHeaveAddedMass,
strip_heave_added_mass_per_length, strip_heave_areal_density,
element_added_mass_matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shell_fem.pre_processing.element_library.shell import ShellElement

STRIP_HEAVE_MODEL = "strip_heave"

STRIP_HEAVE_FORMULA = "rho_fluid * pi * (chord/2)**2"

STRIP_HEAVE_SOURCE = (
    "Newman, J. N. (1977). Marine Hydrodynamics. MIT Press. "
    "Two-dimensional added mass of a flat plate of chord c = 2b for motion "
    "normal to the plate is rho * pi * b**2 per unit span. "
    "Theodorsen, T. (1935). NACA Report 496: apparent mass of a thin airfoil "
    "of semi-chord b is rho * pi * b**2 per unit span."
)


def strip_heave_added_mass_per_length(rho_fluid: float, chord: float) -> float:
    """Heave added mass per unit span of a 2-D flat plate of chord ``chord``.

    ``m_a' = rho_fluid * pi * (chord / 2)**2``. ``rho_fluid`` and ``chord`` are
    supplied by the caller; this function does not choose seawater density or a
    blade chord.
    """
    rho = float(rho_fluid)
    c = float(chord)
    if rho < 0.0:
        raise ValueError(f"rho_fluid must be >= 0, got {rho_fluid!r}")
    if c < 0.0:
        raise ValueError(f"chord must be >= 0, got {chord!r}")
    return rho * np.pi * (c / 2.0) ** 2


def strip_heave_areal_density(rho_fluid: float, chord: float) -> float:
    """Strip-heave added mass spread over the chord as a shell areal density.

    ``rho_A = m_a' / chord = rho_fluid * pi * chord / 4``. A rectangular panel
    of width ``chord`` and span ``L`` then carries total normal added mass
    ``m_a' * L``, matching the beam/strip formula.
    """
    c = float(chord)
    if c == 0.0:
        strip_heave_added_mass_per_length(rho_fluid, c)
        return 0.0
    return strip_heave_added_mass_per_length(rho_fluid, c) / c


@dataclass(frozen=True)
class StripHeaveAddedMass:
    """Labelled strip-heave added mass: fluid density and chord.

    ``mass_per_length`` is always the Newman/Theodorsen plate formula from
    ``rho_fluid`` and ``chord``. It is not a measured still-water frequency.
    """

    rho_fluid: float
    chord: float
    model: str = STRIP_HEAVE_MODEL

    def __post_init__(self):
        model = str(self.model).strip().lower()
        if model != STRIP_HEAVE_MODEL:
            raise ValueError(
                f"added-mass model must be {STRIP_HEAVE_MODEL!r} (the labelled strip-heave "
                f"plate formula); got {self.model!r}"
            )
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "rho_fluid", float(self.rho_fluid))
        object.__setattr__(self, "chord", float(self.chord))
        if self.rho_fluid < 0.0:
            raise ValueError(f"rho_fluid must be >= 0, got {self.rho_fluid!r}")
        if self.chord < 0.0:
            raise ValueError(f"chord must be >= 0, got {self.chord!r}")

    @property
    def mass_per_length(self) -> float:
        return strip_heave_added_mass_per_length(self.rho_fluid, self.chord)

    @property
    def areal_density(self) -> float:
        return strip_heave_areal_density(self.rho_fluid, self.chord)


def element_added_mass_matrix(element: ShellElement, model: StripHeaveAddedMass) -> np.ndarray:
    """Consistent ``M_a`` on local-normal translation from the labelled strip-heave density.

    ``C_a[2, 2] = rho_A`` (local ``w`` only). ``M_a = T.T @ (∫ N^T C_a N dA) @ T``.
    """
    d = element.dofs_per_node
    C = np.zeros((d, d))
    C[2, 2] = model.areal_density
    M = np.zeros((element.n_dof, element.n_dof))
    for xi, eta, w in element.stiffness_rule():
        N, _ = element.shape_functions(xi, eta)
        _, detJ, _ = element.jacobian(xi, eta)
        M += np.kron(np.outer(N, N), C) * (w * detJ)
    M = 0.5 * (M + M.T)
    T = element.transformation
    return T.T @ M @ T
