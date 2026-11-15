"""Linear shell buckling under a reference static load.

``operations`` is the geometric stiffness and positive-factor eigenproblem that
``linear_buckling(mesh, bcs, load_case, n_modes=1)`` calls. ``LinearBucklingRunner``
uses ``geometric``, ``eigen`` and ``results`` (dense or Lanczos, positive factors or
both signs) and does not replace that call.

The load factors multiply that reference load. A rotor is not assembled here.
"""

from shell_fem.processing.buckling.operations import AssembleGeometricStiffness, SolveBucklingEigenproblem

__all__ = ["AssembleGeometricStiffness", "SolveBucklingEigenproblem"]
