"""Diagnostics on computed results: discretisation error estimates.

gci.py   ASME grid convergence index (Celik et al. 2008) from three or more meshes
"""

from shell_fem.processing.diagnostics.gci import GCIResult, grid_convergence_index, mesh_size

__all__ = ["GCIResult", "grid_convergence_index", "mesh_size"]
