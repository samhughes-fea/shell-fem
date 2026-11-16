"""Accept either an FEAMesh or a blade mesh that holds one."""

from __future__ import annotations

from shell_fem.pre_processing.containers.mesh import FEAMesh


def as_shell_mesh(model) -> FEAMesh:
    """The mesh a vibration or buckling solve runs on.

    ``model`` is an ``FEAMesh``, or a blade mesh whose ``mesh`` attribute is one.
    Layered solids contribute mass and geometric stiffness, so a local solid
    patch at a ply drop or bond is included.
    """
    mesh = model
    if not hasattr(model, "shell_elements") and hasattr(model, "mesh"):
        mesh = model.mesh
    if not isinstance(mesh, FEAMesh):
        raise TypeError("expected an FEAMesh, or a blade mesh with a .mesh attribute")
    if not mesh.elements:
        raise ValueError("the mesh has no elements")
    return mesh
