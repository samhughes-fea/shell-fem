"""Pre-processing: everything a solve needs, prepared but not solved.

containers/        FEAMesh (nodes, elements, sets, DOF numbering), BoundaryConditions
element_library/   shell, curved shell and layered solid elements; RBE2 / RBE3 connectors
property_library/  ply books (Layup) and material orientation on curved shells
load_library/      LoadCase and LoadCombination
mesh_library/      plate, panel, Gmsh and blade mesh builders, including the kernel shell mesh
model_check        pre-solve model checks
"""
