"""Processing: the computations, one folder per analysis type.

static/              solve stages (operations/) and result recovery (results/)
buckling/            geometric stiffness, buckling eigenproblem and modes (one set of stages)
nonlinear/           geometric nonlinearity scope (Newton / follower pressure: not implemented)
vibration/           shell natural frequencies and mode shapes
progressive_damage/  ply damage states and stiffness degradation
diagnostics/         discretisation error: the ASME grid convergence index
stage_log.py         wall-clock reports on the blade.stage logger
"""
