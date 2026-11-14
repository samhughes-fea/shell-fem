"""Simulation runners: each sequences the processing stages of one analysis and owns its results.

static/              LinearStaticRunner, StructuralGlobalSolver, solve_kernel_pcomp
buckling/            linear_buckling
vibration/           vibration_modes (the blade; a rotor is not assembled)
progressive_damage/  ProgressiveDamageAnalysis
backends/            native, CalculiX and Nastran solver kernels
"""
