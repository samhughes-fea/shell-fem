"""Simulation runners: each sequences the processing stages of one analysis and owns its results.

static/              LinearStaticRunner, StructuralGlobalSolver, solve_kernel_pcomp
buckling/            LinearBucklingRunner
progressive_damage/  ProgressiveDamageAnalysis
backends/            native, CalculiX and Nastran solver kernels
"""
