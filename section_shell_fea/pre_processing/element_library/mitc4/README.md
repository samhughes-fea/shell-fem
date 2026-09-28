# MITC4 Reissner–Mindlin formulation contract

This directory implements a four-node flat-shell finite element whose continuum
contract is the small-strain Reissner–Mindlin (first-order shear-deformation)
plate theory below. The discrete formulation ID is `MITC4-FSDT`. Registration,
canonical projection, and capability tags live in
`pre_processing/element_library/registry.py` and
[`docs/FORMULATION_CONTRACT.md`](../../../docs/FORMULATION_CONTRACT.md).
The notation and signs in this document are normative for
future work. They follow the project theory reference,
[Reissner–Mindlin Plate Theory (FSDT)](https://app.notion.com/p/55c8c922b3c446608139c89cc5e4f677).

## Canonical continuum fields and axes

On each flat facet, \(x_1=x\) is spanwise, \(x_2=s\) is contour-tangential,
and \(z=x_3=n\) is through-thickness. The canonical primary-field and nodal
orders are

\[
\mathbf d=[u^0_1,u^0_2,w,\theta_1,\theta_2]^{\mathsf T},
\qquad
\mathbf d_e=[\mathbf d_1,\mathbf d_2,\mathbf d_3,\mathbf d_4]^{\mathsf T}.
\]

Translations \(u^0_1,u^0_2,w\) have units of length and rotations
\(\theta_1,\theta_2\) are dimensionless. The assumed displacement is

\[
u_1=u^0_1-z\theta_1,\qquad
u_2=u^0_2-z\theta_2,\qquad
u_3=w.
\]

The canonical generalized strains, including engineering shear components,
are ordered and signed as

\[
\boldsymbol\varepsilon^0=
\begin{bmatrix}
u^0_{1,1}\\u^0_{2,2}\\u^0_{1,2}+u^0_{2,1}
\end{bmatrix},
\quad
\boldsymbol\kappa=-
\begin{bmatrix}
\theta_{1,1}\\\theta_{2,2}\\\theta_{1,2}+\theta_{2,1}
\end{bmatrix},
\quad
\boldsymbol\gamma_s=
\begin{bmatrix}
w_{,1}-\theta_1\\w_{,2}-\theta_2
\end{bmatrix}.
\]

\(\boldsymbol\varepsilon^0\) and \(\boldsymbol\gamma_s\) are dimensionless;
\(\boldsymbol\kappa\) has units \(L^{-1}\). The Kirchhoff constraint is
\(\boldsymbol\theta=\nabla w\).

## Resultants and constitutive order

The exact canonical orders are

\[
\mathbf N=[N_{11},N_{22},N_{12}]^{\mathsf T},\quad
\mathbf M=[M_{11},M_{22},M_{12}]^{\mathsf T},\quad
\mathbf Q=[Q_1,Q_2]^{\mathsf T}.
\]

They are defined by

\[
\mathbf N=\int\boldsymbol\sigma_p\,dz,\qquad
\mathbf M=\int z\boldsymbol\sigma_p\,dz,\qquad
\mathbf Q=\int\boldsymbol\tau_s\,dz,
\]

and obey

\[
\begin{bmatrix}\mathbf N\\\mathbf M\\\mathbf Q\end{bmatrix}
=
\begin{bmatrix}
\mathbf A&\mathbf B&\mathbf0\\
\mathbf B^{\mathsf T}&\mathbf D&\mathbf0\\
\mathbf0&\mathbf0&\mathbf A_s
\end{bmatrix}
\begin{bmatrix}
\boldsymbol\varepsilon^0\\\boldsymbol\kappa\\\boldsymbol\gamma_s
\end{bmatrix}
-
\begin{bmatrix}\mathbf N^{T,H}\\\mathbf M^{T,H}\\\mathbf0\end{bmatrix}.
\]

For the standard laminate convention \(\mathbf B=\mathbf B^{\mathsf T}\).
\(\mathbf A\) and \(\mathbf A_s\) have units force/length, \(\mathbf B\) has
units force, and \(\mathbf D\) has units force-length. Thus \(\mathbf N\) and
\(\mathbf Q\) have units force/length and \(\mathbf M\) has units force. The
shear correction is included once, at stiffness level, in \(\mathbf A_s\).

## Weak form, equilibrium, and boundary pairs

For every admissible virtual field,

\[
\delta W_{\mathrm{int}}=
\int_A\left(
\delta\boldsymbol\varepsilon^{0\mathsf T}\mathbf N+
\delta\boldsymbol\kappa^{\mathsf T}\mathbf M+
\delta\boldsymbol\gamma_s^{\mathsf T}\mathbf Q
\right)dA
=\delta W_{\mathrm{ext}}.
\]

With positive transverse load \(q\) in positive \(z\), no in-plane body force,
and outward edge normal \(\mathbf n=[n_1,n_2]^{\mathsf T}\), the strong form is

\[
\nabla\!\cdot\mathbf N=\mathbf0,\qquad
\nabla\!\cdot\mathbf M-\mathbf Q=\mathbf0,\qquad
\nabla\!\cdot\mathbf Q+q=0.
\]

The work-conjugate essential/natural pairs are

| Essential | Natural |
|---|---|
| \(\mathbf u_0=[u^0_1,u^0_2]^{\mathsf T}\) | \(\mathbf t_N=[N_{11}n_1+N_{12}n_2,\;N_{12}n_1+N_{22}n_2]^{\mathsf T}\) |
| \(\boldsymbol\theta=[\theta_1,\theta_2]^{\mathsf T}\) | \(\mathbf m_n=-[M_{11}n_1+M_{12}n_2,\;M_{12}n_1+M_{22}n_2]^{\mathsf T}\) |
| \(w\) | \(Q_n=Q_1n_1+Q_2n_2\) |

The minus sign in \(\mathbf m_n\) follows from the canonical negative curvature
definition. Prescribe one member of each pair on a boundary segment.

## Finite-element enhancements

The continuum has five physical fields and requires only \(H^1/C^0\)
regularity. The following are discretisation choices, not additions to the
Reissner–Mindlin theory:

- **MITC4** replaces the naive bilinear transverse-shear interpolation with
  mixed interpolation of tensorial components at tying points. It is a
  shear-locking remedy and must preserve rigid motion, constant membrane and
  bending states, admissible shear states, and the locking-free thin limit.
- **Drilling stabilisation** adds the sixth nodal variable \(\beta_z\) and a
  Hughes–Brezzi penalty on
  \(u^0_{2,1}-u^0_{1,2}-\beta_z\). It regularises an otherwise unsupported
  in-plane rotation. It is not a sixth Reissner–Mindlin field and must not
  alter the five-field continuum contract. This code calls the six-DOF form
  `MITC4+`; `include_drilling=False` is the five-DOF MITC4 form.

## Legacy adapter and completed sign migration

The current public DOF order is

```text
[u_x, u_s, w, beta_s, beta_x, (beta_z)]
```

and maps to canonical rotations as

\[
\boxed{\theta_1=\beta_s,\qquad\theta_2=-\beta_x}.
\]

The shear rows are canonical:
\(\gamma_{13}=w_{,1}-\beta_s\) and
\(\gamma_{23}=w_{,2}+\beta_x\). The superseded bending operator returned

\[
\boldsymbol\kappa_{\mathrm{legacy}}=
\begin{bmatrix}
\beta_{s,1}\\-\beta_{x,2}\\\beta_{s,2}-\beta_{x,1}
\end{bmatrix}
=-\boldsymbol\kappa_{\mathrm{canonical}}.
\]

The production `StrainDisplacementOperator` now returns
\(\boldsymbol\kappa_{\mathrm{canonical}}\). The corresponding fixture changes
are intentional and recorded in `test_mitc4_wp0_baseline.py`.
Unsymmetric laminates (\(\mathbf B\ne0\)) are especially migration-sensitive
because the curvature sign affects membrane force through
\(\mathbf B\boldsymbol\kappa\), while membrane strain affects moment through
\(\mathbf B^{\mathsf T}\boldsymbol\varepsilon^0\).

## Admissibility and limits

- The \(ABD\) block and \(\mathbf A_s\) must be symmetric positive
  (semi-)definite on non-rigid admissible strains.
- A symmetric laminate has \(\mathbf B=\mathbf0\), so linear membrane and
  bending/shear response decouple absent recoupling by loads or boundaries.
- As \(\mathbf A_s\to\infty\), finite energy requires
  \(\boldsymbol\gamma_s\to\mathbf0\), recovering the Kirchhoff/CLPT constraint.
- In the thin limit, shear deformation vanishes; MITC interpolation must avoid
  discrete shear locking.
- Finite \(\mathbf A_s\) supplies moderately thick plates with shear
  compliance.
- Constant through-thickness FSDT shear strain does not satisfy free-face
  shear stress pointwise. Detailed transverse shear requires equilibrium-based
  recovery, layerwise theory, or 3-D analysis.

## Code and verification traceability

| Continuum statement | Production symbol | Verification |
|---|---|---|
| Bilinear \(\phi_I\), \(\mathbf d=\Phi_u\mathbf d_e\) | `ShapeFunctionOperator` | interpolation and linear-completeness tests |
| Flat-Q4 Jacobian and gradients | `FlatQuad4Geometry` | partition, distortion, folded-Q4 tests |
| \(\varepsilon^0,\kappa,\gamma_s\) | `StrainDisplacementOperator` | rigid/constant patch and Kirchhoff-constraint tests |
| Four edge-midpoint tying conditions | `mitc_shear` | distorted-Q4 admissible-shear patch |
| \(D_g=C_{ABDAs}\) | `MaterialStiffnessOperator` | ABDAs block-placement tests |
| \(K_e=\int B_g^TD_gB_g\,dA\) | `assemble_mitc4_stiffness` | contribution sum, symmetry, energy tests |
| \(r_g=D_gB_gd_e\) | `mitc4_generalized_resultants_at` | exact \(Q=A_s\gamma_s\) test |
| Area/edge virtual work | `ElementLoadOperator` | force, moment, and virtual-work tests |
| Thin/thick plate limits | `validation.fsdt_navier` | independent 3×3 modal Navier tests |
| Interface shear screening | `interface_shear_screening_from_tau` | frame, strength, provenance tests |

The operator class names are stable API surfaces. `N`, `M`, and `Q` refer only
to physical resultants; interpolation uses `phi` and `Phi_u`.

## Compatibility, limitations, and future seam

- `mitc4_stiffness(..., G_eff=...)`, `mitc4_shear_resultants`,
  `panel_abd_tuple_from_lam`, `panel_G_eff`, and `g_p` remain scalar
  compatibility/diagnostic surfaces. Production FE uses
  `PanelConstitutiveSpec.fsdt`.
- Shear correction precedence is explicit corrected `A_s`, transformed
  ply-integrated `A_s`, then the documented scalar compatibility builder.
- Thermal/hygroscopic resultants are zero. Curved directors, large rotations,
  nonlinear response, dynamics, HSDT/layerwise kinematics, \(\sigma_{33}\)
  recovery, 3-D Hashin, cohesive damage, and delamination propagation are out
  of scope.
- A future layered HEX8/solid-shell plus cohesive-interface submodel may consume
  shell \(N/M/Q\), panel geometry, ply stacks, interface positions, and
  material-frame strengths. It is not implemented here.
