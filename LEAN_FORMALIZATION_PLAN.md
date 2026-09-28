# Lean Formalization Plan

This document lays out a practical plan for formalizing selected Isometry Pursuit
claims in Lean 4 with Mathlib. The goal is not to translate the paper line by
line. The goal is to build a small, checkable formal layer that forces the
paper's definitions and propositions to become precise.

## What Lean Artifacts Are

Lean files are code-like mathematical proof files. They contain definitions,
theorem statements, and proof scripts. A typical theorem has the form:

```lean
theorem example_statement (x : ℝ) : 0 ≤ x^2 := by
  nlinarith [sq_nonneg x]
```

The command `lake build` checks the whole project. A theorem is genuinely
verified only if it builds without `sorry`.

Lean permits temporary placeholders:

```lean
theorem hard_statement : True := by
  sorry
```

This is useful for theorem-statement scaffolding, but it is not a proof. A
`sorry` acts like a temporary axiom and produces a warning. Any proposition we
want to advertise as formalized must be sorry-free.

## Project Layout

Create a standalone Lean project under `formal/`:

```text
formal/
  lakefile.lean
  lean-toolchain
  Formal/
    Basic.lean
    Normalization.lean
    MatrixLosses.lean
    MercedesBenz.lean
    BasisPursuit.lean
    IsometrySelection.lean
```

Initial setup:

```bash
cd formal
lake build
```

Use Mathlib as the primary dependency. Add automation libraries only after
there is a clear need.

## Proof Standards

We will track three statuses:

1. **Statement-only**
   The theorem statement type-checks, but proof contains `sorry`.

2. **Partial formalization**
   Supporting lemmas are proven, but one or more major steps still contain
   `sorry`.

3. **Verified**
   `lake build` succeeds and the target theorem has no `sorry`.

Before citing Lean in the paper, run:

```lean
#print axioms theorem_name
```

The output should not include `sorryAx`.

## Phase 1: Low-Risk Algebraic Facts

These are the best first targets because they are concrete and should not
require a custom optimization library.

### `Formal/Normalization.lean`

Define the scalar normalization

\[
g(t)=\frac{2e}{e^t+e^{1/t}}.
\]

Target statements:

- positivity on `t > 0`;
- `g 1 = 1`;
- log-symmetry, `g t = g t⁻¹` for `t > 0`;
- inverse identity

\[
\frac{1}{g(t)}=\frac{e^t+e^{1/t}}{2e};
\]

- LogSumExp identity

\[
\frac{1}{g(t)}
=\exp(\log(e^t+e^{1/t})-1-\log 2).
\]

This phase should also clarify whether the paper should use `t > 0` everywhere
instead of informal `\mathbb R^+`.

### `Formal/MercedesBenz.lean`

Formalize the concrete example

\[
X =
\begin{bmatrix}
0 & \sqrt 3/2 & -\sqrt 3/2\\
1 & -1/2 & -1/2
\end{bmatrix},
\qquad
\beta =
\begin{bmatrix}
0 & 2/3\\
1/\sqrt 3 & -1/3\\
-1/\sqrt 3 & -1/3
\end{bmatrix}.
\]

Target statements:

- `X * beta = I`;
- each row norm of `beta` is `2/3`;
- `‖beta‖_{1,2} = 2`.

This is a good early win because it tests finite matrices, square roots, row
norms, and exact arithmetic without needing SVD or optimization.

## Phase 2: Matrix-Loss Identities

These statements should be formalized only after checking Mathlib's current API
for finite real matrices, determinants, adjoints, eigenvalues, and singular
values.

### `Formal/MatrixLosses.lean`

Target paper corrections/statements:

- Determinant/log-volume identity:

\[
\log|\det X| = \sum_i \log \sigma_i(X)
\]

for full-rank square real matrices.

- Frobenius inverse identity:

\[
\|X^{-1}\|_F^2 = \sum_i \sigma_i(X)^{-2}.
\]

- Pseudoinverse version:

\[
\|X^\dagger\|_F^2 = \sum_{\sigma_i(X)>0} \sigma_i(X)^{-2}.
\]

The inverse case should come before the pseudoinverse case. The pseudoinverse
will likely require more library spelunking and may need local definitions if
Mathlib's API is not pleasant.

Important: Lean will force the paper to distinguish the Frobenius norm from its
square. The singular-value sum corresponds to the squared Frobenius norm of the
inverse/pseudoinverse.

## Phase 3: Properties of Losses

Formalize the properties from the method section as predicates, not prose.

Possible definitions:

```lean
def LogSymmetric (g : ℝ → ℝ) : Prop :=
  ∀ t, 0 < t → g t = g t⁻¹
```

For pathwise monotonicity, do not start with derivatives through arbitrary SVDs.
Start with restricted, paper-relevant slices:

- 1D scalar family;
- 2D two-column family;
- equicorrelation family.

The full statement

\[
\frac{\partial \ell}{\partial \|x_{\cdot d}\|_2}
\begin{cases}
>0 & \|x_{\cdot d}\|_2 > 1\\
=0 & \|x_{\cdot d}\|_2 = 1\\
<0 & \|x_{\cdot d}\|_2 < 1
\end{cases}
\]

for all matrices is a hard target. It should not be the first proof we attempt.

## Phase 4: Basis Pursuit Without General KKT

The best first route is not to formalize a general KKT theorem. Instead, prove
the Isometry Pursuit optimality certificate directly from norm inequalities.
For normalized columns satisfying `‖x_p‖ ≤ 1`, any feasible coefficient matrix
`β` with `X β = I` should obey

\[
D
= \operatorname{tr}(I)
= \operatorname{tr}(X\beta)
\leq \sum_p \|x_p\|_2 \|\beta_{p\cdot}\|_2
\leq \sum_p \|\beta_{p\cdot}\|_2
= \|\beta\|_{1,2}.
\]

If an identity/orthonormal submatrix is available, the corresponding coefficient
matrix attains this lower bound. This is the Lean-friendly core theorem.

### `Formal/BasisPursuit.lean`

Define:

- matrix row norm;
- group-lasso norm `‖β‖_{1,2}`;
- feasible set `{β | X β = I}`;
- objective minimizers;
- support of a coefficient matrix;
- minimum-Frobenius-norm minimizer, if needed.

First theorem targets:

- row norms are invariant under right multiplication by an orthogonal matrix;
- therefore `‖β U‖_{1,2} = ‖β‖_{1,2}`;
- feasible sets transform correctly under orthogonal basis changes;
- every feasible solution satisfies `D ≤ ‖β‖_{1,2}` under normalized-column
  assumptions;
- the identity-submatrix solution attains `‖β‖_{1,2} = D`.

These are the Lean-shaped core of Proposition
`\ref{prop:basis_pursuit_selection_invariance}`.

### `Formal/IsometrySelection.lean`

Eventually target:

If an orthonormal rank-`D` column submatrix exists, then the support selected by
the Isometry Pursuit program contains that submatrix.

The direct optimality theorem above should be tractable. The remaining delicate
part is the exact selected support under nonunique minimizers and the
minimum-Frobenius tie-breaker. That likely requires:

- a precise choice of solution when minimizers are nonunique;
- a corrected and formal statement of the "maximally non-sparse" min-Frobenius
  tie-breaker claim.

This part should begin as statement-only Lean, with `sorry`s, after Phases 1 and
2 have stabilized the notation.

## Stretch Goal: ADMM/SCS Behavior

The paper currently treats the numerical solver's ability to obtain the desired
minimum-norm solution as an assumption. Formalizing that behavior is a stretch
goal, not a blocker for the core theorem pass.

Possible eventual targets:

- state the exact optimization problem solved by the ADMM/SCS iteration;
- formalize a convergence theorem under standard convexity/closedness
  assumptions;
- connect the limiting point to the minimum-Frobenius tie-breaker used in the
  paper.

This would require either reusing existing convex-analysis infrastructure in
Mathlib or building a local theorem specialized to this program. We should only
attempt it after the direct basis-pursuit optimality theorem is working.

## Current Paper Statements Needing Sharpening

Before formalization, rewrite these in theorem-ready form:

- "The determinant of the normalized matrix `∝ ∑ log σ_d`" should become a
  specific log-determinant equality.

- "The Frobenius norm of the inverse
  `= ∑ 1/σ_d^2 ∝ ∑ σ_d`" should distinguish:

\[
\|X^{-1}\|_F^2 = \sum_d \sigma_d^{-2},
\qquad
\|X^{-1}\|_F = \sqrt{\sum_d \sigma_d^{-2}}.
\]

- The determinant loss should consistently specify whether it is
  `|det(n_g(X))|`, `|det(n_g(X))|^{-1}`, `|det(n_g(X))|^{-1}-1`, or
  `-\log |det(n_g(X))|`.

- `arg min` statements need to say whether the minimizer is unique, a set of
  minimizers, or a selected minimizer under a tie-breaker.

- The KKT proof should be replaced, where possible, by the direct lower-bound
  certificate

\[
D \leq \|\beta\|_{1,2}.
\]

  KKT/subgradient formalization can remain a secondary route if the direct
  proof does not cover a later theorem.

## First Sprint

1. Create `formal/` with Mathlib.
2. Add `Formal/Normalization.lean`.
3. Prove `g 1 = 1`, positivity, and log-symmetry.
4. Add `Formal/MercedesBenz.lean`.
5. Prove `X * beta = I`.
6. Attempt the row-norm and `‖β‖_{1,2}=2` facts.
7. Add theorem skeletons for determinant and Frobenius inverse identities.
8. Prove the direct feasible-solution lower bound for the normalized
   identity-submatrix case.
9. Report which statements are verified, which have `sorry`, and which paper
   claims need restatement.

## Success Criteria

The first successful milestone is not the main theorem. It is:

- a Mathlib project that builds;
- at least one nontrivial paper example verified without `sorry`;
- normalization facts verified without `sorry`;
- the direct group-lasso lower-bound theorem verified without `sorry` or reduced
  to a small number of clearly named analytic lemmas;
- theorem skeletons for the main propositions that type-check;
- a list of manuscript statements revised because Lean made hidden assumptions
  visible.
