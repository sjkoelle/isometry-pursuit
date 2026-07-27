# Paper Organization

Use `paper-organization.yaml` as the machine-readable graph of definitions, properties, propositions, and proof-placement edges. Use this file as the human-facing organization plan for rewriting the archival paper so it avoids redundancy and flows logically.

## Background

The background should set up the mathematical object before discussing algorithms. Prefer moving existing prose into this order where possible.

### Isometry

Status: Partial.

Define an isometry of a linear map.

Define an isometry of a non-linear function, via the differential/Jacobian inducing isometries between tangent spaces.

State the two basic linear characterizations:

- singular-value characterization: an isometry has all singular values equal to `1`;
- orthonormal-column characterization: in the square case, isometry is equivalent to orthonormal columns.

Current gap: the archival draft now has the nonlinear definition, but the linear isometry definition and orthonormal-column characterization still need notation cleanup.

### Best Subset Selection

Status: Partial.

Use the isometry characterizations to motivate best subset selection: choose a column subset whose submatrix minimizes an isometry loss.

Say this reduces the problem to choosing an isometry loss `\ell`.

Current gap: the optimization problem exists in `archival/background.tex`, but it currently appears after S-divergence rather than before the loss-design discussion.

### Functions Minimized By Isometries

Status: Partial.

Characterize functions minimized by isometries.

Introduce S-divergence as the first concrete example of an isometry-minimized loss, including its singular-value form.

Current gap: S-divergence is introduced, but the section does not yet give a general characterization of isometry-minimized losses before presenting examples.

### Basis Pursuit

Status: Incomplete.

Explain that exact subset selection is combinatorial, and that sparse convex relaxation is the standard way to replace support search with coefficient sparsity.

Introduce basis pursuit as the optimization template `arg min f(\beta) : Y = X\beta`, with `\ell_0` giving a non-convex sparse problem and `\ell_1` giving a convex relaxation.

Introduce multitask/group basis pursuit as the row-sparse version used when the response is matrix-valued.

Connect to Koelle et al. on functional independence and selecting interpretable dictionary elements, then say this paper adapts that paradigm to isometry selection where the response is `I_D` and selected rows of `\beta` correspond to selected columns/features.

Current gap: the archival draft mentions sparse convex relaxation and scalar basis pursuit, but it does not yet introduce multitask/group basis pursuit or the Koelle et al. functional-independence connection in Background.


## Method

<!-- Introduce hyperbolic cosine and explain why it is not great. -->
The method should explain why S-divergence-type metrics are not sufficient, then build the proposed loss and sparse selection method in a logical sequence.

1. Explain that S-divergence and related spectral metrics do not have the desired level-set or pathwise length property.
2. State the proposition that no loss of the form `sum_d g(sigma_d(X))` can satisfy the desired property together with log symmetry.
3. Introduce the method-of-frames solution norm for the rescaled/normalized matrix.
4. State that the `D`-minimizer is an isometry.
5. State that this method-of-frames objective is invariant under the relevant basis changes.
6. State that it satisfies the good level-set property.
7. Connect the objective with trace, so the reader sees the bridge between frame geometry and the optimization form.
8. Transition from the dense method-of-frames objective to sparse selection.
9. Introduce the multitask basis pursuit formulation as the convex sparse-selection method.
10. State that the minimizing support is not unique.
11. State the recovery/containment proposition: when a true isometric support exists, the true support is contained in the recovered support.
12. Explain the practical consequence: use Isometry Pursuit to prune candidates, then optionally apply brute search on the reduced support.

The main text should contain definitions, properties, propositions, intuition, and consequences. Put all proofs in the supplement.

## Reorganization Notes

- Avoid repeating motivation for coordinate selection in multiple sections; introduce it once in the introduction and use the background to formalize it.
- Keep the background about isometry and isometry-minimized losses; keep algorithmic claims in the method.
- Make properties parallel: if pathwise length preference and log symmetry are both central, present both as `property` environments.
- Split duplicated proposition labels before serious cross-reference cleanup.
- Prefer copy/paste movement of existing paragraphs over rewriting from scratch when the necessary prose already exists.
