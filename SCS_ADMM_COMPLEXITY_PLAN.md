# Plan: a complexity derivation covering direct ADMM and SCS

This document plans the revision of Proposition 17 and its supplemental proof.
It does not change the manuscript, solver implementations, or experimental results.

## Intended claim

Explain why both solver formulations admit an arithmetic cost of order `PD²` per iteration for a dense, full-row-rank dictionary `X ∈ R^{D×P}`, with `P ≥ D`, after the required factorizations.
State the assumptions supporting this bound and separate setup work from iteration work.
Connect the result to Figure 3's proxy `observed iterations × D²P` without claiming that an ADMM iteration and an SCS iteration take equal time or produce equal iterates.

Keep writing the normalized dictionary as `n_q^{×P}(X)`.
Do not introduce `N`, `w(X)`, or another abbreviation for it.
Leave the proposed subtraction of `D` from the objective as the existing commented TODO.

## Evidence already established

- The published SCS paper describes ADMM applied to the homogeneous self-dual embedding.
- Section 3.2, printed page 1051, states that direct ADMM on the primal or dual cone program has the same per-iteration cost as the embedding method: a linear-system solve and a cone projection.
- Section 4.1 describes solving the embedding's linear system through a smaller system and cached factors.
- ADMM's equivalence to Douglas–Rachford splitting on the dual explains the shared algorithmic framework.
- These facts support a shared complexity argument, but do not establish that the existing direct-ADMM equations are SCS's actual update equations.

Primary references:

1. [O'Donoghue, Chu, Parikh, and Boyd, JOTA 2016](https://web.stanford.edu/~boyd/papers/pdf/scs.pdf), Sections 3.2 and 4.1; existing bibliography key `ocpb:16`.
2. [Eckstein and Bertsekas, Mathematical Programming 1992](https://stuff.mit.edu/people/dimitrib/Eckstein_Bertsekas_D-R.pdf), for the ADMM/Douglas–Rachford relationship.
3. [SCS algorithm documentation](https://www.cvxgrp.org/scs/algorithm/index.html) and [linear-solver documentation](https://www.cvxgrp.org/scs/linear_solver/index.html).

Use the journal paper for the published mathematical argument.
Check the installed and deployed SCS versions before using implementation details from current documentation; current documentation may describe a newer release.

## 1. Fix the common problem and the scope of the bound

Use the existing objective and constraint:

\[
\min_{\beta\in\mathbb R^{P\times D}}\sum_{p=1}^P\|\beta_{p.}\|_2
\quad\text{subject to}\quad
n_q^{\times P}(X)\beta=I_D.
\]

State full row rank, `P ≥ D`, fixed normalization, and fixed solver scaling during the iteration phase.
Full row rank makes the affine constraint feasible and permits the direct-ADMM projection below.
Count normalization separately as `O(PD)`.
Distinguish the quadratic coefficient matrix in SCS's standard form from our dictionary cardinality `P`; do not reuse `P` for that matrix.

## 2. Align the ADMM derivation with the implemented algorithm

Read `isometry_pursuit/cloud_experiments.py::solve_matrix`, ADMM branch.
The current implementation uses affine projection, row shrinkage, and a consensus-dual update.
The existing supplement instead writes a different augmented-Lagrangian scheme involving both `rho` and `gamma`; do not assume that this scheme is the implemented algorithm.

Use a two-block split with one copy constrained by reconstruction and another carrying the row-norm objective.
With scaled consensus-dual variable `u`, verify these updates against the code:

\[
V^k=z^k-u^k.
\]
\[
\beta^{k+1}=V^k+(n_q^{\times P}(X))^\top
\left(n_q^{\times P}(X)(n_q^{\times P}(X))^\top\right)^{-1}
\left(I_D-n_q^{\times P}(X)V^k\right).
\]
\[
z^{k+1}=\operatorname{prox}_{\|\cdot\|_{1,2}/\rho}(\beta^{k+1}+u^k).
\]
\[
u^{k+1}=u^k+\beta^{k+1}-z^{k+1}.
\]

Define the rowwise proximal operation before using it in manuscript prose, or describe it directly as row shrinkage.
Check the factor of `1/2` in the quadratic penalty so that the shrinkage threshold is exactly `1/rho`, as in the code.
Do not retain the existing threshold formula without checking its compatibility with the stated Lagrangian.
State that the inverse notation denotes a solve using cached factors, not an explicit inverse formed at each iteration.

Derive the operation counts:

- Setup: form the `D×D` Gram matrix in `O(PD²)` and factor it in `O(D³)`.
- Projection: two products involving the normalized dictionary and `D` right-hand sides cost `O(PD²)`; the cached triangular solves cost `O(D³)`.
- Row shrinkage: `P` row norms and scalings, each of length `D`, cost `O(PD)`.
- Consensus update and ordinary residual checks cost `O(PD)`; reconstruction and primal-dual diagnostic checks can cost `O(PD²)` and do not increase the claimed order.
- Since `P ≥ D`, the total per iteration is `O(PD²)`.

This route does not require the current Woodbury derivation for a `P×P` inverse.
Replace that calculation if we use the implemented affine-projection split.
If retaining another ADMM formulation, identify it explicitly and derive its updates correctly instead of presenting it as the implemented solver.

## 3. Express exactly the same problem as a cone program

Introduce `t ∈ R^P` and minimize its sum, with reconstruction unchanged and one second-order-cone constraint per row:

\[
\|\beta_{p.}\|_2\leq t_p,\qquad p\in\{1,\ldots,P\}.
\]

Explain that each cone contains `(t_p, beta_{p.})` and has dimension `D+1`.
The cone program therefore has:

- `PD+P` scalar decision variables;
- `D²` reconstruction equalities;
- `P` second-order cones, each of dimension `D+1`;
- no quadratic objective term.

Define the conic constraint operator's domain and codomain if it appears in the proof.
Its matrix, denoted `mathcal A`, has `D²+P(D+1)` rows and `PD+P` columns before redundant constraints or other canonicalization changes.
The reconstruction part contains `D` copies of the normalized dictionary, up to variable ordering; the cone-membership part consists of signed selectors.

Verify this description against CVXPY's actual `get_problem_data(cp.SCS)` output rather than inferring it solely from the mathematical formulation.
Record the variable permutation and cone dimensions on one small test problem.

## 4. Bridge from direct splitting to SCS with the published result

Use the SCS paper's Section 3.2 statement to explain why the embedding does not require a fundamentally different cost argument.
Identify its cached linear solve, cone projection, and embedding-vector updates.
Do not derive the entire homogeneous embedding unless necessary to justify an operation count.

Bound the operations specific to the cone formulation:

- Projecting `P` cones of dimension `D+1` costs `O(PD)`.
- Updates involving the embedding vectors cost `O(PD+D²)`.
- Applying the reconstruction operator and its transpose costs `O(PD²)` because the constraint acts on `D` response columns.
- The remaining task is to establish the cached linear-solve cost for this structured operator.

The Section 3.2 citation transfers the embedding-versus-primal cone-program cost comparison.
It does not alone transfer the complexity of an arbitrary norm-proximal ADMM split to CVXPY's SCS implementation.
Include the small structural argument connecting those two formulations.

## 5. Verify the structured SCS solve before asserting the bound

### Check completed on November 18, 2026

The conic formulation and the structured reduced solve check out.
This establishes an available structured implementation of the bound; it does not yet establish a worst-case fill bound for SCS's default AMD ordering.

I inspected CVXPY 1.7.3 with SCS 3.2.9, matching the versions recorded in the saved convergence study.
For dense normalized dictionaries, the canonical variable order was the `P` epigraph variables followed by the column-major vectorization of `beta`.
The first `D²` constraint rows represented reconstruction.
The remaining constraint rows formed a signed permutation matrix selecting the epigraph and coefficient variables into the second-order cones.
Each response column's reconstruction block exactly matched `n_q^{×P}(X)`.

For positive diagonal scaling matrices `R_x` and `R_y`, eliminating the dual variables from SCS's linear system gives the coefficient matrix

\[
R_x+\mathcal A^\top R_y^{-1}\mathcal A.
\]

The cone-selector rows contribute only diagonal entries to this matrix.
The reconstruction rows leave `D` independent coefficient blocks, each of size `P×P`, plus `P` uncoupled epigraph entries.
For response index `d`, the coefficient block has the form

\[
C_d+(n_q^{\times P}(X))^\top E_d n_q^{\times P}(X),
\]

where `C_d ∈ R^{P×P}` and `E_d ∈ R^{D×D}` are positive diagonal matrices.
These matrices describe the solver's diagonal scaling, not additional normalization functions or aliases for the dictionary.
Diagonal equilibration of the data also preserves this block structure; its column weights can be included in `C_d` and the diagonal factors in the Woodbury application.

For a right-hand side `r_d ∈ R^P`, Woodbury applies this block's inverse by forming `C_d^{-1}r_d`, multiplying by the normalized dictionary, solving with

\[
E_d^{-1}+n_q^{\times P}(X)C_d^{-1}(n_q^{\times P}(X))^\top,
\]

and multiplying back by the transpose of the normalized dictionary and `C_d^{-1}`.
The reduced matrix is `D×D`.
After caching its factorization, one response costs `O(PD+D²)`.
All `D` responses therefore cost `O(PD²+D³)` per iteration, with another `O(P)` for the epigraph entries.
This is `O(PD²)` when `P ≥ D`.
With unrelated response weights, forming and factoring all reduced systems once costs at most `O(PD³+D⁴)`; shared weights can reduce setup, but this must not be assumed.

Numerical checks used independent random full-row-rank dictionaries, the repository's actual normalization, and arbitrary positive diagonal weights in the normal system.
I compared the full dense normal-system solve against the independent-response Woodbury solves:

| D | P | Conic matrix shape | Cone-selector check | Response-block error | Relative solve error |
|---|---|---|---|---|---|
| 2 | 5 | 19 × 15 | Signed permutation | 0 | 2.36e-16 |
| 3 | 7 | 37 × 28 | Signed permutation | 0 | 1.48e-16 |
| 4 | 11 | 71 × 55 | Signed permutation | 0 | 2.04e-16 |
| 8 | 19 | 235 × 171 | Signed permutation | 0 | 2.16e-16 |

The weighted normal matrices had exactly `P` singleton components and `D` components of size `P` in every case.
The largest relative residual was below `1.8e-16`.
These checks validate the algebra and canonicalization on the tested dimensions; the dimensional argument establishes the structured operation count.
They are not runtime benchmarks or a proof about arbitrary sparse elimination orderings.

The version-pinned [SCS 3.2.9 direct-solver source](https://github.com/cvxgrp/scs/blob/3.2.9/linsys/cpu/direct/private.c) uses AMD ordering and QDLDL factorization, then solves through cached factors.
It does not explicitly implement the response-wise Woodbury procedure described above.
The mathematical structure survives scaling, but the remaining implementation question is whether the actual factor ordering has the required fill bound.
Resolve that question by inspecting or instrumenting its symbolic factorization, or qualify Proposition 17 as a structured-solve bound and keep Figure 3's quantity explicitly a proxy.
Do not claim that the numerical normal-system checks alone certify the default factorization's asymptotic cost.

### Remaining work

Translate the verified cone-selector elimination and independent-response solve into the final proof.
Account explicitly for the embedding's scalar correction and additional vector operations.
Resolve the default-factorization question below before attributing the structured implementation's bound to the measured SCS backend.

Check the following implementation details explicitly:

1. **Scaling:** CVXPY/SCS equilibration and diagonal scaling can change the weights in each response block.
   Do not assume a single common `D×D` factorization serves all response columns unless the scaling actually preserves that structure.
2. **Setup:** if there are `D` different weighted Gram matrices, setup can exceed the direct-ADMM setup cost.
   Give the corresponding setup bound separately; do not claim the same setup cost for both solvers without establishing it.
3. **Sparse direct solver:** the measured SCS calls use its direct solver by default.
   Check the factorization structure or fill under its ordering before claiming that the default implementation realizes the structured bound, rather than merely admitting such an implementation.
4. **Adaptive scaling:** the cloud code disables adaptive scaling and acceleration, allowing a fixed-factorization iteration analysis.
   Confirm this for the exact runs used in Figure 3.
5. **Indirect solver:** exclude it from the proposition's implementation-specific claim unless separately accounting for its inner iteration count.

If the default SCS factorization bound cannot be established concisely, distinguish the proved structured operation bound from the empirical work proxy.
Do not promote an unverified estimate to an exact implementation theorem.

## 6. Make the smallest manuscript revision justified by the checks

Target files:

- `archival/method.tex`: Proposition 17 and the adjacent arithmetic-work explanation.
- `archival/supplement.tex`: its repeated statement, proof, and existing SCS TODO.
- `archival/ref.bib`: reuse `ocpb:16` and existing ADMM references; add a reference only if the final argument needs it.

The intended proof has four short parts: direct-ADMM projection cost, conic reformulation, the cited embedding-cost bridge, and the SCS structured solve plus cone-projection count.
Replace the obsolete update derivation rather than adding a second long derivation beneath it.
Keep setup and per-iteration work distinct and state the `P ≥ D` assumption in both proposition statements.
Update any labels referenced elsewhere before deleting old update equations.

For Figure 3, retain the actual observed SCS iteration counts.
Call `iteration count × D²P` an arithmetic-work proxy, not an exact flop count or a solver-independent runtime prediction.
No numerical reruns or plot changes follow merely from correcting the derivation.

## 7. Checks needed to finish

- Match the ADMM projection, shrinkage, and dual update to `cloud_experiments.py` line by line.
- Conic dimensions, selector structure, diagonal-weighted response blocks, and the structured solve have been checked as recorded in Section 5.
- Verify the deployed version metadata separately, and establish the default sparse factorization's fill bound or qualify the theorem accordingly.
- Verify every function argument, matrix dimension, transpose, scalar Lagrangian, and inverse/solve operation.
- Use explicit normalization formulas, matching lowercase indices and uppercase cardinalities, and one relation per displayed line.
- Add a short purpose sentence before each derivation and repeat the final proposition statement with its original number.
- Rebuild through the existing LaTeX watcher, check for errors and unresolved references, and confirm Figure 3 remains at the top of page 7.

Completion means the manuscript's complexity claim covers the stated ADMM and SCS formulations under explicit assumptions, with no assertion of identical iterates, stopping behavior, or minimum-norm selection.
