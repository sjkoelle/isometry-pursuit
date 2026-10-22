# Plan for presenting the four greedy + swap strategies

## Recommendation

Present this as a comparison of two choices: full versus IP-restricted candidate
sets, and forward versus backward greedy initialization. Every method then uses
the same best-improvement, one-column exchange refinement.

Use final loss and planted-basis recovery as separate outcomes. For computational
cost, report measured search work and controlled end-to-end wall time together.
Estimated FLOPs can be an additional implementation-dependent model; they should
not replace both measurements. Current timings exclude cached IP solves and were
collected with concurrent jobs, so they do not establish end-to-end speedups.

This document is a plan. It does not change the manuscript or claim that the
additional cost measurements below have already been collected.

## 1. State the experimental questions

1. Does backward initialization improve the final solution relative to forward
   initialization after applying the same exchange refinement?
2. Does IP restriction preserve useful candidate bases while reducing subsequent
   search work?
3. What solution quality does each complete pipeline deliver for a given
   computational budget?

Avoid the conclusion that every method is equally useful. Current evidence favors
backward + swap for real-data mean loss, and IP + backward + swap for synthetic
planted-basis recovery. Forward + swap remains a useful inexpensive baseline.
Explain where these observations hold rather than declaring a universal winner.

## 2. Define the methods once

| Presentation name | Initial candidate set | Initialization | Refinement |
|---|---|---|---|
| Greedy + swap | Full dictionary | Add until D columns | Best improving exchange |
| Backward + swap | Full dictionary | Delete until D columns | Best improving exchange |
| IP + greedy + swap | IP support | Add until D columns | Exchange within IP support |
| IP + backward + swap | IP support | Delete until D columns | Exchange within IP support |

Use these names in tables, legends, and pseudocode. Existing `forward_backward_*`
result fields and `foba` plotting option keys refer to exchange search, not paper
FoBa. Preserve compatible data keys if needed while changing displayed names.

Explain that forward construction scores the identity on the current selected
span; backward construction keeps the ambient I_D target with redundant columns.
Final exchanges always compare size-D subsets under the same loss. State the
strict-improvement rule, tie handling, and dictionary-size swap-round cap.

## 3. Lead with solution quality

### Real-data table

Use one row per dataset, showing D, P, replicate count, IP support cardinality,
and all four final losses. Report mean and spread, retaining the paired replicate
structure. Do not compare raw loss magnitudes between datasets as though they
shared a scale.

Current mean-loss results, from 25 replicates per dataset:

| Method | Wine | Iris | Ethanol |
|---|---:|---:|---:|
| Greedy + swap | 88.86823 | 7.34319 | 2.70636 |
| Backward + swap | 86.23419 | 7.18169 | 2.69091 |
| IP + greedy + swap | 88.07535 | 7.24601 | 2.70563 |
| IP + backward + swap | 86.24437 | 7.21911 | 2.70563 |

State that backward + swap has the smallest observed mean on all three datasets.
The Wine difference between restricted and unrestricted backward methods is small;
show paired differences and their scale before attaching practical importance.

### Pairwise heatmaps

Keep the 4-by-4 heatmaps as the direct paired comparison. Define each
off-diagonal cell as the paired probability of superiority,
P(row loss < column loss) + 0.5 P(tie), with absolute tie tolerance 1e-9.
Leave diagonal self-comparisons blank and print the tie percentage beneath each
effect size on a common 0–100% scale.

Provide strict wins, ties, and losses in an accompanying table or data file, since
50% could mean all ties or an even split of decisive outcomes. Include a small
sensitivity check for the numerical tie tolerance. Put paired loss-difference
plots in the supplement to show effect sizes that winrates discard.

### Synthetic recovery figure

Use recovery curves against distractor count, separated by ambient D and
corrupting-block dimension. Distinguish:

- exact recovery of the planted column indices;
- recovery of an equally good orthonormal basis;
- final objective value.

An alternative orthonormal basis can tie the planted basis in objective value.
This is why the two IP strategies can tie in the loss heatmap yet differ in exact
support recovery. Do not treat exact planted-support recovery as the only measure
of solving the isometry objective.

Across the current full grid, exact planted-basis recovery is 139/300, 224/300,
244/300, and 252/300 in the method order above. Treat these as descriptive totals:
the grid has unequal numbers of cases across dimensions and only five replicates
per configuration. Lead with stratified curves, not the pooled percentage.

## 4. Report computational cost in three layers

### A. Search work: the primary portable measurement

Instrument initialization and exchange refinement separately. Record:

- number of candidate loss evaluations, by stage and candidate matrix shape;
- rank-check SVD calls and pseudoinverse calls, including rejected candidates;
- initial and final dictionary sizes, accepted deletions, accepted exchanges,
  exchange rounds, and termination reason;
- whether the round cap was reached rather than no improving exchange remaining.

Loss-evaluation count alone is useful but incomplete: backward initialization
scores wide matrices, while final swaps score D-by-D matrices. The current loss
first computes singular values for a rank check and then calls `pinv`, so a valid
candidate generally incurs two decomposition calls. Count actual executed work,
including early rejection and repeated evaluations, rather than an idealized
cached algorithm. NumPy documents the SVD basis of `pinv`:
https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html

For an initial dictionary of size n, the current loops imply:

- forward initialization: sum over k=0,...,D-1 of (n-k) candidate scores;
- backward initialization: sum over k=D+1,...,n of k candidate scores;
- each completed exchange scan: D(n-D) candidate scores, all on D-column sets.

These are candidate-search counts; initial/final reporting scores and other
bookkeeping evaluations must be logged separately. For IP-restricted stages,
replace n by the realized IP support size q. This directly explains the search
savings without treating the IP stage as free.

### B. Estimated arithmetic work: a useful secondary axis

If one scalar is needed to account for varying matrix sizes, accumulate a dense
linear-algebra work proxy. A leading-order model for an SVD or pseudoinverse of a
D-by-m matrix is proportional to D*m*min(D,m). Sum the appropriate model over the
actual recorded kernel calls, not just the number of candidate subsets.

Call the result **estimated dense linear-algebra work** unless a specific kernel
and its constants have been modeled and validated. Do not label a big-O expression
as an exact measured FLOP count. Document whether multiply-add counts as one or
two FLOPs, which kernels are included, and what is omitted.

Record rank-check SVDs, pseudoinverse SVDs/reconstruction, residual products, and
row norms separately if estimating full loss-evaluation FLOPs. Python control,
allocation, memory movement, and threading overhead are not described by this
arithmetic estimate. Rank-update implementations could have different costs, so
frame comparisons as measurements of our current implementations.

### Use the appendix ADMM analysis for an explicit IP arithmetic model

The appendix already derives beta, row-shrinkage, and dual updates in
`archival/supplement.tex`, under `proof:computational_complexity`. Use that as
the starting point for the IP operation count rather than treating IP as an
opaque solver. Audit the derivation before assigning constants: the beta-update
matrix is P-by-P, so its identity must be I_P, and the stated O(P D^2) cost
requires a reduced solve or equivalent structure; it is not the cost of a
generic P-by-P inverse. Also verify splitting dual variables and penalty factors
before implementing these updates as an ADMM solver.

For fixed penalties and D <= P, a Woodbury reduction can use a cached D-by-D
factorization. Separate O(P D^2 + D^3) setup from O(P D^2) leading-order work
per iteration, including the matrix products, small solves with D right-hand
sides, row shrinkage, and dual/residual updates. Derive actual operation-count
constants from the chosen implementation. Then report setup work plus the sum
over executed ADMM iterations, including stopping-check work.

The experiments currently call CVXPY/SCS, not a direct implementation of those
appendix updates. Keep SCS and account for its actual operations; do not replace
it with an appendix solver for this comparison. Do not multiply recorded SCS
iteration counts by the appendix's ADMM per-iteration estimate. The appendix
model can be presented as theoretical algorithmic cost with that distinction
explicit. The source-matched audit in `results/scs_work_audit.md` now counts the
numeric sparse-factorization and factor-solve component for all 75 real-data
problems and records tolerance diagnostics on one replicate per dataset.

Do not manufacture an end-to-end FLOP total by adding SCS iterations to this work
proxy. An SCS iteration is a different unit, and its linear-system setup/solve
cost depends on backend, sparsity, and factorization reuse. The official SCS
linear-solver documentation describes these distinctions:
https://www.cvxgrp.org/scs/linear_solver/index.html

### C. End-to-end wall time: retain practical relevance

Run controlled timing measurements serially, with fixed BLAS thread counts and
recorded hardware/software versions. Include a warm-up and repeat timing trials;
report a median and spread rather than a single concurrent-run measurement.

Separate common preprocessing, IP model construction/solver setup, IP solve, greedy
initialization, and swaps. Report the total pipeline time as well as stage times.
For IP methods, add the cost of computing the support once per dataset replicate;
reuse of cached supports in the existing quality comparison is not a zero-cost IP
algorithm. If supports are intentionally reused across multiple searches, present
amortized cost as a separately defined scenario with the reuse count stated.

For the IP solve also log solver status, iterations, residuals, problem size,
setup/solve time, and linear-system backend. Use matched accuracy settings and
report failures or inaccurate solves. The current iteration count by itself is
not a hardware-independent comparison against the greedy algorithms.

## 5. Proposed figures and tables

Main text:

1. Four-method definition table or compact pseudocode.
2. Synthetic recovery curves, stratified by dimension and distractor type.
3. Real-data final-loss table plus paired winrate heatmaps if space permits.
4. Quality versus computational work plot, with IP setup cost explicitly shown
   separately; an end-to-end time version provides the practical comparison.

Supplement:

- strict win/tie/loss counts and paired loss-difference distributions;
- initialization-to-final improvements, isolating what swaps contribute;
- candidate-evaluation counts, shape-weighted work, and stage timing tables;
- IP support sizes, solver diagnostics, numerical-tolerance sensitivity;
- full configuration grid, seeds, versions, caps, and failure accounting.

Avoid selecting one budget-dependent winner before collecting comparable cost
measurements. A strong unrestricted baseline is essential to demonstrating the
specific contribution of IP restriction.

## 6. Statistical interpretation and numerical checks

Keep all four methods paired on the same input and the same IP support per
replicate. Report effect sizes before significance tests. If intervals or tests
are added, respect the replicate-generation process: overlapping Iris/Wine
subsamples are not new independent datasets, and synthetic configurations are
heterogeneous strata. Do not present a naive pooled interval as a generalization
guarantee. Predefine the primary comparisons and account for multiplicity if
formal claims use all six pairs.

Retain the current basis-size, rank, score, and restriction checks. Inspect
normalization overflow and inaccurate IP solves before publication. Fixes to
numerical handling or solver settings require a documented rerun of affected
comparisons; do not silently mix revised outputs with existing numbers.

## 7. Execution checklist

- [ ] Adopt the four method names and separate loss from planted-support recovery.
- [ ] Add stage-level evaluation/kernel counters, shape histograms, and cap status.
- [ ] Audit the appendix ADMM derivation and make its reduced-solve assumption
      explicit; retain SCS and extend the source-matched factor-work audit to
      the rest of its executed operations.
- [ ] Save IP solver diagnostics and stage costs alongside cached-support provenance.
- [ ] Rerun the same paired experiments with instrumentation and verify unchanged
      selected supports/losses before interpreting the new counters.
- [ ] Run serial controlled timing trials including IP cost.
- [ ] Generate quality, paired-comparison, and cost figures from one result schema.
- [ ] Replace the current two-method archival table and stale FoBa labels; keep
      filenames and LaTeX references coordinated.
- [ ] Write captions that define ties, budgets, recovery targets, and exclusions.
- [ ] Rebuild and inspect the manuscript only after the presentation changes.

## Existing artifacts

- Quality report: `results/swap_comparison_report.md`.
- Four-method summary: `results/swap_comparison_summary.csv`.
- Per-replicate real data: `results/{wine,iris,ethanol}_swap_comparison.json`.
- Synthetic data: `results/identity_swap_comparison_full/`.
- Heatmaps and strict-win/tie counts: `figures/swap_winrates/`.
- Synthetic figures: `figures/identity_swap_comparison_full/`.
- Current manuscript presentation: `archival/experiments.tex`; its table generator
  still summarizes only the two forward-initialized strategies.
