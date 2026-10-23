# Finish plan for the Isometry Pursuit paper

## Paper claim

Isometry Pursuit is a convex screen for finding approximately orthonormal
cardinality-\(D\) subsets of an overcomplete dictionary. The paper contributes:

1. tangency and pathwise arguments showing defects in S-divergence and related
   spectral losses;
2. a symmetric normalization that removes the resulting length preference -> determinant and frob norm of pseudoinverse both satisfy tangency condition;
3. Extend to P>D case.  Isometry Pursuit finds sparsity from P > D, with basis invariance, exact-isometry containment. The
   Mercedes--Benz example showing why containment cannot be promised;
4. experiments showing when screening improves the final discrete search and
   when it does not.

Coherence is not useful here because the sparsity level is \(k=D\). Standard
coherence guarantees deteriorate with \(k\) and are already vacuous in the
overcomplete regimes of interest. Worse, a worst-case pairwise correlation does
not describe the collective geometry that determines the best \(D\)-column
subset or whether IP retains it. A coherence analysis would obscure the actual
problem rather than explain it.

## Results to present

- **Synthetic identity-block experiment:** use excess loss \(L-D\) as the main
  outcome. Show that IP can remove structured corrupting frames that trap direct
  greedy paths. Retain target dimension, corrupting dimension, and number of
  corrupting blocks. Use support recovery and IP support size as secondary
  diagnostics.
- **Wine, Iris, and Ethanol:** report four-method oracle win/tie rate, oracle
  regret, and \(|\widehat S_{IP}|/P\). State the mixed result plainly: full
  backward search is strongest on average, while IP sometimes finds a better
  solution and often makes the second-stage search much cheaper.
- **Computation:** show arithmetic-work proxy against wall time with early
  stopping. Explain that the \(10^9\) line is the work budget for omitting costly
  backward runs, not a runtime cutoff.

The four pipelines are forward greedy or backward greedy, run either on the full
dictionary or the IP support, followed by the same exhaustive best-improvement
one-column swap search. The swap stage checks every add/delete pair until no
strict improvement remains or the documented cap is reached.

<!-- TODO(citations): Find verified sources for forward greedy and backward
greedy. Find a swap-search citation only if its move and objective actually
match this implementation; otherwise describe the algorithm directly. -->

## Main-text order

1. Geometric motivation, normalization, and one-dimensional losses.
2. Two- and three-dimensional loss landscapes.
3. Isometry Pursuit, its exact results, and the Mercedes--Benz limitation.
4. Synthetic excess loss beside arithmetic work versus runtime.
5. Real-data oracle-regret table with IP support sizes.
6. Support recovery and trimming diagnostics, moved to the appendix if needed.
7. A short discussion of the mixed empirical result.

## Finish checklist

- [ ] Finish and verify the timed synthetic sweep: 130 conditions, 25 replicates,
      saved runtimes, valid records, and agreement with the untimed results.
- [ ] Regenerate only timing-dependent figures and tables, then freeze results.
- [ ] Record machine, threads, solver version, stopping rule, and omitted
      backward runs.
- [ ] Rewrite the abstract and introduction around the four contributions.
- [ ] Use consistent names for the four pipelines and their common swap stage.
- [ ] Resolve only verified citations; remove broken references, duplicate
      labels, abandoned propositions, and stale TODO text.
- [ ] Build and visually inspect the paper and a neutral arXiv package.
- [ ] Submit the preprint and stop.

Do not add a coherence analysis, another theorem, more datasets, more search
variants, a replacement solver, or conference-driven filler unless final
verification exposes a correctness problem.
