---
name: isometry-pursuit-source-verification
description: Verify cited algorithms, mathematical claims, and literature comparisons for Isometry Pursuit before recommending citations or asserting equivalence, novelty, or transferable guarantees in chat or manuscript text.
---

# Isometry Pursuit Source Verification

Verify the specific connection a citation is meant to support. A shared title, problem name, or use of additions and deletions is not enough to establish algorithmic equivalence.

## Read the evidence for the claim

Use the original paper's definitions, algorithm, or theorem and record the relevant section, page, or equation when useful. An abstract can establish broad relevance, but it does not verify a detailed update rule or theorem hypothesis. Distinguish what the source states from our own deduction. If access is incomplete, say exactly what was verified rather than filling gaps from memory. Check existing manuscript citations before recommending a new entry.

For local sources, search the provided directory first. This project has also used `/Users/samsonkoelle/bench/pdfs/isometry_pursuit/` for downloaded references. Scanned PDFs may require OCR; check equations and indices visually when OCR is ambiguous.

## Compare mechanisms, not names

Before claiming that a source matches our algorithm, check the dimensions and selected objects, objective, constraints, initialization, candidate scoring, refitting, move selection, and stopping rule. Read our current pseudocode or implementation when describing what we actually run. In particular:

- Forward greedy may evaluate the full resulting loss, whereas OMP ranks residual correlations. These are not interchangeable rules.
- FoBa alternates forward additions and adaptive backward pruning. Our swap refinement jointly evaluates incoming/outgoing pairs at fixed cardinality; call it **swap search**, not FoBa.
- Choosing an addition and then choosing a deletion need not yield the best complete swap.
- Sampled swap search examines fewer incoming candidates. The search structure may apply to our loss even when the source's sampling probabilities do not.
- Continuous transfers between design weights are not automatically swaps between fixed-size subsets.

When translating formulas, follow `../isometry-pursuit-notation/SKILL.md`. Check what each axis indexes and whether the source uses gradients, an orthonormal tangent basis, or a metric. Matching shapes alone does not identify these matrices.

## Keep guarantees attached to their hypotheses

Check the objective and assumptions before importing a guarantee: rank, matrix shape, coefficient bounds, noise regime, and target cardinality can all matter. For example, backward regression is not globally optimal in general just because a paper proves optimality under sufficient conditions. A theorem for least-squares reconstruction does not establish one for our isometry loss.

An optimal solution and dual certificate for the IP convex problem do not by themselves certify the best cardinality-constrained subset or safe pruning for arbitrary dictionaries. State our support-preservation claim only under the actual theorem's assumptions. Likewise, distinguish local differential independence from global injectivity.

Use simple limiting cases to test purported equivalences. For example, selecting a full-rank square basis makes projection reconstruction error zero regardless of orthonormality. That objective cannot distinguish all the selections our loss distinguishes.

## Separate verification from exposition

Keep the detailed comparison in working analysis or chat when needed. In the paper, cite only the connection relevant at that location; verification does not require a survey or a new experiment. Follow `../isometry-pursuit-style/SKILL.md` for the final prose. Reviewing a source does not authorize manuscript edits unless requested.
