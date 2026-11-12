# Manuscript issues observed while building the dependency graph

These are notes only. None of the corrections below has been applied to the
manuscript.

## Mechanical and notation issues

- archival/background.tex:105: citation key Koelle2024- is incomplete;
  Koelle2024-no exists in the bibliography.
- archival/method.tex:75: \label{prop:log_svd_bad} appears where a
  \ref{prop:log_svd_bad} is intended. This also duplicates the proposition's
  actual label.
- archival/method.tex:207-245: \(w(X)\) is used as the matrix normalization
  without a formal definition connecting it to the columnwise extension of
  \(n_g\).
- archival/method.tex:131: the determinant identity is misstated. The standard
  identity is \(\log|\det Z|=\sum_d\log\sigma_d(Z)\) for invertible square
  \(Z\).
- archival/method.tex:135: the inverse-Frobenius identity is misstated. The
  standard identity is
  \(\|Z^{-1}\|_F^2=\sum_d\sigma_d(Z)^{-2}\).
- archival/method.tex:224-226: the minimum-norm MBP selector is named MLP,
  and the displayed solution-set notation does not match the preceding
  definition.
- archival/method.tex:244: the support map has codomain
  \(\binom{[P]}D\), although an IP support need not have cardinality \(D\).
- archival/method.tex:271: the exact-containment statement compares the index
  set \(S\) directly with the coefficient matrix \(\widehat\beta(X)\), rather
  than its row support.
- archival/method.tex:307 and archival/supplement.tex:903: the two examples
  share the label prop:the_benzito.
- archival/method.tex:332 and archival/method.tex:338: two different
  complexity propositions share prop:computational_complexity.
- archival/supplement.tex:121: both property references are broken.
- archival/supplement.tex:473 and archival/supplement.tex:489: two different
  propositions share prop:basis_pursuit_loss_equivalence.
- archival/supplement.tex:489-503: the claimed equivariance appears to need a
  transpose:
  \(\widehat\beta(UX)=\widehat\beta(X)U^T\).
- archival/supplement.tex:512: the KKT proposition's set-membership statement
  is malformed; its intended meaning appears to be existence of an MBP solution
  whose support equals \(S\).
- archival/method.tex:234: sec:deep_dive is undefined, “Multilpliers” is a
  typo, and “feature of the feature” is duplicated prose.
- archival/method.tex:343: SCS is described as an interior-point solver even
  though it is a first-order splitting conic solver.

Several definitions and propositions also lack labels. The dependency YAML
records them with status: unlabeled.

## Mathematical points requiring review

- archival/supplement.tex:569-583: the support-containment proof says the
  vectors of row norms of the MBP solution set form an affine subspace. Row
  norms are nonlinear, so that argument does not establish the proposition as
  written. This matters because exact-isometry containment depends on it.
- archival/method.tex:138-142: scalar properties of \(g\) are asserted for
  matrix losses without defining the corresponding matrix-level statements.
  Both propositions need precise statements and proofs.
- archival/supplement.tex:621: \(w(X)^Tw(X)\) is \(P\times P\), but the
  displayed update adds \(\gamma I_D\). The ADMM update and the claimed
  \(O(D^2P)\) per-iteration complexity need to be derived with consistent
  dimensions and the actual SCS linear-system kernel.
- archival/supplement.tex:901 onward: the higher-dimensional frame example
  and following proof are unfinished.

