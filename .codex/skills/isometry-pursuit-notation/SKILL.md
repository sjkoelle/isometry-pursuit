---
name: isometry-pursuit-notation
description: Use the Isometry Pursuit paper's matrix indexing and dimension conventions when discussing mathematics in chat or writing manuscript formulas, including translations of cited objectives.
---

# Isometry Pursuit Notation

Use these conventions in chat as well as manuscript edits.

- The dictionary is `X \in \mathbb R^{D\times P}`: `D` rows, `P` candidate columns. A selected set is `S \subseteq [P]`; the target cardinality is `|S|=D` unless explicitly stated otherwise.
- Matrix subscripts give row indices first and column indices second. A dot means every index along that axis. Write `X_{.s}` for column `s`, `X_{.S}` for the selected columns, `X_{k.}` for row `k`, and `X_{kS}` for row `k` restricted to columns `S`. More generally, `X_{RS}` selects rows `R` and columns `S`.
- In particular, write `X_{kS}`, **not** `X_{k.S}`, `X_{k,S}`, or `X_{k.S}^{\top}`. The dot is an entire-axis selector, not punctuation separating indices. Add a transpose only when the mathematical operation requires one; a Euclidean norm does not require transposing a row.
- For `|S|=M`, `X_{.S}` has shape `D\times M`, `X_{kS}` has `M` entries, and `X_{.s}` has `D` entries. Consequently, `X_{.S}^{\top}X_{.S}` is `M\times M`, whereas `X_{.S}X_{.S}^{\top}` is `D\times D`.
- Match indices to their objects: `d\in[D]`, `p\in[P]`, and `s\in S`. Preserve an established local index when explaining an existing formula.

When translating another paper, first identify what each source axis indexes, which axis the selected set restricts, and what the denominator's index ranges over. Then translate into our notation while preserving those meanings. Do not identify a source tangent basis with our dictionary merely because a transpose gives matching dimensions: state any actual change of matrix or preprocessing explicitly. Check dimensions before claiming that a formula needs correction. Use source notation temporarily when the user specifically asks what the source indexes.

For additional notation, read the relevant manuscript definitions and follow them; do not invent a new convention.
