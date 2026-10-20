---
name: isometry-pursuit-manuscript
description: Edit, reorganize, build, or review the Isometry Pursuit LaTeX manuscript, archival draft, workshop paper, bibliography, supplement, or NeurIPS talk. Use for changes to .tex, .bib, figure references, theorem prose, and paper structure.
---

# Isometry Pursuit Manuscript

## Workflow

Make very sparse edits. Change only the text, LaTeX, figure placement, or structure the user explicitly asks to change. Do not opportunistically rewrite neighboring paragraphs, polish unrelated prose, rename labels, reformat files, update generated outputs, or move content unless requested. If an adjacent issue is noticed, mention it as a suggestion instead of editing it.

Identify the target surface before editing:

- `archival/`: current archival or submission-oriented manuscript.
- `workshoppaper/`: workshop paper variant.
- `isometry_pursuit_neurips_talk/`: Beamer talk.

Read the local `main.tex` and adjacent `config.tex` before making structural edits. For the archival paper, always read `archival/config.tex` so theorem-like environments and macros are used consistently. Preserve the existing split-file structure: introduction, background, method, experiments, discussion, bibliography, then supplement where applicable.

For prose edits, keep mathematical claims precise and avoid introducing terminology not already established nearby. For notation edits, update all dependent files and macros together.

For paper-writing style, prefer copy/paste reorganization of existing prose over inventing new prose when the needed idea is already somewhere in the manuscript. Actively look for redundancy, repeated motivation, duplicated definitions, and claims introduced before their context. When the paper would read better by moving material, suggest the reorganization explicitly, then make the smallest requested or clearly implied move. Aim for a logical flow where each section introduces concepts once, uses them consistently, and points forward to the next part of the argument.

For mathematical writing in the archival paper, make sure formal claims are expressed with the existing `definition`, `proposition`, and `property` environments when appropriate. Keep proofs out of the main paper body; put all proofs in `supplement.tex` and leave the main text focused on statements, intuition, and consequences.

For notation in formulas and pseudocode, index lowercase over the matching uppercase set or cardinality. For example, use `d = 1, \dots, D`, `p = 1, \dots, P`, `r = 1, \dots, R`, and `s \in S`. Avoid unrelated loop letters such as `t` unless the indexed object is actually time or a quantity named `T`; if the natural lowercase index is already in use, choose another lowercase index tied to the same object rather than switching to an unrelated symbol.

For figure edits, check both the Python generator under `isometry_pursuit/exec/` and the LaTeX include path. Prefer preserving figure filenames that are already referenced. Ensure compiled figures are top-of-page floats whenever possible, usually with `\begin{figure}[t]` or `\begin{figure*}[t]`, and adjust surrounding text so figures do not drift awkwardly.

For ICLR-style presentation, make the compiled paper look polished: avoid cramped captions, orphaned headings, overfull boxes, underexplained figures, stale cross-references, and float placement that interrupts the argument. Prefer concise captions that state what the reader should notice.

Do not edit generated PDFs unless the user explicitly asks. Rebuild them from LaTeX instead.

## Build Checks

From the relevant manuscript directory, prefer `latexmk` if available:

```bash
latexmk -pdf main.tex
```

Fallback:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Report unresolved references, citation warnings, overfull boxes that affect visible layout, float-placement problems, and compile errors with file and line when available. When figure placement changes are requested, inspect the compiled PDF or LaTeX log enough to confirm figures land at the top of pages whenever feasible.

## References

Read `references/latex-map.md` for manuscript directory structure and build notes.
