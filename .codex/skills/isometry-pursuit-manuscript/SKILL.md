---
name: isometry-pursuit-manuscript
description: Edit, reorganize, build, or review the Isometry Pursuit LaTeX manuscript, archival draft, workshop paper, bibliography, supplement, or NeurIPS talk. Use for changes to .tex, .bib, figure references, theorem prose, and paper structure.
---

# Isometry Pursuit Manuscript

## Python environment

Before running Python, `pip`, `pytest`, or a Python module, activate the project's canonical Conda environment. Keep activation and the dependent command in the same shell invocation:

```bash
source activate isometry_pursuit
python -c 'import sys; print(sys.prefix)'
```

Never silently use the system interpreter or switch to another environment. If `isometry_pursuit` lacks a dependency, report or install it there.

## Workflow

Make very sparse edits. Change only the text, LaTeX, figure placement, or structure the user explicitly asks to change. Do not opportunistically rewrite neighboring paragraphs, polish unrelated prose, rename labels, reformat files, update generated outputs, or move content unless requested. If an adjacent issue is noticed, mention it as a suggestion instead of editing it.

Identify the target surface before editing:

- `archival/`: current archival or submission-oriented manuscript.
- `workshoppaper/`: workshop paper variant.
- `isometry_pursuit_neurips_talk/`: Beamer talk.

Read the local `main.tex` and adjacent `config.tex` before making structural edits. For the archival paper, always read `archival/config.tex` so theorem-like environments and macros are used consistently. Preserve the existing split-file structure: introduction, background, method, experiments, discussion, bibliography, then supplement where applicable.

For prose edits, keep mathematical claims precise and avoid introducing terminology not already established nearby. For notation edits, update all dependent files and macros together.

For prose and proposed paper sentences, read `../isometry-pursuit-style/SKILL.md`
and follow its guidance on informative comparisons, grounded terminology, and direct attribution.

For matrix indexing, dimensions, and translations of source formulas, read
`../isometry-pursuit-notation/SKILL.md` and follow its conventions in both manuscript text and chat explanations.

When verifying or comparing cited methods and guarantees, read
`../isometry-pursuit-source-verification/SKILL.md` before asserting the connection or adding the citation.

When restating, comparing, or tabulating results from cited work, always translate
the source's notation into the notation already established in this manuscript.
Do not import source-specific symbols merely to mirror the source. Preserve the
source's mathematical meaning, and state the notation mapping only when it is not
clear from context.

For paper-writing style, prefer copy/paste reorganization of existing prose over inventing new prose when the needed idea is already somewhere in the manuscript. Actively look for redundancy, repeated motivation, duplicated definitions, and claims introduced before their context. When the paper would read better by moving material, suggest the reorganization explicitly, then make the smallest requested or clearly implied move. Aim for a logical flow where each section introduces concepts once, uses them consistently, and points forward to the next part of the argument.

For mathematical writing in the archival paper, make sure formal claims are expressed with the existing `definition`, `proposition`, and `property` environments when appropriate. Keep proofs out of the main paper body; put all proofs in `supplement.tex` and leave the main text focused on statements, intuition, and consequences.

For notation in formulas and pseudocode, index lowercase over the matching uppercase set or cardinality. For example, use `d = 1, \dots, D`, `p = 1, \dots, P`, `r = 1, \dots, R`, and `s \in S`. Avoid unrelated loop letters such as `t` unless the indexed object is actually time or a quantity named `T`; if the natural lowercase index is already in use, choose another lowercase index tied to the same object rather than switching to an unrelated symbol.

For figure edits, check both the Python generator under `isometry_pursuit/exec/` and the LaTeX include path. Prefer preserving figure filenames that are already referenced. Ensure compiled figures are top-of-page floats whenever possible, usually with `\begin{figure}[t]` or `\begin{figure*}[t]`, and adjust surrounding text so figures do not drift awkwardly.

For ICLR-style presentation, make the compiled paper look polished: avoid cramped captions, orphaned headings, overfull boxes, underexplained figures, stale cross-references, and float placement that interrupts the argument. Prefer concise captions that state what the reader should notice.

Do not edit generated PDFs unless the user explicitly asks. Rebuild them from LaTeX instead.

## Build Checks

This checkout has a macOS LaunchAgent, `org.isometry-pursuit.manuscript-watch`,
running `latexmk -pdf -pvc -view=none` for `archival/main.tex`.
See `infra/latex/README.md` for installation and control commands.
When it is running, let it rebuild after dependency changes and verify completion
in `archival/automatic-build.log`; do not start a competing build.
It updates the PDF without opening or activating any windows.

The user uses TeXShop on macOS. After completing requested manuscript or
paper-figure edits, rebuild the affected manuscript's normal `main.pdf` in its
manuscript directory (`archival/main.pdf` for the archival paper), so the open
TeXShop preview can detect the updated file. This rebuild is part of the edit
workflow and does not require another confirmation. Temporary preview builds
may supplement this check, but do not substitute for updating the normal PDF
unless the user explicitly requests a preview-only build. Report build failures
and whether the normal PDF was successfully updated.

Run builds through the shell without opening or activating TeXShop, source,
log, or PDF windows; preserve the user's window focus and arrangement. If the
normal PDF updates but TeXShop does not refresh, distinguish the viewer's reload
behavior from a build failure rather than claiming the preview refreshed.

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
