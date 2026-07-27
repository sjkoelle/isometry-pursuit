# LaTeX Map

## Manuscript Directories

`archival/` includes:

- `main.tex`
- `config.tex`
- `introduction.tex`
- `background.tex`
- `method.tex`
- `experiments.tex`
- `discussion.tex`
- `supplement.tex`
- `ref.bib`
- local style files including `iclr2026_conference.sty`, `neurips_2025.sty`, and TikZ support.

`workshoppaper/` includes a similar split with `iai_neurips_2024.sty`.

`isometry_pursuit_neurips_talk/` contains the Beamer talk, `ref.bib`, and `files/` section inputs.

## Current Archival Main Flow

`archival/main.tex` loads:

```tex
\input{introduction.tex}
\input{background.tex}
\input{method.tex}
\input{experiments.tex}
\input{discussion.tex}
\bibliography{ref}
\input{supplement.tex}
```

Keep section-level edits in the corresponding file unless the user asks to reorganize the paper.

## Notes

- Bibliography style in the archival draft is `unsrtnat`.
- The archival draft currently uses `\usepackage{iclr2026_conference}`.
- `archival/main.pdf` is generated output; prefer rebuilding it instead of hand-editing.
