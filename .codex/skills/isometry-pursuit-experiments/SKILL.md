---
name: isometry-pursuit-experiments
description: Run, inspect, extend, or debug Isometry Pursuit experiments, datasets, result files, convergence analyses, and generated figures in this repository. Use for data preparation, experiment execution, plotting, and analysis workflows.
---

# Isometry Pursuit Experiments

## Workflow

Make very sparse edits. Change only the experiment script, command, config, result handling, or figure requested by the user. Do not opportunistically rerun unrelated experiments, regenerate unrelated figures, alter output paths, rename files, or rewrite analysis code unless required for the requested change. If an adjacent issue is noticed, mention it as a suggestion instead of editing it.

Read `README.md` first for canonical commands, then inspect the specific script under `isometry_pursuit/exec/`. Use `--help` on entry points before assuming flags.

Keep experiment outputs under existing locations unless the user requests otherwise:

- `data/` for prepared tensors and downloaded/raw data;
- `results/` for JSON, pickle, CSV, and run summaries;
- `figures/` for generated paper figures.

Prefer small validation runs or existing result files before launching expensive full experiments. Ethanol preparation depends on an external `ethanol.mat` download and `configs/ethanol/ethanol_config.json`.

When changing plotted figures, check both the generator and every manuscript file that includes the figure. Preserve filenames used in LaTeX unless intentionally updating all references.

## Result Shape

Tensor experiments usually consume arrays shaped `(D, P, R)` and iterate over replicate index `R`.

Common per-replicate fields include:

- `replicate`
- `support_cardinality`
- `group_lasso_norm`
- `basis_pursuit_iterations`
- `basis_pursuit_indices`
- `greedy_gl_indices`
- `greedy_gl_norm`
- `greedy_gl_bp_indices`
- `greedy_gl_bp_norm`

## References

Read `references/commands.md` for the repo's data, experiment, plotting, and convergence commands.
