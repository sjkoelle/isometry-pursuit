---
name: isometry-pursuit-code
description: Work on the isometry_pursuit Python package in this repository. Use when changing package modules, command-line experiment scripts, plotting utilities, imports, package organization, or numerical algorithm code for Isometry Pursuit.
---

# Isometry Pursuit Code

## Workflow

Make very sparse edits. Change only the modules, functions, scripts, or behavior the user explicitly asks to change. Do not opportunistically refactor, rename, reformat, modernize style, reorganize imports, migrate result schemas, or touch nearby code unless it is required for the requested change. If an adjacent issue is noticed, mention it as a suggestion instead of editing it.

Start by reading the nearest package module and the corresponding entry point under `isometry_pursuit/exec/`. Prefer `rg` for symbol discovery.

Keep reusable functions in focused package modules. Treat scripts in `isometry_pursuit/exec/` as thin command-line entry points; move shared plotting, experiment, transformation, and algorithm logic into package modules instead of importing from a large script.

Use NumPy arrays as the common data boundary. Preserve existing result keys unless deliberately migrating all callers:
`beta`, `basis_pursuit_indices`, `basis_pursuit_iterations`, `objective_value`, `support_cardinality`, `group_lasso_norm`, `greedy_gl_indices`, `greedy_gl_norm`.

When editing solver behavior, inspect `isometry_pursuit/algorithm.py`, `isometry_pursuit/loss.py`, `isometry_pursuit/transformation.py`, and all affected callers. CVXPY/SCS behavior can be slow or numerically fragile, so validate with the smallest relevant script or import smoke test first.

When adding a new command-line script, follow the existing module invocation style:

```bash
python -m isometry_pursuit.exec.<module_name> --help
```

## Validation

Prefer targeted checks before long experiments:

```bash
python -m compileall isometry_pursuit
python -m isometry_pursuit.exec.<module_name> --help
```

If changing experiment outputs, run one small data/figure path when available and inspect the produced JSON, pickle, CSV, or figure path.

## References

Read `references/repo-map.md` for package layout, known commands, and module organization rules.
