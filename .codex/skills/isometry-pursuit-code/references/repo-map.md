# Repository Map

## Package Modules

- `isometry_pursuit/algorithm.py`: greedy selection, line search, `isometry_pursuit`, leave-one-out logic, and CVXPY group basis pursuit helpers.
- `isometry_pursuit/loss.py`: isometry loss, subspace loss, and group lasso norm.
- `isometry_pursuit/transformation.py`: data transformations such as exponential normalization.
- `isometry_pursuit/experiment.py`: tensor experiment orchestration and result summaries.
- `isometry_pursuit/plot.py` and `isometry_pursuit/exec/plot_*.py`: plotting and figure generation.
- `isometry_pursuit/manifold.py`, `atomic_geometry.py`, `simulation.py`, `vendor.py`: domain support modules.

## Entry Points

Run scripts with `python -m isometry_pursuit.exec.<script>` from the repo root so imports resolve consistently.

Common commands:

```bash
python -m isometry_pursuit.exec.load_wine_exec --output-dir data --R 25
python -m isometry_pursuit.exec.load_iris_exec --output-dir data --R 25
python -m isometry_pursuit.exec.run_experiment --data-file data/wine_standardized_resampled_tensor.npy --output-path results/wine_results.json
python -m isometry_pursuit.exec.visualize_loss_landscapes --output-dir figures
```

## Organization Rule

Functions should live in focused modules rather than being scattered through entry-point scripts. When a function is needed by multiple files:

- create or use a focused module with a clear responsibility;
- update scripts to import from that module;
- avoid reverse dependencies where small specialized scripts import utilities from a broad script;
- check for circular imports after moving code.

This mirrors the repo's existing `.claude/skills/module_organization` guidance.
