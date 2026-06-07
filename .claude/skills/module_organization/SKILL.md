# Module Organization Policy

## Principle
**Functions should live in their own focused modules rather than being scattered across multiple files.**

Each module should have a single, clear responsibility. When a function is needed by multiple files, it should live in its own module and be imported by those files, rather than having a circular or reverse dependency.

## Bad Pattern
```python
# plot_results.py (30KB, lots of plotting functions)
def plot_sparsity_convergence_boxplot(...): ...
def plot_coefficient_convergence(...): ...
# ... many other functions

# plot_convergence_analysis.py (imports from plot_results)
from isometry_pursuit.exec.plot_results import (
    plot_coefficient_convergence,
    plot_sparsity_convergence_boxplot,
)
```

This creates a situation where utility functions live in a large file and are imported back by smaller, more specialized files.

## Good Pattern
```python
# plot_convergence.py (focused module for convergence plotting)
def plot_sparsity_convergence_boxplot(...): ...
def plot_coefficient_convergence(...): ...

# plot_convergence_analysis.py (specialized analysis script)
from isometry_pursuit.exec.plot_convergence import (
    plot_coefficient_convergence,
    plot_sparsity_convergence_boxplot,
)

# plot_results.py (general results plotting)
from isometry_pursuit.exec.plot_convergence import (
    plot_coefficient_convergence,
    plot_sparsity_convergence_boxplot,
)
```

Each plotting function lives in a focused module, and specialized scripts import from it.

## Refactoring Checklist
- [ ] Identify related functions that belong together
- [ ] Create a focused module with a clear name
- [ ] Move functions to the new module
- [ ] Update all imports to point to the new module
- [ ] Remove the old import dependencies
- [ ] Verify no circular imports exist
