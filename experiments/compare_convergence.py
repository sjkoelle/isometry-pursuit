"""Compare warm-started vs fresh-restart glnorm vectors at matched iteration counts."""

import numpy as np
import matplotlib.pyplot as plt

warm = np.load('/Users/samsonkoelle/isometry-pursuit/experiments/glnorms_warmstart.npz')
rest = np.load('/Users/samsonkoelle/isometry-pursuit/experiments/glnorms_restarts.npz')

iters = warm['iters']
assert np.array_equal(iters, rest['iters']), "iteration checkpoints don't match"

eps_labels = ['eps=1e-4', 'eps=1e-12', 'eps=1e-20']

# --- print table ---
for label in eps_labels:
    w = warm[label]
    r = rest[label]
    diff = np.abs(w - r)
    print(f"\n{label}")
    print(f"  max |warm - fresh|: {diff.max():.2e}")
    print(f"  mean |warm - fresh|: {diff.mean():.2e}")
    print(f"  all close (atol=1e-6): {np.allclose(w, r, atol=1e-6)}")
    print(f"\n  {'N':>6}  {'warm':>12}  {'fresh':>12}  {'|diff|':>10}  match")
    for i in range(len(iters)):
        match = np.isclose(w[i], r[i], atol=1e-6)
        print(f"  {iters[i]:>6.0f}  {w[i]:>12.6f}  {r[i]:>12.6f}  {diff[i]:>10.2e}  {match}")

# --- plot: one row per eps config ---
fig, axes = plt.subplots(len(eps_labels), 2, figsize=(14, 4 * len(eps_labels)))

for row, label in enumerate(eps_labels):
    w = warm[label]
    r = rest[label]
    diff = np.abs(w - r)

    # left: warm vs fresh on same axes
    axes[row, 0].plot(iters, w, 'o-', ms=3, label='warm-start')
    axes[row, 0].plot(iters, r, 's-', ms=3, label='fresh restart')
    axes[row, 0].set_xscale('log')
    axes[row, 0].set_yscale('log')
    axes[row, 0].set_xlabel('iterations')
    axes[row, 0].set_ylabel('glnorm − D')
    axes[row, 0].set_title(f'{label}: warm vs fresh')
    axes[row, 0].legend()

    # right: absolute difference
    axes[row, 1].plot(iters, diff, 'o-', ms=3, color='tab:red')
    axes[row, 1].set_xscale('log')
    axes[row, 1].set_yscale('log')
    axes[row, 1].set_xlabel('iterations')
    axes[row, 1].set_ylabel('|warm − fresh|')
    axes[row, 1].set_title(f'{label}: absolute difference')

plt.tight_layout()
plt.savefig('/Users/samsonkoelle/isometry-pursuit/experiments/compare_convergence.png', dpi=150)
plt.close()
print("saved compare_convergence.png")
