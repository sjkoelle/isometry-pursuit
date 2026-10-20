"""Reproducible sparse least-squares checks for the paper's FoBa variants."""
import argparse
import json
from pathlib import Path

import numpy as np

from isometry_pursuit.foba import foba_least_squares


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--output-path', type=Path, default=Path('results/foba_smoke.json'))
    args = parser.parse_args()
    if args.trials < 1:
        parser.error('--trials must be positive')
    rows = []
    # Well determined, noiseless sparse regression: 80 samples, 40 features,
    # five nonzeros. Both algorithms see exactly the same seeded problems.
    for seed in range(args.trials):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(80, 40))
        X /= np.linalg.norm(X, axis=0)
        truth = np.zeros(40)
        support = rng.choice(40, size=5, replace=False)
        truth[support] = rng.choice([-1., 1.], size=5) * rng.uniform(1., 2., size=5)
        y = X @ truth
        for variant in ('obj', 'gdt'):
            result = foba_least_squares(X, y, variant=variant, delta=1e-10, epsilon=1e-8)
            rows.append(dict(seed=seed, variant=variant,
                             exact_support=set(result['support']) == set(support),
                             coefficient_error=float(np.linalg.norm(result['beta']-truth)),
                             objective=result['objective_value'], status=result['status'],
                             backward_steps=sum(e['action'] == 'remove' for e in result['history'])))
    summary = {}
    for variant in ('obj', 'gdt'):
        selected = [r for r in rows if r['variant'] == variant]
        summary[variant] = dict(trials=args.trials,
                               exact_support_recoveries=sum(r['exact_support'] for r in selected),
                               max_coefficient_error=max(r['coefficient_error'] for r in selected),
                               max_objective=max(r['objective'] for r in selected),
                               all_converged=all(r['status'] != 'max_iterations' for r in selected))
    output = dict(problem=dict(samples=80, features=40, sparsity=5, noise=0,
                               delta=1e-10, epsilon=1e-8), summary=summary, trials=rows)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print(f'Saved {args.output_path}')


if __name__ == '__main__':
    main()
