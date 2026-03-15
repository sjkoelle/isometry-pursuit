"""Micro experiment: 2 iterations, chunk=1. Fresh restart vs warm-start."""

import csv
import numpy as np
import cvxpy as cp
import scs

from isometry_pursuit.transformation import exponential_transformation

data_tensor = np.load('/Users/samsonkoelle/isometry-pursuit/data/ethanol_tensor.npy')
X_t = exponential_transformation(data_tensor[:, :, 1])
D, P = X_t.shape

beta_var = cp.Variable((P, D))
prob = cp.Problem(
    cp.Minimize(cp.sum(cp.norm(beta_var, axis=1))),
    [X_t @ beta_var == np.eye(D)]
)
data, chain, inverse_data = prob.get_problem_data(cp.SCS)
dims = data['dims']
scs_data = {'A': data['A'], 'b': data['b'], 'c': data['c']}
cone = {'z': dims.zero, 'l': dims.nonneg, 'q': list(dims.soc), 'ep': dims.exp}

eps = 1e-4
rows = []

def record(method, cumulative_iters, sol):
    info = sol['info']
    prob.unpack_results(sol, chain, inverse_data)
    beta_val = beta_var.value
    glnorm = np.sum(np.linalg.norm(beta_val, axis=1)) if beta_val is not None else np.nan
    row = {
        'method': method,
        'cumulative_iters': cumulative_iters,
        'glnorm_minus_D': glnorm - D,
        'status': info['status'],
        'iter': info['iter'],
        'pobj': info['pobj'],
        'dobj': info['dobj'],
        'res_pri': info['res_pri'],
        'res_dual': info['res_dual'],
        'gap': info['gap'],
    }
    print(f"\n[{method} @ {cumulative_iters}]")
    for k, v in row.items():
        print(f"  {k}: {v}")
    rows.append(row)

# fresh-1: run 1 iter
s = scs.SCS(scs_data, cone, eps_abs=eps, eps_rel=eps, max_iters=1, verbose=False)
record('fresh', 1, s.solve())

# fresh-2: run 2 iters from scratch
s = scs.SCS(scs_data, cone, eps_abs=eps, eps_rel=eps, max_iters=2, verbose=False)
record('fresh', 2, s.solve())

# warm-1: run 1 iter
s = scs.SCS(scs_data, cone, eps_abs=eps, eps_rel=eps, max_iters=1, verbose=False)
record('warm', 1, s.solve())

# warm-2: warm_start from previous
record('warm', 2, s.solve(warm_start=True))

out = '/Users/samsonkoelle/isometry-pursuit/experiments/micro_experiment.csv'
with open(out, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"\nsaved {out}")
