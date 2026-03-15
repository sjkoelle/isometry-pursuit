"""Check that warm-started chunks give the same result as fresh restarts at each N."""

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

chunk = 10
n_checkpoints = 10
eps = 1e-20  # never converge early

# --- warm-started: run chunk iters at a time, record glnorm at each checkpoint ---
s_warm = scs.SCS(scs_data, cone, eps_abs=eps, eps_rel=eps, max_iters=chunk, verbose=False)
glnorms_warm = []
for i in range(n_checkpoints):
    sol = s_warm.solve(warm_start=(i > 0))
    prob.unpack_results(sol, chain, inverse_data)
    glnorms_warm.append(np.sum(np.linalg.norm(beta_var.value, axis=1)))

# --- fresh restarts: for each N, run from scratch ---
glnorms_fresh = []
for i in range(1, n_checkpoints + 1):
    N = i * chunk
    s_fresh = scs.SCS(scs_data, cone, eps_abs=eps, eps_rel=eps, max_iters=N, verbose=False)
    sol = s_fresh.solve()
    prob.unpack_results(sol, chain, inverse_data)
    glnorms_fresh.append(np.sum(np.linalg.norm(beta_var.value, axis=1)))

# --- comparison table ---
print(f"{'N':>6}  {'warm':>12}  {'fresh':>12}  {'diff':>10}  match")
for i in range(n_checkpoints):
    N = (i + 1) * chunk
    diff = abs(glnorms_warm[i] - glnorms_fresh[i])
    match = np.isclose(glnorms_warm[i], glnorms_fresh[i], atol=1e-6)
    print(f"{N:>6}  {glnorms_warm[i]:>12.6f}  {glnorms_fresh[i]:>12.6f}  {diff:>10.2e}  {match}")
