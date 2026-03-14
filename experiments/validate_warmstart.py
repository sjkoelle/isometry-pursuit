"""Check that warm-started chunks give the same result as one full run."""

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

total = 100
chunk = 10
eps = 1e-20  # never converge early

# --- single run ---
s1 = scs.SCS(scs_data, cone, eps_abs=eps, eps_rel=eps, max_iters=total, verbose=False)
sol1 = s1.solve()
prob.unpack_results(sol1, chain, inverse_data)
beta_single = beta_var.value.copy()
glnorm_single = np.sum(np.linalg.norm(beta_single, axis=1))

# --- warm-started chunks ---
s2 = scs.SCS(scs_data, cone, eps_abs=eps, eps_rel=eps, max_iters=chunk, verbose=False)
sol2 = s2.solve()
for _ in range(total // chunk - 1):
    sol2 = s2.solve(warm_start=True)
prob.unpack_results(sol2, chain, inverse_data)
beta_chunked = beta_var.value.copy()
glnorm_chunked = np.sum(np.linalg.norm(beta_chunked, axis=1))

print(f"single  run  glnorm: {glnorm_single:.8f}")
print(f"chunked run  glnorm: {glnorm_chunked:.8f}")
print(f"beta max abs diff:   {np.max(np.abs(beta_single - beta_chunked)):.2e}")
print(f"match: {np.allclose(beta_single, beta_chunked, atol=1e-6)}")
