"""Check whether 2x100 warm-started == 1x200 fresh at N=200."""

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

eps = 1e-20

# fresh 200
s1 = scs.SCS(scs_data, cone, eps_abs=eps, eps_rel=eps, max_iters=200, verbose=False)
sol1 = s1.solve()
prob.unpack_results(sol1, chain, inverse_data)
beta_200 = beta_var.value.copy()
glnorm_200 = np.sum(np.linalg.norm(beta_200, axis=1))

# 2x100 warm-start
s2 = scs.SCS(scs_data, cone, eps_abs=eps, eps_rel=eps, max_iters=100, verbose=False)
s2.solve()
sol2 = s2.solve(warm_start=True)
prob.unpack_results(sol2, chain, inverse_data)
beta_2x100 = beta_var.value.copy()
glnorm_2x100 = np.sum(np.linalg.norm(beta_2x100, axis=1))

print(f"fresh 200     glnorm: {glnorm_200:.8f}")
print(f"2x100 warm    glnorm: {glnorm_2x100:.8f}")
print(f"max abs diff:         {np.max(np.abs(beta_200 - beta_2x100)):.2e}")
print(f"match (atol=1e-6):    {np.allclose(beta_200, beta_2x100, atol=1e-6)}")
