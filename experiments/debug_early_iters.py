"""Fine-grained glnorm tracking at N=1..200 for eps=1e-4: fresh restart vs warm-start."""

import numpy as np
import cvxpy as cp
import scs
import matplotlib.pyplot as plt

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
iters = list(range(1, 201))

# --- fresh restarts: N=1..200 ---
glnorms_fresh = []
for N in iters:
    s = scs.SCS(scs_data, cone, eps_abs=eps, eps_rel=eps, max_iters=N, verbose=False)
    sol = s.solve()
    prob.unpack_results(sol, chain, inverse_data)
    beta_val = beta_var.value
    glnorm = np.sum(np.linalg.norm(beta_val, axis=1)) if beta_val is not None else np.nan
    glnorms_fresh.append(glnorm - D)
    if sol['info']['status'] != 'unfinished':
        print(f"  fresh N={N}: converged ({sol['info']['status']}), glnorm-D={glnorm-D:.4f}")
print("fresh restarts done")

# --- warm-start chunk=1: each call does 1 iter ---
s_w1 = scs.SCS(scs_data, cone, eps_abs=eps, eps_rel=eps, max_iters=1, verbose=False)
glnorms_warm1 = []
for i, N in enumerate(iters):
    sol = s_w1.solve(warm_start=(i > 0))
    prob.unpack_results(sol, chain, inverse_data)
    beta_val = beta_var.value
    glnorm = np.sum(np.linalg.norm(beta_val, axis=1)) if beta_val is not None else np.nan
    glnorms_warm1.append(glnorm - D)
print("warm-start chunk=1 done")

# --- warm-start chunk=10 ---
s_w10 = scs.SCS(scs_data, cone, eps_abs=eps, eps_rel=eps, max_iters=10, verbose=False)
glnorms_warm10, iters_w10 = [], []
for i in range(20):  # 20*10 = 200
    sol = s_w10.solve(warm_start=(i > 0))
    prob.unpack_results(sol, chain, inverse_data)
    beta_val = beta_var.value
    glnorm = np.sum(np.linalg.norm(beta_val, axis=1)) if beta_val is not None else np.nan
    glnorms_warm10.append(glnorm - D)
    iters_w10.append((i + 1) * 10)
print("warm-start chunk=10 done")

# --- warm-start chunk=100 ---
s_w100 = scs.SCS(scs_data, cone, eps_abs=eps, eps_rel=eps, max_iters=100, verbose=False)
glnorms_warm100, iters_w100 = [], []
for i in range(2):  # 2*100 = 200
    sol = s_w100.solve(warm_start=(i > 0))
    prob.unpack_results(sol, chain, inverse_data)
    beta_val = beta_var.value
    glnorm = np.sum(np.linalg.norm(beta_val, axis=1)) if beta_val is not None else np.nan
    glnorms_warm100.append(glnorm - D)
    iters_w100.append((i + 1) * 100)
    print(f"  warm chunk=100 iter={iters_w100[-1]}: status={sol['info']['status']}, glnorm-D={glnorm-D:.4f}")
print("warm-start chunk=100 done")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(iters, glnorms_fresh,  '-',  lw=1,   label='fresh restart')
ax.plot(iters, glnorms_warm1,  '-',  lw=1,   label='warm chunk=1')
ax.plot(iters_w10,  glnorms_warm10,  'o-', ms=4, label='warm chunk=10')
ax.plot(iters_w100, glnorms_warm100, 's-', ms=6, label='warm chunk=100')
ax.set_yscale('log')
ax.set_xlabel('iterations')
ax.set_ylabel('glnorm − D')
ax.set_title(f'eps=1e-4: glnorm−D vs iterations (N=1..200)')
ax.legend()
plt.tight_layout()
plt.savefig('/Users/samsonkoelle/isometry-pursuit/experiments/debug_early_iters.png', dpi=150)
plt.close()
print("saved debug_early_iters.png")
