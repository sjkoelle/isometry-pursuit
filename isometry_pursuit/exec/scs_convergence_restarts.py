"""Plot group lasso norm vs iterations using fresh restarts at each checkpoint."""

import numpy as np
import cvxpy as cp
import scs
import matplotlib.pyplot as plt
import click

from isometry_pursuit.transformation import exponential_transformation


@click.command()
@click.option('--adaptive-scale/--no-adaptive-scale', default=True, show_default=True)
@click.option('--acceleration-lookback', default=0, show_default=True, type=int)
@click.option('--normalize/--no-normalize', default=True, show_default=True)
@click.option('--indirect/--direct', default=False, show_default=True)
def main(adaptive_scale, acceleration_lookback, normalize, indirect):
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

    checkpoints = np.arange(100, 10001, 100)
    eps_configs = [(1e-4, 'eps=1e-4'), (1e-12, 'eps=1e-12'), (1e-20, 'eps=1e-20')]

    saved = {'iters': checkpoints}
    _fig, ax = plt.subplots(figsize=(12, 5))
    for eps_val, label in eps_configs:
        glnorms = []
        for n_iters in checkpoints:
            s = scs.SCS(scs_data, cone, eps_abs=eps_val, eps_rel=eps_val,
                        max_iters=int(n_iters), verbose=False,
                        acceleration_lookback=acceleration_lookback,
                        adaptive_scale=adaptive_scale,
                        normalize=normalize,
                        use_indirect=indirect)
            sol = s.solve()
            prob.unpack_results(sol, chain, inverse_data)
            beta_val = beta_var.value
            glnorm = np.sum(np.linalg.norm(beta_val, axis=1)) if beta_val is not None else np.nan
            glnorms.append(max(glnorm - D, 1e-10))
        saved[label] = np.array(glnorms)
        ax.plot(checkpoints, glnorms, 'o-', ms=3, label=label)
        print(f"{label} done")

    np.savez('/Users/samsonkoelle/isometry-pursuit/experiments/glnorms_restarts.npz', **saved)
    print("saved glnorms_restarts.npz")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('max_iters (fresh restart each time)')
    ax.set_ylabel('group lasso norm − D')
    ax.set_title(f'Convergence: group lasso norm − D={D} (rep 1, fresh restarts)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('/Users/samsonkoelle/isometry-pursuit/experiments/scs_glnorm_restarts.png', dpi=150)
    plt.close()
    print("saved scs_glnorm_restarts.png")


if __name__ == '__main__':
    main()
