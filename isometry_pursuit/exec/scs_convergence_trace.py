"""Plot SCS primal objective over iterations for a single optimization run."""

import numpy as np
import pandas as pd
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
    r = 1
    X = data_tensor[:, :, r]
    X_t = exponential_transformation(X)
    D, P = X_t.shape

    beta_var = cp.Variable((P, D))
    prob = cp.Problem(
        cp.Minimize(cp.sum(cp.norm(beta_var, axis=1))),
        [X_t @ beta_var == np.eye(D)]
    )

    data, chain, inverse_data = prob.get_problem_data(cp.SCS)
    dims = data['dims']

    scs_data = {
        'A': data['A'],
        'b': data['b'],
        'c': data['c'],
    }
    cone = {
        'z': dims.zero,
        'l': dims.nonneg,
        'q': list(dims.soc),
        'ep': dims.exp,
    }

    trace_path = '/Users/samsonkoelle/isometry-pursuit/experiments/scs_trace.csv'

    solver = scs.SCS(
        scs_data,
        cone,
        eps_abs=1e-4,
        eps_rel=1e-4,
        max_iters=100000,
        verbose=True,
        log_csv_filename=trace_path,
        acceleration_lookback=acceleration_lookback,
        adaptive_scale=adaptive_scale,
        normalize=normalize,
        use_indirect=indirect,
    )

    sol = solver.solve()
    print(sol['info'])

    trace = pd.read_csv(trace_path)

    _fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(trace['iter'], trace['pobj'], '-', label='primal')
    axes[0].plot(trace['iter'], trace['dobj'], '-', label='dual')
    axes[0].legend()
    axes[0].set_xlabel('iteration'); axes[0].set_ylabel('objective')
    axes[0].set_title('Primal & dual obj vs iteration (rep 1)')
    axes[0].set_ylim(0, trace['pobj'].clip(lower=0).max() * 1.2)

    axes[1].plot(trace['iter'], trace['gap'], '-')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('iteration'); axes[1].set_ylabel('gap')
    axes[1].set_title('Duality gap vs iteration')

    axes[2].plot(trace['iter'], trace['res_pri'], '-', label='pri res')
    axes[2].plot(trace['iter'], trace['res_dual'], '-', label='dua res')
    axes[2].set_yscale('log'); axes[2].legend()
    axes[2].set_xlabel('iteration'); axes[2].set_ylabel('residual')
    axes[2].set_title('SCS residuals vs iteration')

    plt.tight_layout()
    plt.savefig('/Users/samsonkoelle/isometry-pursuit/experiments/scs_convergence.png', dpi=150)
    plt.close()

    # --- group lasso norm - D: warm-started runs, one per eps ---
    delta = 100
    checkpoints = np.arange(100, 10001, 100)
    n_checkpoints = len(checkpoints)

    eps_configs = [(1e-4, 'eps=1e-4'), (1e-12, 'eps=1e-12'), (1e-20, 'eps=1e-20')]

    saved = {'iters': checkpoints}
    _fig2, ax = plt.subplots(figsize=(12, 5))
    for eps_val, label in eps_configs:
        s = scs.SCS(scs_data, cone, eps_abs=eps_val, eps_rel=eps_val,
                    max_iters=delta, verbose=False,
                    acceleration_lookback=acceleration_lookback,
                    adaptive_scale=adaptive_scale,
                    normalize=normalize,
                    use_indirect=indirect)
        glnorms = []
        for i in range(n_checkpoints):
            sol_cp = s.solve(warm_start=(i > 0))
            prob.unpack_results(sol_cp, chain, inverse_data)
            beta_val = beta_var.value
            glnorm = np.sum(np.linalg.norm(beta_val, axis=1)) if beta_val is not None else np.nan
            glnorms.append(max(glnorm - D, 1e-10))
        saved[label] = np.array(glnorms)
        ax.plot(checkpoints, glnorms, 'o-', ms=3, label=label)
        print(f"{label} done")

    np.savez('/Users/samsonkoelle/isometry-pursuit/experiments/glnorms_warmstart.npz', **saved)
    print("saved glnorms_warmstart.npz")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('cumulative iters (warm-started)')
    ax.set_ylabel('group lasso norm − D')
    ax.set_title(f'Convergence: group lasso norm − D={D} (rep 1, warm-started)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('/Users/samsonkoelle/isometry-pursuit/experiments/scs_glnorm.png', dpi=150)
    plt.close()
    print("saved scs_glnorm.png")


if __name__ == '__main__':
    main()
