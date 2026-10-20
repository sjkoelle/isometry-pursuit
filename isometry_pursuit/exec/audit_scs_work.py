"""Audit our SCS problem structure, factor work, and stopping diagnostics."""
import argparse
import json
from pathlib import Path

import cvxpy as cp
from cvxpy.reductions.solvers.conic_solvers.scs_conif import dims_to_solver_dict
import numpy as np
import scs

from isometry_pursuit.scs_work import build_audit_library, sparse_direct_work
from isometry_pursuit.transformation import exponential_transformation


DATA = {'wine': 'data/wine_standardized_resampled_tensor.npy',
        'iris': 'data/iris_standardized_resampled_tensor.npy',
        'ethanol': 'data/ethanol_tensor.npy'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scs-source', type=Path, required=True)
    parser.add_argument('--library-path', type=Path, default=Path('/tmp/isometry_scs_329_audit/libaudit.dylib'))
    parser.add_argument('--output-path', type=Path, default=Path('results/scs_work_audit.json'))
    parser.add_argument('--eps-values', type=float, nargs='+', default=[1e-12, 1e-8, 1e-6])
    parser.add_argument('--all-structures', action='store_true',
                        help='Also account for factor work of every saved 100,000-iteration tensor run')
    args = parser.parse_args()
    if scs.__version__ != '3.2.9' or scs._scs_direct.sizeof_int() != 8:
        raise ValueError('This audit expects SCS 3.2.9 with 64-bit indices')
    if '3.2.9' not in (args.scs_source/'include/glbopts.h').read_text():
        raise ValueError('Source version must match installed SCS 3.2.9')
    build_audit_library(args.scs_source, args.library_path)
    output = dict(scs_version=scs.__version__, cvxpy_version=cp.__version__,
                  source_root=str(args.scs_source),
                  counted_work='QDLDL numeric factorization and factor solves only; not total solver FLOPs',
                  multiplication_addition_division_flops_each=1,
                  datasets={})
    for name, filename in DATA.items():
        tensor = np.load(filename)
        X = exponential_transformation(tensor[:, :, 0])
        D, P = X.shape
        beta = cp.Variable((P, D))
        problem = cp.Problem(cp.Minimize(cp.sum(cp.norm(beta, axis=1))), [X@beta == np.eye(D)])
        data, chain, inverse = problem.get_problem_data(cp.SCS)
        work = sparse_direct_work(data['A'], args.library_path)
        cones = dims_to_solver_dict(data['dims'])
        entry = dict(replicate=0, D=D, P=P, structure=work, solves=[])
        output['datasets'][name] = entry
        if args.all_structures:
            source = json.loads(Path(f'results/{name}_results.json').read_text())
            structure_runs = []
            for i, record in enumerate(source['replicates']):
                Xi = exponential_transformation(tensor[:, :, i])
                bi = cp.Variable((P, D))
                pi = cp.Problem(cp.Minimize(cp.sum(cp.norm(bi, axis=1))), [Xi@bi == np.eye(D)])
                di, _, _ = pi.get_problem_data(cp.SCS)
                wi = sparse_direct_work(di['A'], args.library_path)
                steps = record['basis_pursuit_iterations']
                if steps != 100000:
                    raise ValueError('Saved-run accounting assumes the observed 100,000-step cap')
                structure_runs.append(dict(
                    replicate=i, recorded_iterations=steps, **wi,
                    counted_factor_work=wi['factorization_flops']+(steps+1)*wi['factor_solve_flops']))
            entry['saved_run_factor_work'] = structure_runs
        for eps in args.eps_values:
            solver = scs.SCS({key: data[key] for key in ('A', 'b', 'c')}, cones,
                             eps_abs=eps, eps_rel=eps, acceleration_lookback=0,
                             adaptive_scale=False, verbose=False)
            raw = solver.solve(warm_start=False)
            problem.unpack_results(raw, chain, inverse)
            info = raw['info']
            coefficients = beta.value.copy()
            coefficients[np.abs(coefficients) < 1e-6] = 0
            support = np.where(np.linalg.norm(coefficients, axis=1) > 1e-4)[0].tolist()
            existing = json.loads(Path(f'results/{name}_results.json').read_text())['replicates'][0]
            # One initialization solve for g, then one factor solve per DR step.
            # At the max-iteration cap the loop executes exactly max_iters steps.
            # Earlier convergence reports the zero-based final step index.
            steps = info['iter'] if 'max_iters' in info['status'] else info['iter'] + 1
            solve = dict(eps=eps, status=info['status'], iterations=info['iter'],
                         primal_residual=info['res_pri'], dual_residual=info['res_dual'],
                         duality_gap=info['gap'], objective=info['pobj'],
                         setup_ms=info['setup_time'], solve_ms=info['solve_time'],
                         linear_system_ms=info['lin_sys_time'], cone_ms=info['cone_time'],
                         support=support, support_matches_saved=set(support)==set(existing['basis_pursuit_indices']),
                         reconstruction_fro=float(np.linalg.norm(X@coefficients-np.eye(D))),
                         dr_steps=steps, factor_solve_calls=steps+1,
                         counted_factor_work=work['factorization_flops']+(steps+1)*work['factor_solve_flops'])
            entry['solves'].append(solve)
            args.output_path.parent.mkdir(parents=True, exist_ok=True)
            args.output_path.write_text(json.dumps(output, indent=2)+'\n')
            print(name, eps, solve, flush=True)


if __name__ == '__main__':
    main()
