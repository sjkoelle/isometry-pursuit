"""Paired optimizer comparisons on already sampled matrices.

SCS scale and direct ADMM rho are separate parameters. Random seeds initialize
the optimization variables; they never change the input matrix.
"""

import math
import time

import cvxpy as cp
import numpy as np
import scs
from scipy.linalg import cho_factor, cho_solve

from .transformation import exponential_transformation


def _admm_gap_check(matrix, sparse, feasible, correction, normal_shift, rho, absolute, relative):
    """Check original-problem feasibility and a feasible-primal/dual gap."""
    dimension = matrix.shape[0]
    dual_norm = rho * float(np.max(np.linalg.norm(normal_shift, axis=1)))
    scale = max(1., dual_norm)
    lower = float(rho * np.trace(correction) / scale)
    upper = float(np.linalg.norm(feasible, axis=1).sum())
    returned = float(np.linalg.norm(sparse, axis=1).sum())
    residual = float(np.linalg.norm(matrix @ sparse - np.eye(dimension)))
    projection_residual = float(np.linalg.norm(matrix @ feasible - np.eye(dimension)))
    feasibility_tolerance = absolute + relative * math.sqrt(dimension)
    gap_tolerance = absolute + relative * max(1., abs(upper), abs(lower))
    gap = upper - lower
    information = dict(dual_objective=lower, feasible_primal_objective=upper,
                       returned_primal_objective=returned, primal_dual_gap=gap,
                       returned_primal_dual_gap=returned-lower,
                       feasibility_residual_frobenius=residual,
                       projection_residual_frobenius=projection_residual,
                       feasibility_tolerance=feasibility_tolerance,
                       gap_tolerance=gap_tolerance, dual_max_row_norm=dual_norm/scale)
    passed = (residual <= feasibility_tolerance and projection_residual <= feasibility_tolerance
              and abs(gap) <= gap_tolerance and abs(returned-lower) <= gap_tolerance)
    return passed, information


def solve_matrix(matrix, settings, time_limit=100):
    """Return raw coefficients and diagnostics, before applying support thresholds."""
    started = time.perf_counter()
    dimension, columns = matrix.shape
    rng = np.random.default_rng(settings.get("seed", 0))
    initialization = settings.get("initialization", "zero")
    if initialization not in ("zero", "random"):
        raise ValueError("initialization must be zero or random")
    absolute = float(settings.get("eps_abs", 1e-8))
    relative = float(settings.get("eps_rel", 1e-8))
    iterations = int(settings.get("max_iters", 100000))
    if absolute <= 0 or relative <= 0 or iterations < 1:
        raise ValueError("Solver tolerances and iteration limits must be positive")
    if settings["solver"] == "scs" and settings.get("implementation") == "repository":
        from .algorithm import group_basis_pursuit

        if initialization != "zero" or float(settings.get("scale", 0.1)) != 0.1:
            raise ValueError("Repository baseline uses its original initialization and scale")
        coefficients, count, objective = group_basis_pursuit(
            matrix, steps=iterations, threshold=float(settings.get("entry_threshold", 1e-6)))
        result = {"beta": coefficients, "status": "repository_result",
                  "basis_pursuit_iterations": count,
                  "returned_coefficients_thresholded": True,
                  "solver_info": {"implementation": "group_basis_pursuit",
                                  "stopping_tolerances": "SCS defaults when steps is set",
                                  "returned_objective": float(objective)}}
    elif settings["solver"] == "scs":
        from cvxpy.reductions.solvers.conic_solvers.scs_conif import dims_to_solver_dict

        if float(settings.get("scale", 0.1)) <= 0:
            raise ValueError("SCS scale must be positive")
        beta = cp.Variable((columns, dimension))
        problem = cp.Problem(cp.Minimize(cp.sum(cp.norm(beta, axis=1))),
                             [matrix @ beta == np.eye(dimension)])
        data, chain, inverse = problem.get_problem_data(cp.SCS)
        inputs = {key: data[key] for key in ("A", "b", "c")}
        if "P" in data:
            inputs["P"] = data["P"]
        solver = scs.SCS(
            inputs, dims_to_solver_dict(data["dims"]),
            eps_abs=absolute, eps_rel=relative, max_iters=iterations,
            scale=float(settings.get("scale", 0.1)),
            normalize=bool(settings.get("normalize", True)),
            rho_x=float(settings.get("rho_x", 1e-6)),
            adaptive_scale=False, acceleration_lookback=0,
            time_limit_secs=float(time_limit), verbose=False,
        )
        initial = {}
        if initialization == "random":
            initial = {"x": rng.normal(scale=0.01, size=data["A"].shape[1]),
                       "y": rng.normal(scale=0.01, size=data["A"].shape[0]),
                       "s": rng.normal(scale=0.01, size=data["A"].shape[0])}
        solution = solver.solve(warm_start=bool(initial), **initial)
        problem.unpack_results(solution, chain, inverse)
        if beta.value is None:
            raise ValueError(f"SCS returned no coefficients: {solution['info']['status']}")
        result = {"beta": beta.value, "status": solution["info"]["status"],
                  "basis_pursuit_iterations": int(solution["info"]["iter"]),
                  "solver_info": {k: solution["info"].get(k) for k in
                                  ("res_pri", "res_dual", "gap", "pobj", "dobj", "status_val")}}
    elif settings["solver"] == "admm":
        rho = float(settings.get("rho", 1.0))
        if rho <= 0:
            raise ValueError("rho must be positive")
        stopping = settings.get('stopping', 'residual')
        if stopping not in ('residual', 'primal_dual'):
            raise ValueError('Unknown ADMM stopping rule')
        check_interval = int(settings.get('gap_check_interval', 25))
        if check_interval < 1:
            raise ValueError('Gap check interval must be positive')
        gap_information = {}
        factor = cho_factor(matrix @ matrix.T)
        identity = np.eye(dimension)
        z = np.zeros((columns, dimension))
        if initialization == "random":
            z = rng.normal(scale=0.01, size=z.shape)
        dual_variable = np.zeros_like(z)
        status = "max_iters"
        for step in range(iterations):
            previous = z.copy()
            value = z - dual_variable
            correction = cho_solve(factor, identity - matrix @ value)
            normal_shift = matrix.T @ correction
            feasible = value + normal_shift
            value = feasible + dual_variable
            lengths = np.linalg.norm(value, axis=1)
            z = np.maximum(0, 1 - 1 / (rho * np.maximum(lengths, 1e-300)))[:, None] * value
            dual_variable += feasible - z
            primal = np.linalg.norm(feasible - z)
            dual = rho * np.linalg.norm(z - previous)
            primal_tolerance = math.sqrt(z.size) * absolute + relative * max(np.linalg.norm(z), np.linalg.norm(feasible))
            dual_tolerance = math.sqrt(z.size) * absolute + relative * rho * np.linalg.norm(dual_variable)
            if stopping == 'primal_dual':
                if step % check_interval == 0 or step == iterations-1:
                    passed, gap_information = _admm_gap_check(
                        matrix, z, feasible, correction, normal_shift, rho, absolute, relative)
                    if passed:
                        status = 'solved'
                        break
            elif primal <= primal_tolerance and dual <= dual_tolerance:
                status = 'solved'
                break
            if step % 25 == 0 and time.perf_counter() - started >= time_limit:
                status = "time_limit"
                break
        result = {"beta": z, "status": status, "basis_pursuit_iterations": step + 1,
                  "solver_info": {"res_pri": float(primal), "res_dual": float(dual),
                                  "primal_tolerance": float(primal_tolerance),
                                  "dual_tolerance": float(dual_tolerance),
                                  "stopping": stopping, **gap_information}}
    else:
        raise ValueError("solver must be scs or admm")
    result["basis_pursuit_time_seconds"] = time.perf_counter() - started
    return result


def optimize_sample(matrix, settings, searches=True, time_limit=100):
    """Use the paper's normalization and thresholds; retain the raw solution too."""
    from .algorithm import greedy, forward_backward, greedy_swap
    from .experiment import _gl_loss

    normalized = exponential_transformation(matrix)
    result = solve_matrix(normalized, settings, time_limit=time_limit)
    raw = result.pop("beta")
    if not np.all(np.isfinite(raw)):
        raise ValueError("Optimizer returned nonfinite coefficients")
    thresholded = raw.copy()
    thresholded[np.abs(thresholded) < float(settings.get("entry_threshold", 1e-6))] = 0
    indices = np.flatnonzero(np.linalg.norm(thresholded, axis=1) > float(settings.get("row_threshold", 1e-4)))
    result.update({
        "objective_value": float(np.linalg.norm(raw, axis=1).sum()),
        "thresholded_objective_value": float(np.linalg.norm(thresholded, axis=1).sum()),
        "frobenius_norm_squared": float(np.square(raw).sum()),
        "feasibility_residual_frobenius": float(np.linalg.norm(normalized @ raw - np.eye(matrix.shape[0]))),
        "basis_pursuit_indices": indices.tolist(), "support_cardinality": len(indices),
        "group_lasso_norm": float(_gl_loss(normalized[:, indices])) if len(indices) >= matrix.shape[0] else None,
        "row_norms": np.linalg.norm(raw, axis=1).tolist(),
    })
    if searches:
        search_matrix = matrix if settings.get("search_on_raw", False) else normalized
        for initialization in ("forward", "backward"):
            for restricted in (False, True):
                prefix = initialization + ("_restricted" if restricted else "_full")
                candidates = indices if restricted else np.arange(matrix.shape[1])
                if len(candidates) < matrix.shape[0]:
                    result[prefix + "_status"] = "insufficient_support"
                    continue
                # Backward search is quadratic in dictionary size. Record omissions.
                if len(candidates) > int(settings.get("max_search_columns", 128)):
                    result[prefix + "_status"] = "column_limit"
                    continue
                started = time.perf_counter()
                search_seed = int(settings.get("search_seed", 1234))
                if initialization == "forward":
                    initial = greedy(search_matrix[:, candidates], _gl_loss, random_state=search_seed)
                    result[prefix + "_initial_indices"] = candidates[np.asarray(initial, dtype=int)].tolist()
                    result[prefix + "_initial_loss"] = float(_gl_loss(search_matrix[:, candidates[np.asarray(initial, dtype=int)]]))
                    selected = forward_backward(search_matrix[:, candidates], _gl_loss,
                                                selected_indices=initial, random_state=search_seed)
                else:
                    selected = greedy_swap(search_matrix[:, candidates], _gl_loss,
                                           initialization=initialization, random_state=search_seed)
                selected = candidates[np.asarray(selected, dtype=int)]
                result[prefix + "_indices"] = selected.tolist()
                result[prefix + "_loss"] = float(_gl_loss(search_matrix[:, selected]))
                result[prefix + "_runtime_seconds"] = time.perf_counter() - started
                result[prefix + "_status"] = "complete"
    return raw, result
