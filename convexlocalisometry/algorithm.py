import numpy as np
from itertools import combinations


def greedy(matrix, target_dimension, loss, selected_indices=[]):

    dictionary_dimension = matrix.shape[0]
    parametrizations = combinations(dictionary_dimension, target_dimension)
    matrix  # can use recursion?

    if target_dimension > 0:
        for d in range(dictionary_dimension):
            if d not in selected_indices:
                loss_ = loss(matrix[[selected_indices, d]])
        most_isometric_index = np.nanargmin(loss_)
        selected_indices.append(most_isometric_index)
        selected_indices = greedy(
            matrix, target_dimension - d, loss, ignore_elements=selected_indices
        )
        return selected_indices
    else:
        return None


def brute(matrix, target_dimension, loss):

    dictionary_dimension = matrix.shape[0]
    parametrizations = combinations(dictionary_dimension, target_dimension)
    losses = np.asarray([])
    for parametrization in parametrizations:
        losses.append(loss(matrix[parametrization]))
    return parametrization[losses.argmin()]


import cvxpy as cp


def group_basis_pursuit(
    matrix,
    eps=1e-12,
    threshold=1e-6,
):
    p, d = matrix.shape
    beta = cp.Variable((p, d))  # could initialize with lasso?
    objective = cp.Minimize(cp.sum(cp.norm(beta, axis=1)))
    constraints = [matrix @ beta == np.identity(d)]
    problem = cp.Problem(objective, constraints)
    scs_opts = {"eps": eps}
    output = problem.solve(solver=cp.SCS, **scs_opts)
    if output is np.inf:
        raise ValueError("No solution found")
    beta_optimized = beta.value
    beta_sparse = beta_optimized.copy()
    beta_sparse[np.abs(beta_sparse) < threshold] = 0
    # assert entire rows are 0 or raise warning
    return beta_sparse


# # what is rank of retained functions?
# import numpy as np
# from einops import rearrange
# import cvxpy as cp
# import matplotlib.pyplot as plt
# import seaborn as sns

# print('generating multitask sample data')
# np.random.seed(42)
# n = 1
# d = 4
# p = 30
# sample_grads = .1* np.random.multivariate_normal(np.zeros(d), np.identity(d),p)
# X = rearrange(sample_grads, 'p d -> d p')
# y= np.identity(d)

# print('computing optimizer of full problem')
# beta = cp.Variable((p,d))
# objective = cp.Minimize(cp.sum(cp.norm(beta, axis = 1)))
# constraints = [X @ beta == y]
# problem = cp.Problem(objective, constraints)
# # result = problem.solve(reltol=1e-14)
# scs_opts = {'eps': 1e-12}
# result = problem.solve(solver=cp.SCS, **scs_opts)
# beta_optimized = beta.value
# print('reconstruction constraint and loss', np.linalg.norm(y - X @ beta_optimized ), np.sum(np.linalg.norm(beta_optimized, axis = 1)))

# print('many coefficients are close to 0')
# plt.hist(beta_optimized)
# plt.xscale('symlog')
# plt.title('Coefficients')

# plt.figure()
# print('sparsity is shared across tasks')
# sns.heatmap(low_indices)

# print('enforece sparsity following I selected a number here following https://github.com/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/sparse_solution.ipynb')
# low_indices = np.abs(beta_optimized) < 1e-6
# beta_sparse = beta_optimized.copy()
# beta_sparse[np.abs(beta_sparse) < 1e-6] = 0

# print('sparse reconstruction constraint and loss', np.linalg.norm(y - X @ beta_sparse ), np.sum(np.linalg.norm(beta_sparse, axis = 1)))

# print('refitting with only retained coefficients')
# X_restricted = X[:,~low_indices[:,0]]
# nonzerorows = len(np.where(~low_indices[:,0])[0])
# beta = cp.Variable((nonzerorows,d)) # number of non-zero rows


# objective = cp.Minimize(cp.sum(cp.norm(beta, axis = 1)))
# constraints = [X_restricted @ beta == y]
# problem = cp.Problem(objective, constraints)
# result = problem.solve()
# beta_restricted = beta.value

# print('sparse refit reconstruction constraint and penalty', np.linalg.norm(y - X_restricted @ beta_restricted ), np.sum(np.linalg.norm(beta_restricted, axis = 1)))
# print('it looks like the refit solution has better constraint satisfaction and lower loss than the original solution!')
