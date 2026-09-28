import numpy as np


def subspace_loss(matrix_1: np.ndarray, matrix_2: np.ndarray) -> float:
    """Computes the frobenius norm of the projection of the two matrices"""
    output = np.linalg.norm(np.einsum("ij, jk -> ik", matrix_1, matrix_2)) ** 2
    return output


def group_lasso_norm(beta):  # beta_bp
    """Computes the group basis pursuit loss of the matrix beta"""
    output = np.linalg.norm(beta, axis=1).sum()

    return output


# def pseudoinverse_basis_pursuit_loss(beta):  # beta_bp
#     """Computes the basis pursuit loss of the matrix beta"""

#     # beta = np.linalg.pinv(
#     #     X
#     # )  # do all inv have the same norm?  No... pinv is well defined (maybe min w.r.t. a norm, not necessarily 2,1))
#     # so this should be found by the algorithm
#     # in fact this is the whole point of the algorithm
#     output = np.linalg.norm(beta, axis=1).sum()

#     return output
