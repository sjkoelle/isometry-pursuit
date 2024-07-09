import numpy as np


def isometry_loss(matrix: np.ndarray, power=1) -> float:
    """is it a norm?"""
    singular_values = np.linalg.svd(
        matrix, compute_uv=False
    )  # no 0s since we are in full rankistan
    singular_values = singular_values**power
    # print(singular_values)
    # output = np.linalg.norm(np.log(singular_values)) ** 2
    # (np.exp(t) + np.exp(1/t) )/ (2 * math.e)
    output = np.sum(
        (np.exp(singular_values) + np.exp(singular_values ** (-1))) / (2 * np.e)
    )
    return output


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
