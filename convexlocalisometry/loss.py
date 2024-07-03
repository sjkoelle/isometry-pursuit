import numpy as np


def isometry_loss(matrix: np.ndarray) -> float:

    singular_values = np.linalg.svd(matrix, compute_uv=False)
    squared_loss = np.linalg.norm(np.log(singular_values)) ** 2
    return squared_loss


def subspace_loss(matrix_1: np.ndarray, matrix_2: np.ndarray) -> float:
    """Computes the frobenius norm of the projection of the two matrices"""
    output = np.linalg.norm(np.einsum("ij, jk -> ik", matrix_1, matrix_2)) ** 2
    return output


def group_basis_pursuit_loss(beta):  # beta_bp
    """Computes the group basis pursuit loss of the matrix beta"""
    output = np.linalg.norm(beta, axis=1).sum()

    return output
