import numpy as np


def isometry_loss(matrix: np.ndarray) -> float:

    singular_values = np.linalg.svd(matrix, compute_uv=False)
    squared_loss = np.linalg.norm(np.log(singular_values)) ** 2
    return squared_loss
