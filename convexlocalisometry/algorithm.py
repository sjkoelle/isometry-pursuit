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
