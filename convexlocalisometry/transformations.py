import numpy as np


# NOTE (Sam): why not use the exponential transformation after fitting the coefficients?
def exponential_transformation(X):
    """
    Applies the transformation to each row of the input matrix X.

    Parameters:
    X (numpy.ndarray): A 2D array of shape (D, P)

    Returns:
    numpy.ndarray: Transformed matrix of the same shape (D, P)
    """
    # Compute the L2 norm of each column
    norms = np.linalg.norm(X, axis=0)

    # # Compute the logarithm of the norms
    # log_norms = np.log(norms)

    # # Compute the exponential of the negative absolute value of the log norms
    # exp_values = np.exp(-np.abs(log_norms))

    exp_values = (np.exp(norms) + np.exp(norms**(-1)))**(-1)
    # Normalize columns of X
    normalized_X = X / norms[np.newaxis, :]

    # Apply the transformation using einsum
    transformed_X = np.einsum("j,ij->ij", exp_values, normalized_X)

    return transformed_X
