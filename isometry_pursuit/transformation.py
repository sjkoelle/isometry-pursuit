import numpy as np
import warnings


# NOTE (Sam): why not use the exponential transformation after fitting the coefficients?
def exponential_transformation(X, power=1):
    """
    Applies the transformation to each row of the input matrix X.

    Parameters:
    X (numpy.ndarray): A 2D array of shape (D, P)

    Returns:
    numpy.ndarray: Transformed matrix of the same shape (D, P)
    """
    # Compute the L2 norm of each column
    norms = np.linalg.norm(X, axis=0)
    normalized_X = X / norms[np.newaxis, :]

    norms = norms**power

    # Log relevant arguments when overflow occurs
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        exp_values = ((np.exp(norms) + np.exp(norms ** (-1))) ** (-1)) * 2 * np.e

        # Check if we got any overflow warnings
        for warning in w:
            if "overflow" in str(warning.message).lower():
                print(f"Exponential transformation overflow warning: {warning.message}")
                print(f"norms shape: {norms.shape}")
                print(f"norms min: {np.min(norms)}, max: {np.max(norms)}")
                print(
                    f"norms^(-1) min: {np.min(norms**(-1))}, max: {np.max(norms**(-1))}"
                )
                print(
                    f"exp(norms) min: {np.min(np.exp(norms))}, max: {np.max(np.exp(norms))}"
                )
                print(
                    f"exp(norms^(-1)) min: {np.min(np.exp(norms**(-1)))}, max: {np.max(np.exp(norms**(-1)))}"
                )
                print(f"X shape: {X.shape}, power: {power}")
                break
    # (np.exp(t) + np.exp(t**(-1)))**(-1) * (2*math.e)
    # Normalize columns of X

    # Apply the transformation using einsum
    transformed_X = np.einsum("j,ij->ij", exp_values, normalized_X)

    return transformed_X


def g_broadcast_transformation(X):
    """
    Applies the inverse broadcast transformation g^{×P}(X) to the input matrix.

    This uses q^{-1} (inverse of column norms) instead of q, providing direct
    scaling without requiring a reciprocal loss.

    Parameters:
    X (numpy.ndarray): A 2D array of shape (D, P)

    Returns:
    numpy.ndarray: Transformed matrix g^{×P}(X) of the same shape (D, P)
    """
    # Compute the L2 norm of each column
    norms = np.linalg.norm(X, axis=0)
    normalized_X = X / norms[np.newaxis, :]

    # Use inverse of norms (q^{-1})
    inverse_norms = norms ** (-1)

    # Compute scaling factors based on inverse norms
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        exp_values = ((np.exp(inverse_norms) + np.exp(inverse_norms ** (-1))) ** (-1)) * 2 * np.e

        for warning in w:
            if "overflow" in str(warning.message).lower():
                print(f"g_broadcast_transformation overflow warning: {warning.message}")
                print(f"inverse_norms shape: {inverse_norms.shape}")
                print(f"inverse_norms min: {np.min(inverse_norms)}, max: {np.max(inverse_norms)}")
                break

    # Apply the transformation using einsum
    transformed_X = np.einsum("j,ij->ij", exp_values, normalized_X)

    return transformed_X
