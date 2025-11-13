import numpy as np
<<<<<<< HEAD
=======
import warnings
>>>>>>> origin/experiment


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
<<<<<<< HEAD
    # print(norms)
    # # Compute the logarithm of the norms
    # log_norms = np.log(norms)

    # # Compute the exponential of the negative absolute value of the log norms
    # exp_values = np.exp(-np.abs(log_norms))

    exp_values = ((np.exp(norms) + np.exp(norms ** (-1))) ** (-1)) * 2 * np.e
=======

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
>>>>>>> origin/experiment
    # (np.exp(t) + np.exp(t**(-1)))**(-1) * (2*math.e)
    # Normalize columns of X

    # Apply the transformation using einsum
    transformed_X = np.einsum("j,ij->ij", exp_values, normalized_X)

    return transformed_X
