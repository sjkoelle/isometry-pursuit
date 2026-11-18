import numpy as np
from scipy.stats import ortho_group


def _random_orthonormal_block(ambient_dimension, block_dimension, rng):
    if block_dimension == ambient_dimension:
        return ortho_group.rvs(ambient_dimension, random_state=rng)
    basis = np.zeros((ambient_dimension, block_dimension))
    if block_dimension == 1:
        basis[0, 0] = rng.choice([-1.0, 1.0])
    else:
        basis[:block_dimension, :] = ortho_group.rvs(block_dimension, random_state=rng)
    rotation = ortho_group.rvs(ambient_dimension, random_state=rng)
    return rotation @ basis


def generate_identity_instance(dimension, corrupting_dimension, num_corrupting, rng):
    """Generate the planted identity dictionary used by the synthetic experiments."""
    blocks = [np.eye(dimension)]
    block_ranges = [{"kind": "true", "dimension": dimension, "start": 0, "stop": dimension}]
    start = dimension
    for block_index in range(num_corrupting):
        blocks.append(_random_orthonormal_block(dimension, corrupting_dimension, rng))
        block_ranges.append({
            "kind": "corrupting", "dimension": corrupting_dimension,
            "block_index": block_index, "start": start, "stop": start + corrupting_dimension,
        })
        start += corrupting_dimension
    matrix = np.column_stack(blocks)
    return matrix, list(range(dimension)), list(range(dimension, matrix.shape[1])), block_ranges


def simulate_unitary_matrix(ambient_dimension, unitary_dimension, noise_dimension):

    dictionary_dimension = unitary_dimension + noise_dimension
    output = np.zeros((dictionary_dimension, ambient_dimension))
    # could also randomly rotate and place in dictionary_dimension[:unitary_dimension,:]
    output[:unitary_dimension, :unitary_dimension] = np.identity(unitary_dimension)
    # noise_dimension =
    # dictionary_dimension[unitary_dimension:,:] = np.random.multivariate_normal(noise_dimension, (np.zeros(ambient_dimension)))
    # can sample randomly on sphere for quite challenging example
    # why not just zeros to start, can add noise later
    return output
