import numpy as np


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
