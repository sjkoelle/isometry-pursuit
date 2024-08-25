from sklearn.decomposition import PCA
import numpy as np


def local_pca(data, ind, n_neighbors, dimension):
    neighbor_indices = np.linalg.norm(data[:, :] - data[ind, :], axis=1).argsort()[
        :n_neighbors
    ]
    local_neighborhood = data[neighbor_indices]
    local_neighborhood = data[neighbor_indices, :]
    centered_data = local_neighborhood - local_neighborhood.mean(axis=0)

    # Perform PCA on the centered neighborhood data
    pca = PCA(n_components=dimension)
    pca.fit(centered_data)
    tangent_bases = pca.components_.transpose()
    return tangent_bases
