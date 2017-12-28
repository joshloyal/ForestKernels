import numpy as np
import scipy.sparse as sparse

from forest_kernels import distances


X = np.array([[1, 2, 3, 4],
              [3, 2, 3, 1],
              [1, 2, 3, 4],
              [1, 10, 9, 1],
              [3, 10, 9, 1]])


# three node partition
X_binary = np.array([[1, 0, 0],
                     [1, 0, 0],
                     [0, 0, 1],
                     [0, 1, 0],
                     [0, 0, 1]])

expected_binary = np.array([[1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [0, 0, 1, 0, 1],
                            [0, 0, 0, 1, 0],
                            [0, 0, 1, 0, 1]])

X_binary_sp = sparse.csr_matrix(X_binary)


expected = np.array([[1.0, 0.5, 1, 0.25, 0.0],
                     [0.5, 1.0, 0.5, 0.25, 0.5],
                     [1, 0.5, 1.0, 0.25, 0.0],
                     [0.25, 0.25, 0.25, 1.0, 0.75],
                     [0.0, 0.5, 0.0, 0.75, 1.0]])


def test_match_leaves():
    similarity = distances.match_leaves(X)
    np.testing.assert_allclose(similarity, expected)


def test_match_nodes():
    similarity = distances.match_nodes(X_binary_sp)
    np.testing.assert_allclose(similarity, expected_binary)


#def test_binary_hamming_similarity_sparse():
#    similarity = distances.hamming_similarity(X_binary_sp, is_binary=True)
#    np.testing.assert_allclose(similarity, expected)
