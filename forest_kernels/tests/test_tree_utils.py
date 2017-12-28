import numpy as np
import scipy.sparse as sparse

from forest_kernels import tree_utils


def test_count_shared_nodes():
    node_indicators = sparse.csr_matrix(np.array([[1, 0, 0],
                                                  [1, 0, 0],
                                                  [0, 0, 1],
                                                  [0, 1, 0],
                                                  [0, 0, 1]]))
    S_expected = np.array([[1, 1, 0, 0, 0],
                           [1, 1, 0, 0, 0],
                           [0, 0, 1, 0, 1],
                           [0, 0, 0, 1, 0],
                           [0, 0, 1, 0, 1]])

    S = tree_utils.node_similarity(node_indicators)
    np.testing.assert_allclose(S, S_expected)
