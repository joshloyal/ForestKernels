import numpy as np

import forest_kernels

from sklearn.datasets import make_blobs

from forest_kernels.kernels import leaf_node_kernel


def test_leaf_node_kernel():
    X_leaves = np.array([[1, 2, 3, 4],
                         [3, 2, 3, 1],
                         [1, 2, 3, 4],
                         [1, 10, 9, 1],
                         [3, 10, 9, 1]])
    K_expected = np.array([[1.0, 0.5, 1.0, 0.25, 0.0],
                           [0.5, 1.0, 0.5, 0.25, 0.5],
                           [1.0, 0.5, 1.0, 0.25, 0.0],
                           [0.25, 0.25, 0.25, 1.0, 0.75],
                           [0.0, 0.5, 0.0, 0.75, 1.0]])
    K = leaf_node_kernel(X_leaves)
    np.testing.assert_allclose(K, K_expected)


def test_random_forest_kernel_random():
    X, y = make_blobs(random_state=123)

    kernel = forest_kernels.RandomForestKernel(kernel_type='random')
    kernel.fit_transform(X)

def test_random_forest_kernel_leaf():
    X, y = make_blobs(random_state=123)

    kernel = forest_kernels.RandomForestKernel(kernel_type='leaf')
    kernel.fit_transform(X)
