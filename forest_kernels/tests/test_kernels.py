import numpy as np
import pytest

import forest_kernels

from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier

from forest_kernels.kernels import leaf_node_kernel


@pytest.fixture
def balanced_forest():
    """A forest a depth one balanced trees."""
    X = [[-2, -1],
         [-1, -1],
         [-1, -2],
         [1, 1],
         [1, 2],
         [2, 1]]
    y = [-1, -1, -1, 1, 1, 1]
    forest = RandomForestClassifier(n_estimators=3, random_state=123).fit(X, y)

    return X, y, forest


@pytest.fixture
def unbalanced_forest():
    """A forest of unbalanced trees."""
    X = [[-2, -1],
         [-1, -1],
         [-1, -2],
         [1, 1],
         [1, 2],
         [2, 1],
         [1, 3],
         [2, 4],
         [1, 5]]
    y = [-1, -1, -1, 1, 1, 1,-1,-1,-1]
    forest = RandomForestClassifier(n_estimators=3, random_state=123).fit(X, y)

    return X, y, forest


def test_leaf_node_kernel_toy():
    """Test a worked out set of leaves has the correct kernel."""
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


def test_leaf_node_kernel_balanced(balanced_forest):
    """Test the balanced forest has a block diagnol kernel."""
    X, _, forest = balanced_forest

    K_expected = np.array([[1, 1, 1, 0, 0, 0],
                           [1, 1, 1, 0, 0, 0],
                           [1, 1, 1, 0, 0, 0],
                           [0, 0, 0, 1, 1, 1],
                           [0, 0, 0, 1, 1, 1],
                           [0, 0, 0, 1, 1, 1]])
    K = leaf_node_kernel(forest.apply(X))
    np.testing.assert_allclose(K, K_expected)


def test_leaf_node_kernel_unbalanced(unbalanced_forest):
    """Test an unbalanced forest is correct (not calculated by hand,
    but designed to catch changes)."""
    X, _, forest = unbalanced_forest

    K_expected = np.array([[1., 1., 1., 0.33333333, 0.33333333,
                            0.33333333,  0.33333333,  0.33333333,  0.33333333],
                            [1,  1.,  1.,  0.33333333,  0.33333333,
                            0.33333333,  0.33333333,  0.33333333,  0.33333333],
                            [1.,  1.,  1.,  0.33333333,  0.33333333,
                            0.33333333,  0.33333333,  0.33333333,  0.33333333],
                            [0.33333333,  0.33333333,  0.33333333,  1.,  1.,
                            0.66666667,  0.33333333,  0.33333333,  0.33333333],
                            [0.33333333,  0.33333333,  0.33333333,  1.,  1. ,
                            0.66666667,  0.33333333,  0.33333333,  0.33333333],
                            [0.33333333,  0.33333333,  0.33333333,  0.66666667,
                            0.66666667, 1.,  0.33333333,  0.66666667,
                            0.33333333],
                            [0.33333333, 0.33333333, 0.33333333,  0.33333333,
                              0.33333333, 0.33333333, 1.,  0.66666667,  1.],
                            [0.33333333,  0.33333333,  0.33333333,  0.33333333,
                              0.33333333, 0.66666667, 0.66666667, 1.,
                              .66666667],
                            [0.33333333, 0.33333333, 0.33333333, 0.33333333,
                             0.33333333, 0.33333333, 1, 0.66666667, 1]])
    K = leaf_node_kernel(forest.apply(X))
    np.testing.assert_allclose(K, K_expected)


def test_random_forest_kernel_random():
    X, y = make_blobs(random_state=123)

    kernel = forest_kernels.RandomForestKernel(kernel_type='random')
    kernel.fit_transform(X)


def test_random_forest_kernel_leaf():
    X, y = make_blobs(random_state=123)

    kernel = forest_kernels.RandomForestKernel(kernel_type='leaf')
    kernel.fit_transform(X)
