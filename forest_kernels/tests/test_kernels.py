import numpy as np
import pytest

import forest_kernels

from sklearn.datasets import make_blobs, load_boston
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

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


def test_leaf_node_kernel_matches_decision_tree():
    """Test the leaf node kernel matches the predictions of a single regression
    tree."""
    boston = load_boston()
    tree = DecisionTreeRegressor(max_depth=3, random_state=123).fit(
        boston.data, boston.target)
    leaves = tree.apply(boston.data).reshape(-1, 1)

    # predictions using tree kernel
    K = leaf_node_kernel(leaves)
    K /= K.sum(axis=1)
    k_pred = np.dot(K, boston.target)

    y_pred = tree.predict(boston.data)
    np.testing.assert_allclose(k_pred, y_pred)


#def test_leaf_node_kernel_matches_random_forest():
#    boston = load_boston()
#    X, y = boston.data, boston.target
#
#    forest = RandomForestRegressor(max_depth=3, random_state=123).fit(X, y)
#
#    kernel = forest_kernels.RandomForestRegressorKernel(
#        max_depth=3,
#        kernel_type='leaf',
#        random_state=123).fit(X, y)
#
#    # predictions using tree kernel
#    K = kernel.transform(X)
#    K /= K.sum(axis=1)
#    k_pred = np.dot(K, y)
#
#    y_pred = forest.predict(boston.data)
#    np.testing.assert_allclose(k_pred, y_pred)


def test_random_forest_kernel_random():
    X, y = make_blobs(random_state=123)

    kernel = forest_kernels.RandomForestClassifierKernel(kernel_type='random')
    kernel.fit_transform(X)


def test_random_forest_kernel_leaf():
   X, y = make_blobs(random_state=123)

   kernel = forest_kernels.RandomForestClassifierKernel(n_estimators=3,
                                                        kernel_type='leaf',
                                                        random_state=123)
   K = kernel.fit_transform(X)


def test_random_forest_kernel_leaf_new_data():
   X, y = make_blobs(n_samples=100, random_state=123)

   X_train, X_test = X[:90], X[90:]
   y_train, y_test = y[:90], y[90:]

   kernel = forest_kernels.RandomForestClassifierKernel(n_estimators=3,
                                                        kernel_type='leaf',
                                                        random_state=123)
   kernel.fit(X_train, y_train)
   print(kernel.transform(X_test))

