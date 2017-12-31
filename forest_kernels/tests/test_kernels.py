import numpy as np
import pytest

from sklearn.datasets import make_blobs, load_boston
from sklearn.tree import DecisionTreeRegressor

from forest_kernels.kernels import (
    RandomForestClassifierKernel, RandomForestRegressorKernel,
    ExtraTreesClassifierKernel, ExtraTreesRegressorKernel)
from forest_kernels.kernels import leaf_node_kernel


@pytest.fixture
def balanced_data():
    """A forest a depth one balanced trees."""
    X = [[-2, -1],
         [-1, -1],
         [-1, -2],
         [1, 1],
         [1, 2],
         [2, 1]]
    y = [-1, -1, -1, 1, 1, 1]

    return X, y


@pytest.fixture
def unbalanced_data():
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

    return X, y


@pytest.fixture(params=[
    RandomForestClassifierKernel,
    RandomForestRegressorKernel,
    ExtraTreesClassifierKernel,
    ExtraTreesRegressorKernel
    ],
    ids=[
    'rf_classifier',
    'rf_regressor',
    'extra_classifier',
    'extra_regressor'])
def KernelClass(request):
    return request.param


@pytest.fixture(params=[
    RandomForestClassifierKernel,
    ExtraTreesClassifierKernel,
    ],
    ids=[
    'rf_classifier',
    'extra_classifier'
    ])
def ClassifierKernelClass(request):
    return request.param


@pytest.fixture(params=[
    RandomForestRegressorKernel,
    ExtraTreesRegressorKernel,
    ],
    ids=[
    'rf_regressor',
    'extra_regressor'
    ])
def RegressorKernelClass(request):
    return request.param


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


def test_leaf_node_kernel_balanced(balanced_data, ClassifierKernelClass):
    """Test the balanced forest has a block diagnol kernel."""
    X, y = balanced_data

    forest = ClassifierKernelClass(
        n_estimators=3,
        kernel_type='leaves',
        sampling_method='supervised',
        random_state=123)
    K = forest.fit_transform(X, y)

    K_expected = np.array([[1, 1, 1, 0, 0, 0],
                           [1, 1, 1, 0, 0, 0],
                           [1, 1, 1, 0, 0, 0],
                           [0, 0, 0, 1, 1, 1],
                           [0, 0, 0, 1, 1, 1],
                           [0, 0, 0, 1, 1, 1]])
    np.testing.assert_allclose(K, K_expected)


def test_leaf_node_kernel_unbalanced(unbalanced_data):
    """Test an unbalanced forest is correct (not calculated by hand,
    but designed to catch changes)."""
    X, y = unbalanced_data

    forest = RandomForestClassifierKernel(
        n_estimators=3,
        kernel_type='leaves',
        sampling_method='supervised',
        random_state=123)
    K = forest.fit_transform(X, y)

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


def test_random_partitions_kernel_balanced(balanced_data,
                                           ClassifierKernelClass):
    """Test the balanced forest has a block diagnol kernel."""
    X, y = balanced_data

    forest = ClassifierKernelClass(
        n_estimators=3,
        kernel_type='random_partitions',
        sampling_method='supervised',
        random_state=123)
    K = forest.fit_transform(X, y)

    # the trees in the forest are stumps and able to split
    # on the first feature perfectly.
    # So we should perfectly classify the samples.
    K_expected = np.array([[1, 1, 1, 0, 0, 0],
                           [1, 1, 1, 0, 0, 0],
                           [1, 1, 1, 0, 0, 0],
                           [0, 0, 0, 1, 1, 1],
                           [0, 0, 0, 1, 1, 1],
                           [0, 0, 0, 1, 1, 1]])
    np.testing.assert_allclose(K, K_expected)


def test_random_partitions_kernel_unbalanced(unbalanced_data):
    """Test the ubalanced forest. More of a smoke test than anything."""
    X, y = unbalanced_data

    # extra trees chooses a different variable to split, so just use RF
    forest = RandomForestClassifierKernel(
        n_estimators=3,
        kernel_type='random_partitions',
        sampling_method='supervised',
        random_state=123)
    K = forest.fit_transform(X, y)

    K_expected = np.array([[ 1.        ,  1.        ,  1.        ,
                             0.33333333,  0.33333333,  0.33333333,
                             0.33333333,  0.33333333,  0.33333333],
                             [ 1.        ,  1.        ,  1.        ,
                             0.33333333,  0.33333333,  0.33333333,
                             0.33333333,  0.33333333,  0.33333333],
                             [ 1.        ,  1.        ,  1.        ,
                             0.33333333,  0.33333333,  0.33333333,
                             0.33333333,  0.33333333,  0.33333333],
                             [ 0.33333333,  0.33333333,  0.33333333,
                             1.        ,  1.        ,  0.66666667,
                             1.        ,  0.66666667,  1.        ],
                             [ 0.33333333,  0.33333333,  0.33333333,
                             1.        ,  1.        ,  0.66666667,
                             1.        ,  0.66666667,  1.        ],
                             [ 0.33333333,  0.33333333,  0.33333333,
                             0.66666667,  0.66666667,  1.        ,
                             0.66666667,  1.        ,  0.66666667],
                             [ 0.33333333,  0.33333333,  0.33333333,
                             1.        ,  1.        ,  0.66666667,
                             1.        ,  0.66666667,  1.        ],
                             [ 0.33333333,  0.33333333,  0.33333333,
                             0.66666667,  0.66666667,  1.        ,
                             0.66666667,  1.        ,  0.66666667],
                             [ 0.33333333,  0.33333333,  0.33333333,
                             1.        ,  1.        ,  0.66666667,
                             1.        ,  0.66666667,  1.        ]])
    np.testing.assert_allclose(K, K_expected)


@pytest.mark.parametrize('kernel_type', ['random_partitions', 'leaves'])
def test_kernel_type_classifier(kernel_type, ClassifierKernelClass):
    """simple tests over kernel type for a typical pipeline."""
    X, _ = make_blobs(n_samples=150, random_state=123)
    X_train, X_test = X[:100], X[100:]

    kernel = ClassifierKernelClass(
        kernel_type=kernel_type,
        random_state=123)

    # fit_transform pipeline
    K = kernel.fit_transform(X_train)
    assert K.shape == (100, 100)
    assert np.min(K) >= 0.0
    assert np.max(K) <= 1.0

    # fit then transform seperatly
    kernel.fit(X_train)

    # kernel's should match
    np.testing.assert_allclose(K, kernel.transform(X_train))

    # check new data makes sense
    K = kernel.transform(X_test)
    assert K.shape == (50, 100)
    assert np.min(K) >= 0.0
    assert np.max(K) <= 1.0


@pytest.mark.parametrize('kernel_type', ['random_partitions', 'leaves'])
def test_kernel_type_regressor(kernel_type, RegressorKernelClass):
    """simple tests over kernel_type for regressors."""
    boston = load_boston()

    X_train, X_test = boston.data[:400], boston.data[400:]
    y_train, y_test = boston.data[:400], boston.data[400:]

    kernel = RegressorKernelClass(n_estimators=3,
                                  kernel_type=kernel_type,
                                  sampling_method='supervised',
                                  random_state=123)

    # fit_transform pipeline
    K = kernel.fit_transform(X_train, y_train)
    assert K.shape == (400, 400)
    assert np.min(K) >= 0.0
    assert np.max(K) <= 1.0

    # fit then transform seperatly
    kernel.fit(X_train, y_train)

    # kernel's should match
    np.testing.assert_allclose(K, kernel.transform(X_train))

    # check new data makes sense
    K = kernel.transform(X_test)
    assert K.shape == (106, 400)
    assert np.min(K) >= 0.0
    assert np.max(K) <= 1.0


@pytest.mark.parametrize('sampling_method', ['uniform', 'bootstrap'])
def test_regressors_error_without_target(sampling_method, RegressorKernelClass):
    """Regressors currently have no fake data generation method."""
    boston = load_boston()

    with pytest.raises(ValueError):
        RegressorKernelClass(random_state=123).fit(boston.data)
