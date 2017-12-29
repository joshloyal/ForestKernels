import numpy as np

from forest_kernels.synthetic_data import generate_discriminative_dataset


X = np.array([[-2, -3],
              [-1, -3],
              [-1, -4],
              [1, 3],
              [1, 4],
              [2, 3]], dtype=np.float64)


def test_bootstrap_data():
    """Test boostrap data has the appropriate properties."""
    X_, y_ = generate_discriminative_dataset(X, method='bootstrap')

    # dataset should be twice as big
    assert X_.shape == (12, 2)
    assert y_.shape == (12,)

    # bootstrap data should have the sam eunique values
    np.testing.assert_allclose(np.unique(X_[:, 0]), np.array([-2, -1, 1, 2]))
    np.testing.assert_allclose(np.unique(X_[:, 1]), np.array([-4, -3, 3, 4]))

    # of course y should be zero and one
    np.testing.assert_allclose(np.unique(y_), np.array([0, 1]))

    # check unshuffled data has appropriate labels
    X_, y_ = generate_discriminative_dataset(
        X, method='bootstrap', shuffle=False)
    np.testing.assert_allclose(y_, np.concatenate((np.ones(6), np.zeros(6))))


def test_uniform_data():
    """Test uniform data set has the appropriate properties."""
    X_, y_ = generate_discriminative_dataset(X, method='uniform')

    # dataset should be twice as big
    assert X_.shape == (12, 2)
    assert y_.shape == (12,)

    # data should lie in the bounding box of the data
    assert np.all(X_[:, 0] >= -2)
    assert np.all(X_[:, 0] <= 2)
    assert np.all(X_[:, 1] >= -4)
    assert np.all(X_[:, 1] <= 4)

    # of course y should be zero and one
    np.testing.assert_allclose(np.unique(y_), np.array([0, 1]))

    # check unshuffled data has appropriate labels
    X_, y_ = generate_discriminative_dataset(
        X, method='bootstrap', shuffle=False)
    np.testing.assert_allclose(y_, np.concatenate((np.ones(6), np.zeros(6))))
