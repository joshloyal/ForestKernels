import forest_kernels

from sklearn.datasets import make_blobs


def test_random_forest_kernel():
    X, y = make_blobs(random_state=123)

    kernel = forest_kernels.RandomForestKernel()
    kernel.fit_transform(X)
