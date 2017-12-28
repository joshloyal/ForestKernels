import forest_kernels

from sklearn.datasets import make_blobs


def test_random_forest_kernel_random():
    X, y = make_blobs(random_state=123)

    kernel = forest_kernels.RandomForestKernel(kernel_type='random')
    kernel.fit_transform(X)

def test_random_forest_kernel_leaf():
    X, y = make_blobs(random_state=123)

    kernel = forest_kernels.RandomForestKernel(kernel_type='leaf')
    kernel.fit_transform(X)
