import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import KernelPCA

import forest_kernels

digits = load_digits()
X = digits.data
y = digits.target

rf_kernel_pca = make_pipeline(
    forest_kernels.ExtraTreesClassifierKernel(
        n_estimators=500,
        kernel_type='leaves',
        sampling_method='bootstrap',
        random_state=123),
    KernelPCA(kernel='precomputed', n_components=2)
)

X_pca = rf_kernel_pca.fit_transform(X)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
