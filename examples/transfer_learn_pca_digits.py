import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_digits
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import KernelPCA

import forest_kernels

digits = load_digits()
X = digits.data
y = digits.target

train_id = np.logical_or(y == 7, y == 9)
test_id = np.logical_or(y == 3, y == 4)

rf_kernel = forest_kernels.ExtraTreesClassifierKernel(
    n_estimators=500,
    kernel_type='leaves',
    sampling_method='supervised',
    n_jobs=-1,
    random_state=123).fit(X[train_id], y[train_id])

# returns the RF similarity (kernel) between samples in a matrix
rf_kernel.set_kernel_X(X[test_id])
kernel_test = rf_kernel.transform(X[test_id])

pca = KernelPCA(kernel='precomputed', n_components=2)
X_pca = pca.fit_transform(kernel_test)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y[test_id])
