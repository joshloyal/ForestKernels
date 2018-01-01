import matplotlib.pyplot as plt
import numpy as np

from scipy.special import expit
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import ExtraTreesRegressor
from forest_kernels import ExtraTreesRegressorKernel


rng = np.random.RandomState(123)

n_samples = 500
X = rng.uniform(-6, 6, (n_samples, 1))
y = 10 * expit(X[:, 0])
y +=  rng.randn(n_samples)

# ridge regression
param_grid = {"alpha": [1e-1, 1e0, 1e1, 1e2],
              "kernel": ['rbf'],
              "gamma": [1e-2, 1e-1, 1e0, 1e1, 1e2]}
kr = GridSearchCV(KernelRidge(), cv=5, param_grid=param_grid)
kr.fit(X, y)

# tree kernel
tree_kr = make_pipeline(
    ExtraTreesRegressorKernel(
        n_estimators=200,
        sampling_method='supervised',
        kernel_type='random_partitions',
        n_jobs=-1,
        random_state=123),
    KernelRidge(kernel='precomputed'),
    memory='model_cache'
)

param_grid = {"kernelridge__alpha": [1e-1, 1e0, 1e1, 1e2, 1e3]}
tree_kr = GridSearchCV(tree_kr, cv=5, param_grid=param_grid)
tree_kr.fit(X, y)

# extra trees
forest = ExtraTreesRegressor(
    n_estimators=200,
    n_jobs=-1,
    random_state=123).fit(X, y)

# plots
lw = 2
alpha = 0.7

X_plot = np.linspace(-6, 6, 100).reshape(-1, 1)

plt.scatter(X, y, c='k', s=4)
plt.plot(X_plot[:, 0], 10 * expit(X_plot[:, 0]),
         lw=lw, color='tomato', label='true', alpha=alpha)
plt.plot(X_plot[:, 0], forest.predict(X_plot),
         lw=lw, color='forestgreen', label='Extra Trees', alpha=alpha)
plt.plot(X_plot[:, 0], tree_kr.predict(X_plot),
         lw=lw, color='darkorange', label='Tree Kernel', alpha=alpha)
plt.plot(X_plot[:, 0], kr.predict(X_plot),
         lw=lw, color='navy', label='RBF Kernel', alpha=alpha)

plt.xlabel('data')
plt.ylabel('target')
plt.title('RBF Kernel vs. Tree Kernel')
plt.legend(loc='best', scatterpoints=1, prop={'size': 8})
plt.show()
