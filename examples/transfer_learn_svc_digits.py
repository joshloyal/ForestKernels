import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

import forest_kernels

digits = load_digits()
X = digits.data
y = digits.target

# train the kernel on digits 7 and 8
train_id = np.logical_or(y == 7, y == 9)
rf_kernel = forest_kernels.RandomForestClassifierKernel(
    n_estimators=500,
    kernel_type='leaves',
    sampling_method='supervised',
    n_jobs=-1,
    random_state=123).fit(X[train_id], y[train_id])

# lets see how this kernel fairs on classifying digits 3 and 4
train_id = np.logical_or(y == 3, y == 4)
train_id[1000:] = False
test_id = np.logical_or(y == 3, y == 4)
test_id[:1000] = False

# set the kernel to be with respect to the new training data
rf_kernel.set_kernel_X(X[train_id])

# fit an svm with this new kernel
kernel_train = rf_kernel.transform(X[train_id])
svc = SVC(kernel='precomputed')
svc.fit(kernel_train, y[train_id])

# make predictions
kernel_test = rf_kernel.transform(X[test_id])
y_pred = svc.predict(kernel_test)
print(accuracy_score(y[test_id], y_pred))

# compare to an svm trained on the raw 3 and 4 digits
svc = SVC(gamma=0.001)
svc.fit(X[train_id], y[train_id])
y_pred = svc.predict(X[test_id])
print(accuracy_score(y[test_id], y_pred))
