from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from forest_kernels import ExtraTreesClassifierKernel


digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=123)


rf_kernel_svc = make_pipeline(
    ExtraTreesClassifierKernel(
        n_estimators=500,
        kernel_type='random_partitions',
        sampling_method='bootstrap',
        n_jobs=-1,
        random_state=123),
    SVC(kernel='precomputed')
)

rf_kernel_svc.fit(X_train, y_train)
y_pred = rf_kernel_svc.predict(X_test)
print(accuracy_score(y_test, y_pred))

svc = SVC(gamma=0.001).fit(X_train, y_train)
y_pred = svc.predict(X_test)
print(accuracy_score(y_test, y_pred))
