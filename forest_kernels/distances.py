import numpy as np
import scipy.sparse as sparse
from sklearn.metrics import pairwise_distances


def hamming_similarity(X, is_binary=False):
    if sparse.issparse(X):
        if is_binary:
            return hamming_similarity_binary_sparse(X)
        else:
            return hamming_similarity_dense(X.toarray())

    return hamming_similarity_dense(X)


def match_leaves(X):
    return 1 - pairwise_distances(X, metric='hamming')


#def hamming_similarity_binary_sparse(X):
#    n_features = X.shape[1]
#    D = np.dot(1 - X, X.T)
#    return np.asarray((D + D.T) / n_features)


def match_nodes(X):
    H = (X * X.T).toarray()
    return H


def fast_hamming_dense(X):
    unique_values = np.unique(X)
    U = sparse.csr_matrix((X == unique_values[0]).astype(np.int32))
    H = (U * U.transpose()).toarray()
    for unique_value in unique_values[1:]:
        U = sparse.csr_matrix((X == unique_value).astype(np.int32))
        H += (U * U.transpose()).toarray()
    return np.sqrt(1 - H.astype(np.float64) / X.shape[1])
