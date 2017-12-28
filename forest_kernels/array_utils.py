import numpy as np
import scipy.sparse as sparse


def drop_zero_columns(s):
    s = s.tocoo(copy=False)
    nz_cols, new_col = np.unique(s.col, return_inverse=True)
    s.col[:] = new_col
    s._shape = (s.shape[0], len(nz_cols))
    return s.tocsr(copy=False)


def csr_assign_row(s, row, value=0):
    if not isinstance(s, sparse.csr_matrix):
        raise ValueError('Matrix given must be of CSR format.')
    s.data[s.indptr[row]:s.indptr[row+1]] = value


def csr_assign_rows(s, rows, value=0):
    for row in rows:
        csr_assign_row(s, row, value=0)
    s.eliminate_zeros()