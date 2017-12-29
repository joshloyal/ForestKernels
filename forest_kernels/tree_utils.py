import numpy as np
import scipy.sparse as sparse

from sklearn.metrics import pairwise_distances
from sklearn.utils import check_array

from forest_kernels import array_utils


def node_similarity(X_nodes):
    """A binary similarity matrix where samples i and j have a similarity
    of 1 if they are contained in the same node of a decision tree.

    Returns
    -------
    S: array-like, shape = [n_samples, n_samples]
        A symmetric binary matrix S such that S_{i, j} is 1 if the ith
        and jth vectors are contained in the same node and 0 otherwise.
    """
    return (X_nodes * X_nodes.T).toarray()


def get_node_depths(tree):
    """Determine the depth of a node in a decision tree."""
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right

    node_depths = np.zeros(shape=n_nodes, dtype=np.int64)
    stack = [(0, -1)]
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depths[node_id] = parent_depth + 1

        # if we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))

    return node_depths


def get_node_indicators(tree, X, depth=-1):
    """Apply a tree to X at a specified depth. Return zero/one indicator matrix
    of whether the sample landed in that node.
    """
    max_depth = tree.tree_.max_depth

    if depth == -1 or depth > max_depth:
        depth = max_depth

    node_depths = get_node_depths(tree)
    node_indicator = tree.decision_path(X)

    return node_indicator[:, node_depths == depth]


def apply_to_depth(tree, X, depth=-1):
    X = check_array(X, accept_sparse='csr')

    max_depth = tree.tree_.max_depth
    if depth == -1 or depth > max_depth:
        depth = max_depth

    unmatched_samples = np.ones(X.shape[0], dtype=np.bool)

    node_indicators = []
    while np.any(unmatched_samples):
        nodes = get_node_indicators(tree, X, depth=depth)
        matched_indices = np.where(nodes.sum(axis=1) != 0)[0]
        # samples that have be matched do not need to be matched again
        array_utils.csr_assign_rows(nodes,
                                    np.where(~unmatched_samples)[0], value=0)

        # add to the list of indicators
        node_indicators.append(nodes)
        depth = depth - 1
        unmatched_samples[matched_indices] = False

    return array_utils.drop_zero_columns(
        sparse.hstack(node_indicators, format='csr'))
