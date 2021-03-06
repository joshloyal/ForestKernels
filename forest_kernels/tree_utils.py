import numpy as np
import scipy.sparse as sparse

from sklearn.metrics import pairwise_distances
from sklearn.utils import check_array


def node_similarity(X_nodes, Y_nodes=None):
    """A binary similarity matrix where samples i and j have a similarity
    of 1 if they are contained in the same node of a decision tree.

    Returns
    -------
    S: array-like, shape = [n_samples, n_samples]
        A symmetric binary matrix S such that S_{i, j} is 1 if the ith
        and jth vectors are contained in the same node and 0 otherwise.
    """
    if Y_nodes is None:
        Y_nodes = X_nodes

    return (X_nodes * Y_nodes.T).toarray()


def get_leaf_nodes(tree, depth=-1, return_depths=False):
    """Return indices of nodes that become leaf nodes if a tree is truncated
    at a given depth.

    Parameters
    ----------
    tree : A class instance derived from `sklearn.tree.BaseDecisionTree`
        The tree from which the node indices are derived.

    depth : int, optional (default=-1)
        The depth at which the tree is truncted. An integer betwen
        [0, max_depth]. A depth of zero is the root node. A depth of -1 means
        that the tree is not truncated and the leaf nodes are returned.

    return_depths : bool, optional (default=False)
        Whether to return an additional array indicating the depths of
        the new leaf indices in the original tree. Unbalanced trees may
        have leaf nodes of mixed depths.

    Returns
    -------
    leaf_indices : array-like, shape = [n_leaves,]
        Node indices of the new leaf nodes.

    depths : array-like, shape = [n_leaves,]
        Depths of leaf nodes in the original tree.
    """
    max_depth = tree.tree_.max_depth
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right

    if depth == -1 or depth > max_depth:
        depth = max_depth

    leaf_indices = [0]
    node_depths = [0]
    stack = [(0, -1, 0)]  # node_id, parent_depth, parent_id
    while len(stack) > 0:
        node_id, parent_depth, parent_id = stack.pop()
        if parent_depth + 1 <= depth:
            try:
                parent_index = leaf_indices.index(parent_id)
                leaf_indices.pop(parent_index)
                node_depths.pop(parent_index)
            except ValueError:
                pass
            leaf_indices.append(node_id)
            node_depths.append(parent_depth + 1)

        # if we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1, node_id))
            stack.append((children_right[node_id], parent_depth + 1, node_id))

    if return_depths:
        sorted_indices = np.argsort(leaf_indicies)
        return (np.array(leaf_indices)[sorted_indices],
                np.array(node_depths)[sorted_indices])
    return np.sort(leaf_indices)


def apply_until(tree, X, depth=-1):
    """Returns the leaf indicator matrix for a tree truncated at the
    requested depth.

    Parameters
    ----------
    tree : A class instance derived from `sklearn.tree.BaseDecisionTree`
        The tree from which the node indices are derived.

    X : array_like or sparse matrix, shape = [n_samples, n_features]
        The input samples. Internally, it will be converted to
        ``dtype=np.float32`` and if a sparse matrix is provided
        to a sparse ``csr_matrix``.

    depth : int, optional (default=-1)
        The depth at which the tree is truncted. An integer betwen
        [0, max_depth]. A depth of zero is the root node. A depth of -1 means
        that the tree is not truncated and the leaf nodes are returned.

    Returns
    -------
    indicator : sparse csr array, shape = [n_samples, n_leaves]
        Return a leaf indicator matrix where non zero elements indicate
        that the sample ended up in that leaf.
    """
    X = check_array(X, accept_sparse='csr')

    node_indicator = tree.decision_path(X)
    node_indices = get_leaf_nodes(tree, depth=depth)

    return node_indicator[:, node_indices]
