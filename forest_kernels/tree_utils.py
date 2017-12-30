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
    """Mask for selecting nodes at a given depth. If the tree is un-balanced
    this could lead to multiple depths being used."""
    max_depth = tree.tree_.max_depth
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right

    if depth == -1 or depth > max_depth:
        depth = max_depth

    node_mask = [0]
    node_depths = [0]
    stack = [(0, -1, 0)]  # node_id, parent_depth, parent_id
    while len(stack) > 0:
        node_id, parent_depth, parent_id = stack.pop()
        if parent_depth + 1 <= depth:
            try:
                parent_index = node_mask.index(parent_id)
                node_mask.pop(parent_index)
                node_depths.pop(parent_index)
            except ValueError:
                pass
            node_mask.append(node_id)
            node_depths.append(parent_depth + 1)

        # if we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1, node_id))
            stack.append((children_right[node_id], parent_depth + 1, node_id))

    if return_depths:
        sorted_indices = np.argsort(node_mask)
        return (np.array(node_mask)[sorted_indices],
                np.array(node_depths)[sorted_indices])
    return np.sort(node_mask)


def apply_until(tree, X, depth=-1):
    """Return a node indicator matrix corresponding to the node a sample
    lands in when the tree is truncated at a depth = `depth`."""
    X = check_array(X, accept_sparse='csr')


    node_indicator = tree.decision_path(X)
    node_indices = get_leaf_nodes(tree, depth=depth)

    return node_indicator[:, node_indices]
