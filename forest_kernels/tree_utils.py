import numpy as np
import scipy.sparse as sparse

from forest_cluster import array_utils


def which_shared_nodes(sample_ids, node_indicator):
    n_nodes = node_indicator.shape[1]

    common_nodes = (node_indicator.toarray()[sample_ids].sum(axis=0) ==
                    len(sample_ids))

    return np.arange(n_nodes)[common_nodes]


def what_depth(tree):
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # if we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    return node_depth, is_leaves


def apply(tree, X, depth=-1):
    max_depth = tree.tree_.max_depth

    if depth == -1 or depth > max_depth:
        depth = max_depth
    node_depth, _ = what_depth(tree)


    node_indicator = tree.decision_path(X)

    return node_indicator[:, node_depth == depth]


def depth_embedding(tree, X, depth=-1):
    max_depth = tree.tree_.max_depth
    if depth == -1 or depth > max_depth:
        depth = max_depth

    unmatched_samples = np.ones(X.shape[0], dtype=np.bool)

    node_indicators = []
    while np.any(unmatched_samples):
        nodes = apply(tree, X, depth=depth)
        matched_indices = np.where(nodes.sum(axis=1) != 0)[0]
        # samples that have be matched do not need to be matched again
        array_utils.csr_assign_rows(nodes,
                                    np.where(~unmatched_samples)[0], value=0)

        # add to the list of indicators
        node_indicators.append(nodes)
        depth = depth - 1
        unmatched_samples[matched_indices] = False

    return array_utils.drop_zero_columns(sparse.hstack(node_indicators, format='csr'))
