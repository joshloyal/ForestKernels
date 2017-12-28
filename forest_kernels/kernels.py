import numpy as np
import numbers
import six

from sklearn.utils import check_random_state

from forest_cluster import tree_utils
from forest_cluster import distances


def fixed_depth_partition(tree, X, depth='random', random_state=123):
    # pick a random depth
    if depth == 'random':
        rng = check_random_state(random_state)
        max_depth = tree.tree_.max_depth
        depth = rng.choice(range(1, max_depth + 1))

    # get the embedding for that depth
    node_indicator = tree_utils.depth_embedding(tree, X, depth=depth)
    return distances.match_nodes(node_indicator)


def all_partitions(tree, X, normalize=True):
    n_samples = X.shape[0]
    max_depth = tree.tree_.max_depth
    kernel = np.zeros(shape=(n_samples, n_samples))
    for depth in range(1, max_depth + 1):
        node_indicator = tree_utils.depth_embedding(tree, X, depth=depth)
        kernel += distances.match_nodes(node_indicator)

    if normalize:
        kernel /= max_depth

    return kernel


def random_partitions_kernel(forest, X, kernel_depth='all', random_state=123):
    rng = check_random_state(random_state)

    n_partitions = 0
    n_samples = X.shape[0]
    kernel = np.zeros(shape=(n_samples, n_samples))
    for tree_idx, tree in enumerate(forest.estimators_):
        if kernel_depth == 'all':
            n_partitions += tree.tree_.max_depth
            kernel += all_partitions(tree, X, normalize=False)
        elif kernel_depth == 'random' or isinstance(kernel_depth, list):
            n_partitions += 1
            kernel += fixed_depth_partition(
                tree, X, depth=kernel_depth[tree_idx], random_state=rng)
        else:
            raise ValueError('Unrecognized `depth`.')

    return kernel / n_partitions


def random_forest_kernel(forest, X, kernel_depth='all', random_state=123):
    if isinstance(kernel_depth, numbers.Integral):
        kernel_depth = [kernel_depth] * forest.n_estimators

    if kernel_depth == 'leaves':
        leaves = forest.apply(X)
        return distances.match_leaves(leaves)
    elif isinstance(kernel_depth, (six.string_types, list)):
        return random_partitions_kernel(
            forest, X, kernel_depth=kernel_depth, random_state=random_state)
    else:
        raise ValueError("kernel_depth must be a string or an integer")
