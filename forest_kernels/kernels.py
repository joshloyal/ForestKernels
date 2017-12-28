from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division

import numbers
import six

import numpy as np
import scipy.sparse as sparse

from abc import ABCMeta, abstractmethod

from sklearn.base import TransformerMixin
from sklearn.ensemble.forest import BaseForest
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.utils import check_array, check_random_state

from forest_kernels import tree_utils
from forest_kernels.data_generators import generate_discriminative_dataset


__all__ = ['BaseForestKernel', 'RandomForestKernel', 'ExtraTreesKernel']


def fixed_depth_partition(tree, X, depth='random', random_state=123):
    # pick a random depth
    if depth == 'random':
        rng = check_random_state(random_state)
        max_depth = tree.tree_.max_depth
        depth = rng.choice(range(1, max_depth + 1))

    # get the embedding for that depth
    node_indicator = tree_utils.depth_embedding(tree, X, depth=depth)
    return tree_utils.match_nodes(node_indicator)


def all_partitions(tree, X, normalize=True):
    n_samples = X.shape[0]
    max_depth = tree.tree_.max_depth
    kernel = np.zeros(shape=(n_samples, n_samples))
    for depth in range(1, max_depth + 1):
        node_indicator = tree_utils.depth_embedding(tree, X, depth=depth)
        kernel += tree_utils.match_nodes(node_indicator)

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
        return tree_utils.match_leaves(leaves)
    elif isinstance(kernel_depth, (six.string_types, list)):
        return random_partitions_kernel(
            forest, X, kernel_depth=kernel_depth, random_state=random_state)
    else:
        raise ValueError("kernel_depth must be a string or an integer")


class BaseForestKernel(six.with_metaclass(ABCMeta, BaseForest, TransformerMixin)):
    @abstractmethod
    def __init__(self,
                 base_estimator,
                 n_estimators=10,
                 estimator_params=tuple(),
                 bootstrap=False,
                 oob_score=False,
                 kernel_depth='random',
                 sampling_method='bootstrap',
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):

        super(BaseForestKernel, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)

        self.kernel_depth = kernel_depth
        self.sampling_method = sampling_method

    def _set_oob_score(self, X, y):
        raise NotImplementedError("OOB score not supported in tree embedding")

    def fit(self, X, y=None, sample_weight=None):
        X = check_array(X, accept_sparse=['csc'], ensure_2d=False)

        if sparse.issparse(X):
            # Pre-sort indices to avoid each individual tree of the
            # ensemble sorting the indices.
            X.sort_indices()

        if y is not None:
            X_ = X
            y_ = y
        else:
            X_, y_ = generate_discriminative_dataset(X, method=self.sampling_method)

        super(BaseForestKernel, self).fit(X_, y_,
                                          sample_weight=sample_weight)

        # fix the depths used when 'kernel_method == 'random'
        random_state = check_random_state(self.random_state)
        self.depths_ = [random_state.choice(range(1, tree.tree_.max_depth + 1))
                        for tree in self.estimators_]

        return self

    def transform(self, X, kernel_depth=None):
        if kernel_depth is None:
            kernel_depth = self.kernel_depth

        if kernel_depth == 'random':
            kernel_depth = self.depths_

        return random_forest_kernel(self, X, kernel_depth=kernel_depth)


class ExtraTreesKernel(BaseForestKernel):
    def __init__(self,
                 n_estimators=10,
                 criterion='gini',
                 max_depth=5,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features='auto',
                 max_leaf_nodes=None,
                 bootstrap=True,
                 kernel_depth='random',
                 sampling_method='bootstrap',
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super(ExtraTreesKernel, self).__init__(
                base_estimator=ExtraTreeClassifier(),
                n_estimators=n_estimators,
                estimator_params=("criterion", "max_depth",
                                  "min_samples_split",
                                  "min_samples_leaf",
                                  "min_weight_fraction_leaf",
                                  "max_features", "max_leaf_nodes",
                                  "random_state"),
                bootstrap=bootstrap,
                kernel_depth=kernel_depth,
                sampling_method=sampling_method,
                oob_score=False,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
                warm_start=warm_start)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = 1
        self.max_leaf_nodes = max_leaf_nodes


class RandomForestKernel(BaseForestKernel):
    """Very similar to sklearn's RandomTreesEmbedding;
    however, the forest is trained as a discriminator.
    """
    def __init__(self,
                 n_estimators=10,
                 criterion='gini',
                 max_depth=5,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features='auto',
                 max_leaf_nodes=None,
                 bootstrap=True,
                 kernel_depth='random',
                 sampling_method='bootstrap',
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super(RandomForestKernel, self).__init__(
                base_estimator=DecisionTreeClassifier(),
                n_estimators=n_estimators,
                estimator_params=("criterion", "max_depth",
                                  "min_samples_split",
                                  "min_samples_leaf",
                                  "min_weight_fraction_leaf",
                                  "max_features", "max_leaf_nodes",
                                  "random_state"),
                bootstrap=bootstrap,
                kernel_depth=kernel_depth,
                sampling_method=sampling_method,
                oob_score=False,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
                warm_start=warm_start)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
