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
from sklearn.metrics import pairwise_distances
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.utils import check_array, check_random_state

from forest_kernels import tree_utils
from forest_kernels.data_generators import generate_discriminative_dataset


__all__ = ['BaseForestKernel', 'RandomForestKernel', 'ExtraTreesKernel']


def sample_depths(forest, random_state=123):
    """Randomly sample depths from a forest of decision trees. The root
    depth (index zero) is excluded.

    Parameters
    ----------
    forest : A class instance derived from `sklearn.ensemble.BaseForest`.
       The ensemble of tree's whose depths are sampled.

    Returns
    -------
    tree_depths : list
        A list of depths sampled for the tree at that index.
    """
    random_state = check_random_state(random_state)
    return [random_state.choice(range(1, tree.tree_.max_depth + 1))
            for tree in forest.estimators_]


def leaf_node_kernel(X_leaves):
    """The leaf node kernel matrix induced by an ensemble of
    decision trees.

   The terminal or leaf nodes of an unpruned decision tree contain only
   a small number of observations. These observations lie in a similar
   partition of the sample space, where 'similar' is defined by the
   splitting criterion of the tree (gini norm, variance reduction, etc.) as
   well as the target being classified.

   These similar partitions can be used to construct a similarity measure. The
   training data are run down each tree in the ensemble. If observations x_i
   and x_j both land in the same leaf node, the similarity between x_i and x_j
   is increased by one. This is done for each tree of the ensemble, so
   that the final similarity is defined as:

                  Number of leaves shared by x_i and x_j
        K(i, j) = --------------------------------------
                  Total number of trees in the forest.

    Note that this similarity measure lies between 0 and 1.

    Parameters
    ----------
    X_leaves : array-like, shape = [n_samples, n_estimators]
        For each datapoint x in X and for each tree in the forest, contains
        the index of the leaf x ends up in. The result of calling
        `forest.apply(X)`.

    Returns
    -------
    K : array-like, shape = [n_samples, n_samples]
        A kernel matrix K such that K_{i, j} is the similarity between
        the ith and jth vectors of the given leaf matrix X_leaves.
    [1]
    [2]
    [3]

    References
    ----------
    .. [1] E. Scornet, "Random Forests and Kernel Methods," in IEEE
           Transactions on Information Theory, vol. 62, no. 3,
           pp. 1485-1500, March 2016
    .. [2] Tao Shi and Steve Horvath (2006) Unsupervised Learning with Random
           Forest Predictors. Journal of Computational and Graphical
           Statistics. Volume 15, Number 1, March 2006, pp. 118-138(21)
    .. [3] Breiman, L. and Cutler, A. (2003), "Random Forests Manual v4.0",
           Technical report, UC Berkeley,
           ftp://ftp.stat.berkeley.edu/pub/users/breiman/Using_random_forests_v4.0.pdf.
    """
    return 1 - pairwise_distances(X_leaves, metric='hamming')


def random_partition_kernel(forest, X, tree_depths='random', random_state=123):
    """Random Partition Kernel induced by an ensemble of decision trees.
    """
    if tree_depths == 'random':
        tree_depths = sample_depths(forest, random_state=random_state)

    n_samples = X.shape[0]
    kernel = np.zeros(shape=(n_samples, n_samples))
    for tree_idx, tree in enumerate(forest.estimators_):
        node_indicator = tree_utils.apply_to_depth(
            tree, X, depth=tree_depths[tree_idx])
        kernel += tree_utils.node_similarity(node_indicator)

    return kernel / len(forest.estimators_)


class BaseForestKernel(six.with_metaclass(ABCMeta,
                                          BaseForest,
                                          TransformerMixin)):
    """Base class for all kernels derived from a forest of trees.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """
    @abstractmethod
    def __init__(self,
                 base_estimator,
                 n_estimators=10,
                 estimator_params=tuple(),
                 bootstrap=False,
                 oob_score=False,
                 kernel_type='random',
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

        self.kernel_type = kernel_type
        self.sampling_method = sampling_method

    def _set_oob_score(self, X, y):
        raise NotImplementedError("OOB score not supported in tree embedding")

    def fit(self, X, y=None, sample_weight=None):
        X = check_array(X, accept_sparse=['csc'], ensure_2d=False)

        if self.kernel_type not in ['leaf', 'random']:
            raise ValueError("`kernel_type` must be one of {'leaf', 'random'}."
                             " Got `kernel_type = {}".format(kernel_type))

        if sparse.issparse(X):
            # Pre-sort indices to avoid each individual tree of the
            # ensemble sorting the indices.
            X.sort_indices()

        if y is not None:
            X_ = X
            y_ = y
        else:
            X_, y_ = generate_discriminative_dataset(
                X, method=self.sampling_method)

        super(BaseForestKernel, self).fit(X_, y_,
                                          sample_weight=sample_weight)

        # fix the depths used when 'kernel_type == 'random'
        self.tree_depths_ = sample_depths(self, random_state=self.random_state)

        return self

    def transform(self, X, kernel_type=None):
        if kernel_type is None:
            kernel_type = self.kernel_type

        if kernel_type not in ['leaf', 'random']:
            raise ValueError("`kernel_type` must be one of {'leaf', 'random'}."
                             " Got `kernel_type = {}".format(kernel_type))

        if kernel_type == 'leaf':
            return leaf_node_kernel(self.apply(X))
        else:
            return random_partition_kernel(self, X,
                                           tree_depths=self.tree_depths_,
                                           random_state=self.random_state)


class RandomForestKernel(BaseForestKernel):
    """A Random Forest Kernel.

    This class implements a kernel induced by a random-forest classifier.

    Parameters
    ----------

    Notes
    -----

    References
    ----------

    See also
    --------
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
                 kernel_type='random',
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
                kernel_type=kernel_type,
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


class ExtraTreesKernel(BaseForestKernel):
    """A Extra Trees Kernel.

    This class implements a kernel induced by a extra-trees classifier.

    Parameters
    ----------

    Notes
    -----

    References
    ----------

    See also
    --------
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
                 kernel_type='random',
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
                kernel_type=kernel_type,
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
