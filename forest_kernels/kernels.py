from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division

import numbers
import six

import numpy as np
import scipy.sparse as sparse

from abc import ABCMeta, abstractmethod

from sklearn.base import TransformerMixin, is_regressor
from sklearn.ensemble.forest import BaseForest
from sklearn.metrics import pairwise_distances
from sklearn.tree import (DecisionTreeClassifier, ExtraTreeClassifier,
                          DecisionTreeRegressor, ExtraTreeRegressor)
from sklearn.utils import check_array, check_random_state

from forest_kernels import tree_utils
from forest_kernels.synthetic_data import generate_discriminative_dataset


__all__ = ['BaseForestKernel',
           'RandomForestClassifierKernel', 'RandomForestRegressorKernel',
           'ExtraTreesClassifierKernel', 'ExtraTreesRegressorKernel']


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


def leaf_node_kernel(X_leaves, Y_leaves=None):
    """The leaf node kernel matrix induced by an ensemble of
    decision trees. Also known as the connection function of the
    random forest.

    An ensemble of trees (such as a random forest) can be viewed as a
    kernel-based model:

        f(x) = sum_i^n y_i * K_t(x, x_i) / sum_l^n * K_t(x, x_i)

    where the kernel function K_t measures the similarity between any pair
    of inputs x_i and x.

    Note that the terminal or leaf nodes of an unpruned decision tree contain
    only a small number of observations. These observations lie in a similar
    partition of the sample space, where 'similar' is defined by the
    splitting criterion of the tree (gini norm, variance reduction, etc.) as
    well as the target being classified. These similar partitions can be used
    to construct a similarity measure, which also satisfies the properties of a
    kernel-function. The training data are run down each tree in the ensemble.
    If observations x_i and x_j both land in the same leaf node,
    the similarity between x_i and x_j is increased by one.
    This is done for each tree of the ensemble, so that the final similarity
    is defined as:

                  Number of leaves shared by x_i and x_j
        K(i, j) = --------------------------------------
                  Total number of trees in the forest.

    Note that this similarity measure lies between 0 and 1. This quantity
    can also be thought of as the empircal probability that x_i and x_j lie
    in the same cell in the random foreset.

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
    [4]

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
           ftp://ftp.stat.berkeley.edu/pub/users/breiman/
           Using_random_forests_v4.0.pdf.
    .. [4] P. Geurts, D. Ernst., and L. Whenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.
    """
    return 1 - pairwise_distances(X_leaves, Y=Y_leaves, metric='hamming')


def random_partition_kernel(forest, X, Y=None, tree_depths='random',
                            random_state=123):
    """Random Partition Kernel induced by an ensemble of decision trees.

    A random partition kernel is a kernel-function induced by a distribution
    over partitions (or random partitions) of a dataset. Since an ensemble of
    trees such as a random-forest partitions a dataset into groups
    (the tree nodes), these models can be thought of random partition
    generators and so induce a kernel-function.

    By repeatedly cutting a data-set into random partitions we would expect
    data points that are similar to each other to be grouped together
    more often then other samples. Likewise nodes in the
    decision tree should contain similar datapoints. In order to sample the
    whole hierachal structure of the forest a depth is chosen at random to
    sample and then the common partitions are added up. The kernel is as
    follows:
                  Number of times x_i and x_j occur in the same node
        K(i, j) = --------------------------------------------------
                  Total number of trees in the ensemble
    Parameters
    ----------
    forest: A class instance derived from `sklearn.ensemble.BaseForest`.
        The forest from which the kernel is calculated.

    X: array-like, shape = [n_samples, n_features]
       The data to train the kernel on.

    tree_depths: list or str, optional (default='random')
        A list of depths to use for each tree. if `tree_depths`='random'
        then the depths are randomly sampled from a discrete uniform
        distribution between 1 and max_depth.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    K : array-like, shape = [n_samples, n_samples]
        A kernel matrix K such that K_{i, j} is the similarity between
        the ith and jth vectors.
    [1]

    References
    ----------
    .. [1] A. Davis, Z. Ghahramani, "The Random Forest Kernel and creating
           other kernels for big data from random partitions",
           CoRR, 2014.
    """
    if tree_depths == 'random':
        tree_depths = sample_depths(forest, random_state=random_state)

    n_samples_x = X.shape[0]
    n_samples_y = Y.shape[0] if Y is not None else n_samples_x
    kernel = np.zeros(shape=(n_samples_x, n_samples_y))
    for tree_idx, tree in enumerate(forest.estimators_):
        node_indicator_X = tree_utils.apply_until(
            tree, X, depth=tree_depths[tree_idx])

        if Y is not None:
            node_indicator_Y = tree_utils.apply_until(
                tree, Y, depth=tree_depths[tree_idx])
        else:
            node_indicator_Y = node_indicator_X

        kernel += tree_utils.node_similarity(
            node_indicator_X, node_indicator_Y)

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
            if is_regressor(self.base_estimator):
                raise ValueError(
                    "The unsupervised kernels are not available for "
                    "regressors. Either provide a value for `y` or use one "
                    "of the following classes: {RandomForestClassifierKernel, "
                    "ExtraTreesClassifierKernel}.")
            X_, y_ = generate_discriminative_dataset(
                X, method=self.sampling_method)

        super(BaseForestKernel, self).fit(X_, y_,
                                          sample_weight=sample_weight)

        # fix the depths used when 'kernel_type == 'random'
        self.X_ = X_
        self.tree_depths_ = sample_depths(self, random_state=self.random_state)

        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y=y, **fit_params)

        if self.kernel_type == 'leaf':
            return leaf_node_kernel(self.apply(X))
        else:
            return random_partition_kernel(self, X,
                                           tree_depths=self.tree_depths_,
                                           random_state=self.random_state)

    def transform(self, X, kernel_type=None):
        if kernel_type is None:
            kernel_type = self.kernel_type

        if kernel_type not in ['leaf', 'random']:
            raise ValueError("`kernel_type` must be one of {'leaf', 'random'}."
                             " Got `kernel_type = {}".format(kernel_type))

        if kernel_type == 'leaf':
            return leaf_node_kernel(self.apply(X), Y_leaves=self.apply(self.X_))
        else:
            return random_partition_kernel(self, X, Y=self.X_,
                                           tree_depths=self.tree_depths_,
                                           random_state=self.random_state)


class RandomForestClassifierKernel(BaseForestKernel):
    """A Random Forest Classifier Kernel.

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
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features='auto',
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 kernel_type='random',
                 sampling_method='bootstrap',
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        super(RandomForestClassifierKernel, self).__init__(
            base_estimator=DecisionTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"),
            bootstrap=bootstrap,
            kernel_type=kernel_type,
            sampling_method=sampling_method,
            oob_score=False,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split


class RandomForestRegressorKernel(BaseForestKernel):
    """A Random Forest Regressor Kernel.

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
                 criterion='mse',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features='auto',
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 kernel_type='random',
                 sampling_method='bootstrap',
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super(RandomForestRegressorKernel, self).__init__(
            base_estimator=DecisionTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=False,
            kernel_type=kernel_type,
            sampling_method=sampling_method,
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
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split


class ExtraTreesClassifierKernel(BaseForestKernel):
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
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 kernel_type='random',
                 sampling_method='bootstrap',
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super(ExtraTreesClassifierKernel, self).__init__(
            base_estimator=ExtraTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=False,
            kernel_type=kernel_type,
            sampling_method=sampling_method,
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
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split


class ExtraTreesRegressorKernel(BaseForestKernel):
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
                 criterion='mse',
                 max_depth=5,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features='auto',
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 kernel_type='random',
                 sampling_method='bootstrap',
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super(ExtraTreesRegressorKernel, self).__init__(
            base_estimator=ExtraTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=False,
            kernel_type=kernel_type,
            sampling_method=sampling_method,
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
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
