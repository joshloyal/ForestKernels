import numpy as np
import pytest
import scipy.sparse as sparse

from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier

from forest_kernels import tree_utils


@pytest.fixture
def balanced_tree():
    """Generates a depth one balanced tree."""
    X = [[-2, -1],
         [-1, -1],
         [-1, -2],
         [1, 1],
         [1, 2],
         [2, 1]]
    y = [-1, -1, -1, 1, 1, 1]
    tree = DecisionTreeClassifier(random_state=123).fit(X, y)

    return X, y, tree

@pytest.fixture
def unbalanced_tree():
    """Generates a depth two unbalanced tree."""
    X = [[-2, -1],
         [-1, -1],
         [-1, -2],
         [1, 1],
         [1, 2],
         [2, 1],
         [1, 3],
         [2, 4],
         [1, 5]]
    y = [-1, -1, -1, 1, 1, 1,-1,-1,-1]
    tree = DecisionTreeClassifier(random_state=123).fit(X, y)

    return X, y, tree


@pytest.fixture
def tall_tree():
    """A depth three un-balanced tree fit with 100 samples."""
    X, y = make_blobs(random_state=123)
    tree = DecisionTreeClassifier(
        max_depth=3, min_samples_split=33, random_state=13).fit(X, y)

    return X, y, tree


@pytest.mark.parametrize('tree_data, expected_depths', [
    (balanced_tree(), [0]),
    (unbalanced_tree(), [0]),
    (tall_tree(), [0])],
    ids = ['balanced', 'unbalanced', 'tall']
)
def test_get_leaf_nodes_depth_zero(tree_data, expected_depths):
    """test that the depths of the nodes are correct."""
    _, _, tree = tree_data

    depths = tree_utils.get_leaf_nodes(tree, depth=0)
    np.testing.assert_allclose(depths, expected_depths)


@pytest.mark.parametrize('tree_data, expected_depths', [
    (balanced_tree(), [1, 2]),
    (unbalanced_tree(), [1, 2]),
    (tall_tree(), [1, 2])],
    ids = ['balanced', 'unbalanced', 'tall']
)
def test_get_leaf_nodes_depth_one(tree_data, expected_depths):
    """test that the depths of the nodes are correct."""
    _, _, tree = tree_data

    depths = tree_utils.get_leaf_nodes(tree, depth=1)
    np.testing.assert_allclose(depths, expected_depths)


@pytest.mark.parametrize('tree_data, expected_depths', [
    (balanced_tree(), [1, 2]),
    (unbalanced_tree(), [1, 3, 4]),  # go down one level on the right
    (tall_tree(), [1, 3, 6])],  # also go down one level on the right
    ids = ['balanced', 'unbalanced', 'tall']
)
def test_get_leaf_nodes_depth_two(tree_data, expected_depths):
    """test that the depths of the nodes are correct."""
    _, _, tree = tree_data

    depths = tree_utils.get_leaf_nodes(tree, depth=2)
    np.testing.assert_allclose(depths, expected_depths)


@pytest.mark.parametrize('tree_data, expected_depths', [
    (balanced_tree(), [1, 2]),
    (unbalanced_tree(), [1, 3, 4]),
    (tall_tree(), [1, 4, 5, 6])],  # the middle node is split one more time
    ids = ['balanced', 'unbalanced', 'tall']
)
def test_get_leaf_nodes_depth_three(tree_data, expected_depths):
    """test that the depths of the nodes are correct."""
    _, _, tree = tree_data

    depths = tree_utils.get_leaf_nodes(tree, depth=3)
    np.testing.assert_allclose(depths, expected_depths)


@pytest.mark.parametrize('tree_data, expected_depths', [
    (balanced_tree(), [1, 2]),
    (unbalanced_tree(), [1, 3, 4]),
    (tall_tree(), [1, 4, 5, 6])],  # the middle node is split one more time
    ids = ['balanced', 'unbalanced', 'tall']
)
def test_get_leaf_nodes_depth_negative_one(tree_data, expected_depths):
    """test that the depths of the nodes are correct."""
    _, _, tree = tree_data

    depths = tree_utils.get_leaf_nodes(tree, depth=-1)
    np.testing.assert_allclose(depths, expected_depths)


def test_apply_until_balanced(balanced_tree):
    X, _, tree = balanced_tree

    # all samples are in the root
    expected = np.ones((6, 1))
    root = tree_utils.apply_until(tree, X, depth=0).toarray()
    np.testing.assert_allclose(root, expected)

    # three samples go left and three samples go right
    expected = np.array([[1, 0],
                         [1, 0],
                         [1, 0],
                         [0, 1],
                         [0, 1],
                         [0, 1]])
    depth_one = tree_utils.apply_until(tree, X, depth=1).toarray()
    np.testing.assert_allclose(depth_one, expected)


def test_apply_until_unbalanced(unbalanced_tree):
    X, _, tree = unbalanced_tree

    # all samples are in the root
    expected = np.ones((9, 1))
    root = tree_utils.apply_until(tree, X, depth=0).toarray()
    np.testing.assert_allclose(root, expected)

    # three samples go left the rest go right
    expected = np.array([[1, 0],
                         [1, 0],
                         [1, 0],
                         [0, 1],
                         [0, 1],
                         [0, 1],
                         [0, 1],
                         [0, 1],
                         [0, 1]])
    depth_one = tree_utils.apply_until(tree, X, depth=1).toarray()
    np.testing.assert_allclose(depth_one, expected)

    # the remaing six are split at the next level
    expected = np.array([[1, 0, 0],
                         [1, 0, 0],
                         [1, 0 ,0],
                         [0, 1, 0],
                         [0, 1, 0],
                         [0, 1, 0],
                         [0, 0, 1],
                         [0, 0, 1],
                         [0, 0, 1]])
    depth_two = tree_utils.apply_until(tree, X, depth=2).toarray()
    np.testing.assert_allclose(depth_two, expected)


def test_apply_until_does_not_drop_columns(unbalanced_tree):
    """Test that if X does not occupy all nodes the indicator matrix
    still includes them."""
    X, _, tree = unbalanced_tree

    expected = np.array([[1, 0, 0],
                         [1, 0, 0]])
    depth_two = tree_utils.apply_until(tree, X[:2], depth=2).toarray()
    np.testing.assert_allclose(depth_two, expected)


def test_apply_until_negative_depth(unbalanced_tree):
    """Test depth == -1 returns the leaf nodes"""
    X, _, tree = unbalanced_tree

    expected = np.array([[1, 0, 0],
                         [1, 0, 0],
                         [1, 0 ,0],
                         [0, 1, 0],
                         [0, 1, 0],
                         [0, 1, 0],
                         [0, 0, 1],
                         [0, 0, 1],
                         [0, 0, 1]])
    depth_two = tree_utils.apply_until(tree, X, depth=-1).toarray()
    np.testing.assert_allclose(depth_two, expected)


def test_apply_until_too_high_depth(unbalanced_tree):
    """Test a depth larger than max_depth returns leaf nodes."""
    X, _, tree = unbalanced_tree

    # the remaing six are split at the next level
    expected = np.array([[1, 0, 0],
                         [1, 0, 0],
                         [1, 0 ,0],
                         [0, 1, 0],
                         [0, 1, 0],
                         [0, 1, 0],
                         [0, 0, 1],
                         [0, 0, 1],
                         [0, 0, 1]])
    depth_two = tree_utils.apply_until(tree, X, depth=100).toarray()
    np.testing.assert_allclose(depth_two, expected)


def test_node_similarity():
    """Test that the node similarity matrix works for a toy example."""
    node_indicators = sparse.csr_matrix(np.array([[1, 0, 0],
                                                  [1, 0, 0],
                                                  [0, 0, 1],
                                                  [0, 1, 0],
                                                  [0, 0, 1]]))
    S_expected = np.array([[1, 1, 0, 0, 0],
                           [1, 1, 0, 0, 0],
                           [0, 0, 1, 0, 1],
                           [0, 0, 0, 1, 0],
                           [0, 0, 1, 0, 1]])

    S = tree_utils.node_similarity(node_indicators)
    np.testing.assert_allclose(S, S_expected)


def test_node_similarity_balanced(balanced_tree):
    X, _, tree = balanced_tree

    # this is a balanced tree so targets are grouped together
    S_expected = np.array([[1, 1, 1, 0, 0, 0],
                           [1, 1, 1, 0, 0, 0],
                           [1, 1, 1, 0, 0, 0],
                           [0, 0, 0, 1, 1, 1],
                           [0, 0, 0, 1, 1, 1],
                           [0, 0, 0, 1, 1, 1]])
    node_indicators = tree_utils.apply_until(tree, X, depth=-1)
    S = tree_utils.node_similarity(node_indicators)
    np.testing.assert_allclose(S, S_expected)


def test_node_similarity_unbalanced(unbalanced_tree):
    X, _, tree = unbalanced_tree

    # this is a balanced tree so targets are grouped together
    S_expected = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0],
                           [1, 1, 1, 0, 0, 0, 0, 0, 0],
                           [1, 1, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 1, 1, 0, 0, 0],
                           [0, 0, 0, 1, 1, 1, 0, 0, 0],
                           [0, 0, 0, 1, 1, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 1, 1],
                           [0, 0, 0, 0, 0, 0, 1, 1, 1],
                           [0, 0, 0, 0, 0, 0, 1, 1, 1]])
    node_indicators = tree_utils.apply_until(tree, X, depth=-1)
    S = tree_utils.node_similarity(node_indicators)
    np.testing.assert_allclose(S, S_expected)


def test_node_similarity_XY_wide_similarity(unbalanced_tree):
    X, _, tree = unbalanced_tree
    Y = X[:2]

    nodes_X = tree_utils.apply_until(tree, X, depth=2)
    nodes_Y = tree_utils.apply_until(tree, Y, depth=2)

    S_expected = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0],
                           [1, 1, 1, 0, 0, 0, 0, 0, 0]])
    S = tree_utils.node_similarity(nodes_Y, nodes_X)

    assert S.shape == (len(Y), len(X))
    np.testing.assert_allclose(S, S_expected)

def test_node_similarity_XY_long_similarity(unbalanced_tree):
    X, _, tree = unbalanced_tree
    Y = X[:2]

    nodes_X = tree_utils.apply_until(tree, X, depth=2)
    nodes_Y = tree_utils.apply_until(tree, Y, depth=2)

    S_expected = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0],
                           [1, 1, 1, 0, 0, 0, 0, 0, 0]]).T
    S = tree_utils.node_similarity(nodes_X, nodes_Y)

    assert S.shape == (len(X), len(Y))
    np.testing.assert_allclose(S, S_expected)
