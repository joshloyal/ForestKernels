import seaborn as sns
import scipy.spatial as spatial
import scipy.cluster.hierarchy as hierarchy


def forest_dendogram(forest_kernel, X, method='average', **kwargs):
    dissimilarity = 1 - forest_kernel.transform(X)

    # cluster the dissimilarity using a hiearchical clustering algorithm
    D = spatial.distance.squareform(dissimilarity)
    linkage = hierarchy.linkage(D, method=method)

    # display the results
    return sns.clustermap(dissimilarity,
                          row_linkage=linkage,
                          col_linkage=linkage,
                          **kwargs)
