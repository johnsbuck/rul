import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans


def rp_normalized(x, n_clusters=6):
    """Performs regime partitioning by detecting each of the operating condition clusters and performing
    standardization for each cluster's data points.

    Args:
        x ((M, 26) shaped float array): The dataset used to preprocess the data.
        n_clusters (int): The number of clusters to detect. Optional. Default: 6.

    Returns:
        (M, 20) shaped float array. The following columns are removed after processing:
            1. Operating Settings 1-3
            2. Sensor 1
            3. Sensor 22
            4. Sensor 23
    """
    if n_clusters > 1:
        kmeans = KMeans(n_clusters=n_clusters).fit(x[:, 2:5])
        clusters = kmeans.predict(x[:, 2:5]).reshape(-1, 1)

        for i in xrange(n_clusters):
            regime = np.where(clusters == i)[0]
            mean = x[regime, 5:].mean(axis=0)
            std = x[regime, 5:].std(axis=0)
            x[regime, 5:] = np.divide(np.subtract(x[regime, 5:], mean), std)

    x[:, [1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25]] = \
        preprocessing.scale(x[:, [1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25]])

    # Removed columns 5, 22, 23 due to NaNs existing in data
    # Removed Operating Conditions as they become nil from preprocessing and don't improve results
    return x[:, [0, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25]]


def rp_scaled(x, n_clusters=6, min_range=-1, max_range=1):
    """Performs regime partitioning by detecting each of the operating condition clusters and performing
    feature scaling for each cluster's data points.

    Args:
        x ((M, 26) shaped float array): The dataset used to preprocess the data.
        n_clusters (int): The number of clusters to detect. Optional. Default: 6.
        min_range (float): Minimum value for scaled data. Optional. Default: -1.
        max_range (float): Maximum value for scaled data. Optional. Default: 1.

    Returns:
        (M, 20) shaped float array. The following columns are removed after processing:
            1. Operating Settings 1-3
            2. Sensor 1
            3. Sensor 22
            4. Sensor 23
    """
    if n_clusters > 1:
        kmeans = KMeans(n_clusters=n_clusters).fit(x[:, 2:5])
        clusters = kmeans.predict(x[:, 2:5]).reshape(-1, 1)

        for i in xrange(n_clusters):
            regime = np.where(clusters == i)[0]
            x[regime, 5:] = preprocessing.minmax_scale(x[regime, 5:], feature_range=(min_range, max_range))

    x[:, [1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25]] = \
        preprocessing.minmax_scale(x[:, [1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25]],
                                   feature_range=(min_range, max_range))

    # Removed 5, 22, 23 due to NaNs existing in data
    # Removed Operating Conditions as they become nil from preprocessing and don't improve results
    return x[:, [0, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25]]
