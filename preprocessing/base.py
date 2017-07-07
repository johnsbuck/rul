import numpy as np


def split_units(data, target=None):
    """Splits the units of a given array into a list of unit arrays.

    Args:
        data (array_like, shape (k,M)): The turbofan engine data set. First column should be the unit numbers used for
            separating the data.
        target (M-length sequence): An optional target array that splits the data set based on the unit numbers from
            the data parameter.

    Returns:
        List of Arrays. Each array is an engine unit split based on unit number.
    """
    if target is None:
        return np.split(np.copy(data), np.where(np.diff(data[:, 0]))[0]+1)
    else:
        return np.split(np.copy(data), np.where(np.diff(data[:, 0]))[0]+1), \
               np.split(np.copy(target), np.where(np.diff(data[:, 0])[0]+1))[1]


def bucket_units(data, chunks):
    """Separates data into units, which then splits the data points into buckets based on cycle number of the unit.
    Example:
        5 chunks = 5 buckets
        bucket 0: First 20% [Position 0% - 20%]
        bucket 1: Second 20% [Position 20% - 40%]
        ...
        bucket 4: Final 20% [Position 80% - 100%]

    Args:
        data ((M, k) shaped array): The data set containing the unit number and other feature information.
        chunks (int): Number of buckets.

    Returns:
        (Chunks, N, k) shaped array.
    """
    X = split_units(data)

    buckets = np.array_split(X[0], chunks)
    for i in xrange(1, len(X)):
        new_buckets = np.array_split(X[i], chunks)
        for j in xrange(chunks):
            buckets[j] = np.vstack((buckets[j], new_buckets[j]))

    return buckets
