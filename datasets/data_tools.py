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

