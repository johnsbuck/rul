from os.path import abspath
from os.path import dirname
import numpy as np


def generate_train_targets():
    """Used to generate the targets for each training set.
    Note: The target datasets are already pre-generated in each folder as train_target.csv
    """
    folders = ["phm", "cmapss_1", "cmapss_2", "cmapss_3", "cmapss_4"]

    module_path = dirname(abspath(__file__))

    for folder in folders:
        dataset = np.loadtxt(module_path + "/../" + folder + "/train.csv")
        target = []
        for unit in xrange(1, int(dataset[dataset.shape[0] - 1, 0])+1):
            target += (dataset[np.where(dataset[:, 0] == unit)[0]][::-1, 1] - 1).tolist()
        np.savetxt(module_path + "/../" + folder + "/train_target.csv", target)
