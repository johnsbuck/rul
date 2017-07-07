import csv
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import preprocessing

sys.path.append("/home/jsb/Git/thesis/python")

import data_tools as dt
import data_dir as dd
import argparse

np.random.seed(1)

def regime_partition(data, path):
    n_clusters = 6
    if path == dd.CMAPSS_TRAIN_1 or path == dd.CMAPSS_TRAIN_3:
        n_clusters = 1

    if n_clusters > 1:
        kmeans = KMeans(n_clusters=n_clusters).fit(data[:, 2:5])

        clusters = kmeans.predict(data[:, 2:5]).reshape(-1, 1)

        for i in xrange(n_clusters):
            regime = np.where(clusters == i)[0]

            mean = data[regime, 5:].mean(axis=0)
            std = data[regime, 5:].std(axis=0)
            data[regime, 5:] = np.divide(np.subtract(data[regime, 5:], mean), std)
    data[:, [1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25]] = preprocessing.scale(data[:, [1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25]])
    return data[:, [0, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25]]

parser = argparse.ArgumentParser(description='Import from Preprocess Dataset Module')
parser.add_argument('--d', nargs='?', default='PHM_TRAIN')
parser.add_argument('-r', nargs='?', const=True, default=False)
parser.add_argument('-u', nargs='?', const=True, default=False)
parser.add_argument('--rnum', nargs=1, type=int, default=1)
args = parser.parse_args()

path = dd.__dict__[args.d]  # Adjusted Variable

args.rnum = args.rnum[0]

if args.u:
    X = dt.split_units(dt.import_nasa_dataset(path))
    y = []

    for unit in X:
        y.append(np.arange(unit.shape[0]-1, -1, -1).reshape(-1, 1))

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=1)

    train_end_cycle = []
    test_end_cycle = []

    for i in xrange(len(train_y)):
        train_end_cycle.append(np.zeros(train_y[i].shape[0]).reshape(-1, 1))
        random_cycles = np.arange(train_y[i].shape[0])
        np.random.shuffle(random_cycles)
        train_end_cycle[len(train_end_cycle) - 1][random_cycles[:args.rnum], 0] = 1

    for i in xrange(len(test_y)):
        test_end_cycle.append(np.zeros(test_y[i].shape[0]).reshape(-1, 1))
        random_cycles = np.arange(test_y[i].shape[0])
        np.random.shuffle(random_cycles)
        test_end_cycle[len(test_end_cycle) - 1][random_cycles[:args.rnum], 0] = 1

    np.savetxt('data/val_train_end.csv', np.vstack(train_end_cycle))
    np.savetxt('data/val_test_end.csv', np.vstack(test_end_cycle))

    train_y = np.vstack(train_y)
    test_y = np.vstack(test_y)
    train_X = np.vstack(train_X)
    test_X = np.vstack(test_X)

else:
    X = dt.import_nasa_dataset(path)
    y = []

    unit = 1
    test = []

    for i in xrange(X.shape[0]):
        if X[i, 0] != unit:
            y.extend(reversed(test))
            test = []
            unit += 1
        test.append(X[i, 1] - 1)
    y.extend(reversed(test))

    y = np.array(y).reshape(-1, 1)

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=1)

# maxmultiplier = 1.0
maxmultiplier = dt.import_nasa_dataset(path)[:, 1].max() - 1
print maxmultiplier
train_y = train_y/maxmultiplier
test_y = test_y/maxmultiplier

if args.r:
    train_X = regime_partition(train_X, path)
    test_X = regime_partition(test_X, path)
else:
    train_X[:, 1:] = preprocessing.scale(train_X[:, 1:])
    test_X[:, 1:] = preprocessing.scale(test_X[:, 1:])

np.savetxt('data/train_val.csv', train_X)
np.savetxt('data/train_val_target.csv', train_y)
np.savetxt('data/test_val.csv', test_X)
np.savetxt('data/test_val_target.csv', test_y)
np.savetxt('data/val_maxmultiplier.csv', np.asarray([maxmultiplier]))
