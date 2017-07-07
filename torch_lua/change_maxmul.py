import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Import from Preprocess Dataset Module')
parser.add_argument('mult', metavar='N', type=int, nargs=1, default=1)
args = parser.parse_args()

y = np.loadtxt('data/train_target.csv')
maxmul = np.loadtxt('data/maxmultiplier.csv').tolist()

y = np.round(y * maxmul) / args.mult

np.savetxt('data/train_target.csv', y)
np.savetxt('data/maxmultiplier.csv', np.asarray([args.mult]))
