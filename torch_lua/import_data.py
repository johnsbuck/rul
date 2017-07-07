import sys                                             
sys.path.append('/home/jsb/Git/thesis/python')
import data_tools as dt
import data_dir as dd
import preprocess_datasets as pd
import argparse

parser = argparse.ArgumentParser(description='Import from Preprocess Dataset Module')
parser.add_argument('--d', nargs='?', default='PHM_TRAIN')
parser.add_argument('--r', nargs='?', const=True, default=False)
args = parser.parse_args()

if not args.r:
    pd.export_normalized_dataset(dd.__dict__[args.d], dd.PHM_TEST)
else:
    pd.export_regime_partitioned_dataset(dd.__dict__[args.d], dd.PHM_TEST)
