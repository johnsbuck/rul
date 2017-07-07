import regression as rg
import numpy as np

import sys
sys.path.append('/home/jsb/Git/thesis/python')
import data_tools as dt
import data_dir as dd

np.random.seed(1)

def gen_results(test, units):
    return np.array([test[np.where(units == i)[0][np.where(units == i)[0].shape[0] - 1]] for i in xrange(1, 219)])

train_pred = np.loadtxt('train_pred.csv') * 356.
test_pred = np.loadtxt('test_pred.csv') * 356.


'''
# If Validation
print "Getting Validation Data"

maxmultiplier = np.loadtxt('data/val_maxmultiplier.csv').tolist()

train_val = np.loadtxt('data/train_val.csv')
train_split = dt.split_units(train_val)
train_units = np.array([train_split[i][0, 0] for i in xrange(len(train_split))])
train_y = np.loadtxt('data/train_val_target.csv') * maxmultiplier

test_val = np.loadtxt('data/test_val.csv')
test_split = dt.split_units(test_val)
test_units = np.array([test_split[i][0, 0] for i in xrange(len(test_split))])

'''

# If Full
print "==> Getting Full Data"

maxmultiplier = np.loadtxt('data/maxmultiplier.csv').tolist()

train_val = dt.import_nasa_dataset(dd.PHM_TRAIN)
train_split = dt.split_units(train_val)
train_units = np.array([train_split[i][0, 0] for i in xrange(len(train_split))])
train_y = np.loadtxt('data/train_target.csv') * maxmultiplier

test_val = dt.import_nasa_dataset(dd.PHM_TEST)
test_split = dt.split_units(test_val)
test_units = np.array([test_split[i][0, 0] for i in xrange(len(test_split))])


print "==> Beginning Regression Algorithms"

train_poly = []
train_butter = []
train_savitzky = []
train_kalman = []
train_forest = []
train_svm = []

test_poly = []
test_butter = []
test_savitzky = []
test_kalman = []
test_forest = []
test_svm = []


print "==> Training Regression"

print "random forest training"
rf = rg.random_forest(train_pred.reshape(-1, 1), train_y)

print "svm training"
svm = rg.svm(train_pred.reshape(-1, 1), train_y)

print "==> Obtaining Results"
for i in xrange(train_units.shape[0]):
    series = np.where(train_val[:, 0] == train_units[i])[0]
    train_poly.append(rg.third_poly_regression(train_pred[series]).reshape(-1,1) if series.shape[0] > 30 else rg.linear_regression(train_pred[series]).reshape(-1,1))
    train_butter.append(rg.butterworth(train_pred[series], 3 if series.shape[0] > 30 else 1).reshape(-1,1))
    train_savitzky.append(rg.savitzky_golay(train_pred[series], 33 if series.shape[0]/3 > 33 else round(series.shape[0]/3) - (round(series.shape[0]/3)+1)%2, 3 if series.shape[0] > 30 else 1).reshape(-1,1))
    train_kalman.append(rg.kalman(train_pred[series]).reshape(-1,1))
    train_forest.append(rf.predict(train_pred[series].reshape(-1, 1)).reshape(-1, 1))
    train_svm.append(svm.predict(train_pred[series].reshape(-1, 1)).reshape(-1,1))

for i in xrange(test_units.shape[0]):
    series = np.where(test_val[:, 0] == test_units[i])[0]
    test_poly.append(rg.third_poly_regression(test_pred[series]).reshape(-1,1) if series.shape[0] > 30 else rg.linear_regression(test_pred[series]).reshape(-1,1))
    test_butter.append(rg.butterworth(test_pred[series], 3 if series.shape[0] > 30 else 1).reshape(-1,1))
    test_savitzky.append(rg.savitzky_golay(test_pred[series], 33 if series.shape[0]/3 > 33 else round(series.shape[0]/3) - (round(series.shape[0]/3)+1)%2, 3 if series.shape[0] > 30 else 1).reshape(-1,1))
    test_kalman.append(rg.kalman(test_pred[series]).reshape(-1,1))
    test_forest.append(rf.predict(test_pred[series].reshape(-1, 1)).reshape(-1, 1))
    test_svm.append(svm.predict(test_pred[series].reshape(-1, 1)).reshape(-1, 1))

print "==> Saving Results"

# Training
np.savetxt('filter/train_poly.csv', np.vstack(train_poly))
np.savetxt('filter/train_butter.csv', np.vstack(train_butter))
np.savetxt('filter/train_savitzky.csv', np.vstack(train_savitzky))
np.savetxt('filter/train_kalman.csv', np.vstack(train_kalman))
np.savetxt('filter/train_forest.csv', np.vstack(train_forest))
np.savetxt('filter/train_svm.csv', np.vstack(train_svm))

# Testing/Submission
np.savetxt('filter/test_poly.csv', np.vstack(test_poly))
np.savetxt('filter/test_butter.csv', np.vstack(test_butter))
np.savetxt('filter/test_savitzky.csv', np.vstack(test_savitzky))
np.savetxt('filter/test_kalman.csv', np.vstack(test_kalman))
np.savetxt('filter/test_forest.csv', np.vstack(test_forest))
np.savetxt('filter/test_svm.csv', np.vstack(test_svm))

# Results
np.savetxt('filter/poly_results.csv', gen_results(np.vstack(test_poly), test_val[:, 0]))
np.savetxt('filter/butter_results.csv', gen_results(np.vstack(test_butter), test_val[:, 0]))
np.savetxt('filter/sav_results.csv', gen_results(np.vstack(test_savitzky), test_val[:, 0]))
np.savetxt('filter/kal_results.csv', gen_results(np.vstack(test_kalman), test_val[:, 0]))
np.savetxt('filter/forest_results.csv', gen_results(np.vstack(test_forest), test_val[:, 0]))
np.savetxt('filter/svm_results.csv', gen_results(np.vstack(test_svm), test_val[:, 0]))

