import pandas as pd
from numpy import take, average
from matplotlib import pyplot as plt

from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score

import os

cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in %r: %s" % (cwd, files))

#importing the training set
training_set = pd.read_csv('task1a_Cross-validation_for_Ridge_Regression/train.csv')

#defining label y and features x_i
y = training_set.take([0], axis=1)
X = training_set.take([1,2,3,4,5,6,7,8,9,10,11,12,13], axis=1)

#defining the set of regularization parameters
lambda_ = [0.1, 1, 10, 100, 200]
kfold_score = []
lambda_score = []

#putting aside part of the data for testing of the model 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#finding regression in the training data

regressor = RidgeCV(alphas = lambda_, cv=10).fit(X_train, y_train)

## which cv do we need to use? I find contradicting info on whether LOOCV is the same as k-fold CV

#saving rmse performance of the model for each value of lambda in an array
lambda_score.append(average(regressor.cv_values_))

print(lambda_)
print(lambda_score)

    
    

