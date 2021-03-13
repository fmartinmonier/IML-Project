import os
import numpy as np
from numpy import mean, std
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge
from numpy import absolute

filename = "train.csv"
dataframe = pd.read_csv(filename)
data = dataframe.values
X, y_actual = data[:, 1:13], data[:, 0]
print(X)
print(y_actual)
alphas = [0.1, 1, 10, 100, 200]

for alpha in alphas:
    model = Ridge(alpha=alpha)
#    # define ridge regression model with the three repeats of 10-fold cross-validation
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
#    # evaluate model
    scores = cross_val_score(model, X, y_actual, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
#    # force scores to be positive
    scores = absolute(scores)
    print(scores)
    print('alpha [%.3f] - RMSE:  %.3f' % (alpha, mean(scores)))

        

