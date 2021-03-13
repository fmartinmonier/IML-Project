from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from numpy import average, absolute
import numpy as np
import pandas as pd  

#https://scikit-learn.org/0.16/modules/generated/sklearn.linear_model.RidgeCV.html

#load dataset
train = pd.read_csv("train.csv" )
#y, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13 = train.y, train.x1, train.x2, train.x3, train.x4, train.x5, train.x6, train.x7, train.x8, train.x9, train.x10, train.x11, train.x12, train.x13
#X = np.array([[x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13]])
y = train['y']
X = train [['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13']]

lamb = [0.1,1,10,100,200]
i=1
scores_f = []
for alph in lamb:
    model = Ridge(alpha=alph)
    model.fit(X,y)
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error') #cross validation for 10fold with RMSE
    scores_f.append(absolute(average(scores)))

print(scores_f)

DF = pd.DataFrame(scores_f)
DF.to_csv("sumbission.csv", index=False, header=False)
