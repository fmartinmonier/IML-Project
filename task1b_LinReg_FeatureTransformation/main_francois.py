import pandas as pd 
import numpy as np 
import sys
from matplotlib import pyplot as plt 

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer

#import training set
training_set = pd.read_csv('train.csv')

#define labels y, their id and features x
id = np.take(training_set, [0], axis=1)
y = np.take(training_set, [1], axis=1)
X = np.take(training_set, [2, 3, 4, 5, 6], axis=1)

#Linear features
lin_model = LinearRegression().fit(X,y)
lin_intercept = lin_model.intercept_
lin_coef = lin_model.coef_
lin_predictions = lin_model.predict(X) #predict values based on model

#Quadratic features
X_trans_quad = FunctionTransformer(lambda x: x**2).fit_transform(X)
quad_model = LinearRegression().fit(X_trans_quad,y)
quad_coef = quad_model.coef_
quad_predictions = quad_model.predict(X_trans_quad)

#Exponential features
X_trans_exp = FunctionTransformer(np.exp).fit_transform(X)
exp_model = LinearRegression().fit(X_trans_exp,y)
exp_coef = exp_model.coef_
exp_predictions = exp_model.predict(X_trans_exp)

#Cosine features
X_trans_cos = FunctionTransformer(np.cos).fit_transform(X)
cos_model = LinearRegression().fit(X_trans_cos,y)
cos_coef = cos_model.coef_
cos_predictions = cos_model.predict(X)

#Constant features
#const_model = LinearRegression().fit(1,y)
#const_coef = const_model.coef_
#const_predictions = const_model.predict(X)



plt.figure(figsize=(12, 6))
plt.plot(cos_predictions, 'r+')
plt.plot(y, 'bo')
plt.show()

