from sklearn.linear_model import LinearRegression
from numpy import average, absolute
import numpy as np
import pandas as pd  
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedKFold


#load dataset
train = pd.read_csv("train.csv" )

y = train['y']
X = train [['x1', 'x2', 'x3', 'x4', 'x5']]
size=y.size
print(type(X))
#Feature computation
X = X.to_numpy()

square = np.power(X, 2)
exponential = np.exp(X)
cos = np.cos(X)

finale = np.concatenate((X, square, exponential, cos, np.ones((X.shape[0],1))), axis = 1)

df = pd.DataFrame(finale)

model = Ridge(alpha=0.02)
y_pred = cross_validate(model, df, y, cv=10)
model.fit(finale,y)

#Weight extraction from our model
weights = model.coef_
weights = np.transpose(weights)




DF = pd.DataFrame(weights)
DF.to_csv("sumbission.csv", index=False, header=False)
