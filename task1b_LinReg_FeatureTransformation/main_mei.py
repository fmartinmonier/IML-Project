from sklearn.linear_model import LinearRegression
from numpy import average, absolute
import numpy as np
import pandas as pd  
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedKFold
from matplotlib import pyplot as plt

#load dataset
train = pd.read_csv("train.csv" )

y = train['y']
normalize_y = y / np.linalg.norm(y)
X = train [['x1', 'x2', 'x3', 'x4', 'x5']]
size=y.size
print(type(X))
X = X.to_numpy()

square = np.power(X, 2)
exponential = np.exp(X)
cos = np.cos(X)

finale = np.concatenate((X, square, exponential, cos, np.ones((X.shape[0],1))), axis = 1)
print(np.shape(finale))
normalize1 = finale / np.linalg.norm(finale)
df = pd.DataFrame(normalize1)
#df = pd.DataFrame(finale)

model = Ridge(alpha=0.025)
#lm = LinearRegression()
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
y_pred = cross_validate(model, df, normalize_y, cv=cv, return_estimator=True)
model.fit(normalize1,normalize_y)
predictions = model.predict(normalize1)

counter = 0
newmodel_coef_ = np.zeros((finale.shape[1],1))
for model in y_pred['estimator']:
    newmodel_coef_ = newmodel_coef_ + model.coef_
    
print(newmodel_coef_.shape)
weights = np.mean(newmodel_coef_, axis = 0)

weights = np.transpose(weights)
print(weights)

plt.figure(figsize=(12, 6))
plt.plot(predictions, 'r+')
plt.plot(normalize_y, 'bo')
plt.show()

DF = pd.DataFrame(weights)
DF.to_csv("sumbission.csv", index=False, header=False)
