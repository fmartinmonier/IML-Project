from sklearn.linear_model import LinearRegression
from numpy import average, absolute
import numpy as np
import pandas as pd  


#load dataset
train = pd.read_csv("train.csv" )

y = train['y']
X = train [['x1', 'x2', 'x3', 'x4', 'x5']]
size=y.size
X = X.to_numpy()

square = np.power(X, 2)
exponential = np.exp(X)
cos = np.cos(X)
print(type(square),type(X))
print("square", square.shape,"exponential", exponential.shape,"cos", square.shape)
#struggling to find numpy function that append properly X, square, exponential ect..
finale = np.concatenate([X, square, exponential, cos, np.ones(size)])

df = pd.DataFrame(finale, columns = ['Column_A','Column_B','Column_C', 'Column_D'])

model = LinearRegression()
model.fit(finale,y)



#DF = pd.DataFrame(scores_f)
#DF.to_csv("sumbission.csv", index=False, header=False)
