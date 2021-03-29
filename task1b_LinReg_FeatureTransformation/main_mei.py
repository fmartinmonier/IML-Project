from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from numpy import average, absolute
import numpy as np
import pandas as pd  
import math

#load dataset
train = pd.read_csv("train.csv" )
#ID, y, x1, x2, x3, x4, x5

y = train['y']
X = train[['x1', 'x2', 'x3', 'x4', 'x5']]
X_Quadratic = train[['x1', 'x2', 'x3', 'x4', 'x5']]
X_Exponential = train[['x1', 'x2', 'x3', 'x4', 'x5']]
X_Cosine = train[['x1', 'x2', 'x3', 'x4', 'x5']]
X_Constant = np.ones((700,1))

for idx, row in X.iterrows():
    X_Quadratic.loc[idx, 'x1'] = row['x1'] * row['x1']
    X_Exponential.loc[idx, 'x1'] = math.exp( row['x1'] )
    X_Cosine.loc[idx, 'x1'] = math.cos( row['x1'] )

    X_Quadratic.loc[idx, 'x2'] = row['x2'] * row['x2']
    X_Exponential.loc[idx, 'x2'] = math.exp( row['x2'] )
    X_Cosine.loc[idx, 'x2'] = math.cos( row['x2'] )

    X_Quadratic.loc[idx, 'x3'] = row['x3'] * row['x3']
    X_Exponential.loc[idx, 'x3'] = math.exp( row['x3'] )
    X_Cosine.loc[idx, 'x3'] = math.cos( row['x3'] )

    X_Quadratic.loc[idx, 'x4'] = row['x4'] * row['x4']
    X_Exponential.loc[idx, 'x4'] = math.exp( row['x4'] )
    X_Cosine.loc[idx, 'x4'] = math.cos( row['x4'] )

    X_Quadratic.loc[idx, 'x5'] = row['x5'] * row['x5']
    X_Exponential.loc[idx, 'x5'] = math.exp( row['x5'] )
    X_Cosine.loc[idx, 'x5'] = math.cos( row['x5'] )

arr = np.column_stack((X, X_Quadratic, X_Exponential, X_Cosine, X_Constant)) 

arr_final = np.zeros((700, 21))
    
#load weights
weights = np.ones((21, 1))
for i in range(0, 5):
    weights[i] = 0.01

for i in range(5, 10):
    weights[i] = 0.001    

for i in range(10, 15):
    weights[i] = 0.00001

for i in range(15, 20):
    weights[i] = 0.000001

weights[20] = 0.0000001            

for idx, row in enumerate(arr):
    for i in range(0, 21):
        arr_final[idx, i] = row[i] * weights[i]

#print(arr_final)        

scores_f = []
model = LinearRegression()
model.fit(arr_final,y)
#cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(model, arr_final, y, cv=10, scoring='neg_root_mean_squared_error') #cross validation for 10fold with RMSE
scores_f.append(absolute(average(scores)))

print(scores_f)

DF = pd.DataFrame(weights)
DF.to_csv("sample.csv", index=False, header=False)