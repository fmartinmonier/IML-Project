import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
#load dataset
train = pd.read_csv("train.csv" )
test = pd.read_csv("test.csv")

X_train=train['Sequence']
Y_train=train['Active']

X_test=test['Sequence']

amino_acids = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'U', 'G', 'P', 'A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V']

#Encode categorical features as a one-hot numeric array: creates a binary column for each category
encoder = OneHotEncoder(sparse='False')

#Transform Xtrain in a list DNLI -> ['D', 'N', 'L', 'I']
split_Xtrain = []
for x_train in X_train:
    split_train = list(x_train)
    split_Xtrain.append(split_train)

split_Xtest = []
for x_test in X_test:
    split_test = list(x_test)
    split_Xtest.append(split_test)

#Where encoding takes place:
encoded_X_train = encoder.fit_transform(split_Xtrain)
encoded_X_test = encoder.fit_transform(split_Xtest)

#Test_1 : Random Forest
#rfclassifier=RandomForestClassifier(n_estimators=500)
#rfclassifier.fit(encoded_X_train, Y_train)
#preds=rfclassifier.predict(encoded_X_test)

#Test_2 : SVM 
clf = SVC(C=5)
model = clf.fit(encoded_X_train, Y_train)
preds = model.predict(encoded_X_test)

prediction = pd.DataFrame(preds)
prediction.to_csv('submission.csv', index=False, header=False)