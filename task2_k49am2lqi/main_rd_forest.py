import pandas as pd 
from sklearn.svm import SVC, SVR
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=2, weights="uniform")
imputer1 = KNNImputer(n_neighbors=2, weights="uniform")
#Defining classification and regression models for tasks 1,2 and 3



vitals = ['RRate', 'ABPm', 'SpO2', 'Heartrate']
VITALS = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']

tests = ['BaseExcess', 'Fibrinogen', 'AST', 'Alkalinephos', 'Bilirubin_total', 'Lactate', 'TroponinI', 'SaO2', 'Bilirubin_total','EtCO2']
TESTS = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total','LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2','LABEL_Bilirubin_direct', 'LABEL_EtCO2']
head = [ 'pid','LABEL_BaseExcess','LABEL_Fibrinogen','LABEL_AST','LABEL_Alkalinephos','LABEL_Bilirubin_total','LABEL_Lactate','LABEL_TroponinI','LABEL_SaO2','LABEL_Bilirubin_direct','LABEL_EtCO2','LABEL_Sepsis','LABEL_RRate','LABEL_ABPm','LABEL_SpO2','LABEL_Heartrate']

#load dataset
train_features = pd.read_csv("train_features.csv" )
train_labels = pd.read_csv("train_labels.csv" )
test_features = pd.read_csv("test_features.csv" )

train_features = imputer.fit_transform(train_features)
test_features = imputer1.fit_transform(test_features)
DF = pd.DataFrame(train_features)
DF.to_csv("train_f_test.csv", index=False, header=False, float_format='%.3f')



pid = train_features['pid']
pid_list=pid.unique() #applying .unique() takes unique value repeated + transform in ndarray
pid_test = test_features['pid']
pid_list_test = pid_test.unique()




#Creating a new time collumn, with timesteps for each patient being in range(1,12)
time = {'Time':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}
time = pd.DataFrame(time,columns=['Time'])
time_repeated = pd.concat([time]*np.size(pid_list,0), ignore_index=True)
train_features['Time'] = time_repeated['Time']

time_repeated_test = pd.concat([time]*np.size(pid_list_test,0), ignore_index=True)
test_features['Time'] = time_repeated_test['Time']



############################################################################
#TRAINING
print(len(pid_list))
######################################
#Task 1 --> For-loop commented out due to long execution time
svclassifier_task1_ = {}
k = 0 #integer increment at each iterator of the for loop to iterate on the "tests" list
result_t1 = np.expand_dims(pid_list_test, axis=1)
for label in TESTS:
    test_type = tests[k]
    print('Current test type being trained:', test_type)
    X_train_task1 = train_features.pivot_table(index="pid", columns="Time", values=test_type).to_numpy()
    y_train_task1 = train_labels.pivot_table(index="pid", values=label).to_numpy()
    X_test_task1 = test_features.pivot_table(index="pid", columns="Time", values=test_type).to_numpy()
    rfclassifier_t1=RandomForestClassifier(n_estimators=100)
    rfclassifier_t1.fit(X_train_task1, y_train_task1.ravel())
    pred_t1 = rfclassifier_t1.predict_proba(X_test_task1)
    
    result_t1 = np.concatenate((result_t1, np.expand_dims(pred_t1[:,1], axis=1)), axis=1)
    
    k += 1

#print(result_t1)

######################################
#Task 2
label = ['LABEL_Sepsis']
X_train_task2 = train_features.pivot_table(index="pid", columns="Time").to_numpy()
y_train_task2 = train_labels.pivot_table(index="pid", values=label).to_numpy()
rfclassifier_t2=RandomForestClassifier(n_estimators=100)
X_test_task2 = test_features.pivot_table(index="pid", columns="Time").to_numpy()
rfclassifier_t2.fit(X_train_task2, y_train_task2.ravel())
pred_t2 = rfclassifier_t2.predict_proba(X_test_task2)

result_t2 = np.concatenate((result_t1, np.expand_dims(pred_t2[:,1], axis=1)), axis=1)
#print(result_t2)

######################################
#Task 3 

k = 0 #integer increment at each iterator of the for loop to iterate on the "vitals" list
result_t3 = result_t2
for label in VITALS:
    vital_type = vitals[k]
    
    X_train_task3 = train_features.pivot_table(index="pid", columns="Time", values=vital_type).to_numpy()
    y_train_task3 = train_labels.pivot_table(index="pid", values=label).to_numpy()
    X_test_task3 = test_features.pivot_table(index="pid", columns="Time", values=vital_type).to_numpy()
    
    rfregressor_t3=RandomForestRegressor(n_estimators=100)
    rfregressor_t3.fit(X_train_task3,y_train_task3.ravel())
    pred_t3 = rfregressor_t3.predict(X_test_task3)
    result_t3 = np.concatenate((result_t3, np.expand_dims(pred_t3, axis=1)), axis=1)
    k += 1

final_result = np.concatenate((np.expand_dims(head, axis=0), result_t3), axis=0)
DF = pd.DataFrame(final_result)
DF.to_csv("sumbission.csv", index=False, header=False, float_format='%.3f')