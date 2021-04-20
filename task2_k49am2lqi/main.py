import pandas as pd 
from sklearn.svm import SVC, SVR
import numpy as np

#Defining classification and regression models for tasks 1,2 and 3
svclassifier_task1 = SVC(kernel='sigmoid', probability=True) 
svclassifier_task2 = SVC(kernel='sigmoid', probability=True)
svregression_task3 = SVR(kernel='sigmoid')


vitals = ['RRate', 'ABPm', 'SpO2', 'Heartrate']
VITALS = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']

tests = ['BaseExcess', 'Fibrinogen', 'AST', 'Alkalinephos', 'Bilirubin_total', 'Lactate', 'TroponinI', 'SaO2', 'Bilirubin_total','EtCO2']
TESTS = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
         'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
         'LABEL_Bilirubin_direct', 'LABEL_EtCO2']

#load dataset
train_features = pd.read_csv("train_features.csv" )
train_labels = pd.read_csv("train_labels.csv" )
test_features = pd.read_csv("test_features.csv" )

#Replacing NaN values with means of their respective columns (mean computed over all patient data)
train_features = train_features.fillna(train_features.mean())

pid = train_features['pid']
pid_test = test_features['pid']

pid_list = pid.to_numpy()
pid_list = np.unique(pid_list)
pid_list_test = pid_test.to_numpy()
pid_list_test = np.unique(pid_list_test)
print('pid_list', np.size(pid_list,0))

#Creating a new time collumn, with timesteps for each patient being in range(1,12)
time = {'Time':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}
time = pd.DataFrame(time,columns=['Time'])
time_repeated = pd.concat([time]*np.size(pid_list,0), ignore_index=True)
train_features['Time'] = time_repeated['Time']


#TRAINING

#Task 1 --> For-loop commented out due to long execution time

#Merging both training and testin datasets on the patient id number
all_training_data = pd.merge_ordered(train_features, train_labels, on='pid')

svclassifier_task1_ = {}

#for label in TESTS:
    #k = 0
    #test_type = tests[k]
    #print('Current test type being trained:', test_type)
    #X_train = all_training_data.pivot_table(index="pid", columns="Time", values=test_type).to_numpy()
    #y_train = all_training_data.pivot_table(index="pid", values=label).to_numpy()
    #svclassifier_task1_[str(label)] = svclassifier_task1.fit(X_train, y_train.ravel())


#Task 2
label = ['LABEL_Sepsis']
X_train_task2 = all_training_data.pivot_table(index="pid", columns="Time").to_numpy()
y_train_task2 = all_training_data.pivot_table(index="pid", values=label).to_numpy()
svclassifier_task2_sepsis = svclassifier_task2.fit(X_train_task2, y_train_task2.ravel())

#Task 3 --> To do

#TESTING --> To do. The following code is the old version, using numpy instead on pandas for the data manipulation.

pid_attribute_t1 = []
pid_attribute_t2 = []
pid_attribute_t3 = []

i=0
result_final =[]
for i in range(np.size(pid_list,0)):
    j=0
    for j in range(np.size(task_1,0)):
        while pid_list[i] == train_feature[j,0]:
            pid_attribute_t1 = np.vstack((pid_attribute_t1, test_feature[j,:]))
            pid_attribute_t2 = np.concatenate((pid_attribute_t2, test_feature[j,:]), axis=0)
            pid_attribute_t3 = np.concatenate((pid_attribute_t3, test_feature[j,:]), axis=0)
            j = j + 1

    task1_pred=svclassifier_task1.predict(pid_attribute_t1)
    task2_pred=svclassifier_task2.predict(pid_attribute_t2)
    task3_pred = svregression_task3.predict(pid_attribute_t3)
    pid_attribute_t1 = []
    pid_attribute_t2 = []
    pid_attribute_t3 = []

    result = np.concatenate((task1_pred, task2_pred, task3_pred), axis=1)
    result_final = np.concatenate((result_final, result), axis=0)

result_final = np.concatenate((pid_list_test, result_final), axis=1)
DF = pd.DataFrame(result_final)
DF.to_csv("sumbission.csv", index=False, header=False)
