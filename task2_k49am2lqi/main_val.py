import pandas as pd 
from sklearn.svm import SVC
import numpy as np
svclassifier_task1 = SVC(kernel='sigmoid') 
svclassifier_task2 = SVC(kernel='sigmoid')

#How to deal with nan data through pre-processing step?

#load dataset
train_feature = pd.read_csv("train_features.csv" )
train_labels = pd.read_csv("train_labels.csv" )
test_feature = pd.read_csv("test_features.csv" )

pid = train_feature['pid']
pid_test = test_feature['pid']
add_info = train_feature[['Time', 'Age']]
add_info_test = test_feature[['Time', 'Age']]
task_1 = train_feature [['BaseExcess', 'Fibrinogen', 'AST', 'Alkalinephos', 'Bilirubin_total', 'Lactate', 'TroponinI', 'SaO2', 'Bilirubin_total','EtCO2']]
task_1_y = train_labels [['BaseExcess', 'Fibrinogen', 'AST', 'Alkalinephos', 'Bilirubin_total', 'Lactate', 'TroponinI', 'SaO2', 'Bilirubin_total','EtCO2']]
task_1_test = test_feature [['BaseExcess', 'Fibrinogen', 'AST', 'Alkalinephos', 'Bilirubin_total', 'Lactate', 'TroponinI', 'SaO2', 'Bilirubin_total','EtCO2']]
task_2 = train_feature ['Sepsis']
task_2_y = train_labels ['Sepsis']
task_2_test = test_feature ['Sepsis']
task_3 = train_feature [['RRate', 'ABPm', 'SpO2', 'Heartrate']]
task_3_y = train_labels [['RRate', 'ABPm', 'SpO2', 'Heartrate']]
task_3_test = test_feature [['RRate', 'ABPm', 'SpO2', 'Heartrate']]
#train_feature = train_labels.to_numpy()
#train_labels = train_labels.to_numpy()
task_1 = task_1.to_numpy()
task_1_y = task_1_y.to_numpy()
task_1_test = task_1_test.to_numpy
task_2 = task_2.to_numpy()
task_2_y = task_2_y.to_numpy()
task_2_test = task_2_test.to_numpy()
task_3 = task_3.to_numpy()
task_3_y = task_3_y()
task_3_test = task_3_test.to_numpy()
print('train_labels', np.size(train_labels,0), 'task1', np.size(task_1,0))

pid_list = pid.to_numpy()
pid_list = np.unique(pid_list)
pid_list_test = pid_test.to_numpy()
pid_list_test = np.unique(pid_list_test)

#Task 1
print('pid_list', np.size(pid_list,0))

#Training
pid_attribute_t1 = []
added_t1 = []
pid_attribute_t2 = []
added_t2 = []
i=0
for i in range(np.size(pid_list,0)):
    j=0
    for j in range(np.size(task_1,0)):
        while pid_list[i] == train_feature[j,0]:
            added_t1 = np.concatenate((add_info[j,],task_1[j,]), axis=1)
            pid_attribute_t1 = np.concatenate((pid_attribute_t1, added_t1), axis=0)
            added_t2 = np.concatenate((add_info[j,],task_2[j,]), axis=1)
            pid_attribute_t2 = np.concatenate((pid_attribute_t2, added_t2), axis=0)
            j = j + 1

        svclassifier_task_1.fit(pid_attribute_t1, task_1_y[i])
        svclassifier_task_2.fit(pid_attribute_t2, task_2_y[i])
        added_t1 = []
        added_t2 = []
        pid_attribute_t1 = []
        pid_attribute_t2 = []

#Testing

pid_attribute_t1 = []
added_t1 = []
pid_attribute_t2 = []
added_t2 = []
i=0
result_final =[]
for i in range(np.size(pid_list,0)):
    j=0
    for j in range(np.size(task_1,0)):
        while pid_list[i] == train_feature[j,0]:
            added_t1 = np.concatenate((add_info_test[j,],task_1_test[j,]), axis=1)
            pid_attribute_t1 = np.concatenate((pid_attribute_t1, added_t1), axis=0)
            added_t2 = np.concatenate((add_info[j,],task_2[j,]), axis=1)
            pid_attribute_t2 = np.concatenate((pid_attribute_t2, added_t2), axis=0)
            j = j + 1

        task1_pred=svclassifier_task_1.predict(pid_attribute_t1)
        task2_pred=svclassifier_task_2.predict(pid_attribute_t2)
        added_t1 = []
        added_t2 = []
        pid_attribute_t1 = []
        pid_attribute_t2 = []
        result = np.concatenate((task1_pred, task2_pred), axis=1)
        result_final = np.concatenate((result_final, result), axis=0)

result_final = np.concatenate((pid_list_test, result_final), axis=1)
#task3 regression missing 