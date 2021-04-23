import pandas as pd 
from sklearn.svm import SVC, SVR
import numpy as np
from sklearn.impute import KNNImputer


vitals = ['RRate', 'ABPm', 'SpO2', 'Heartrate']
VITALS = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']

tests = ['BaseExcess', 'Fibrinogen', 'AST', 'Alkalinephos', 'Bilirubin_total', 'Lactate', 'TroponinI', 'SaO2', 'Bilirubin_total','EtCO2']
TESTS = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total','LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2','LABEL_Bilirubin_direct', 'LABEL_EtCO2']
head = [ 'pid','LABEL_BaseExcess','LABEL_Fibrinogen','LABEL_AST','LABEL_Alkalinephos','LABEL_Bilirubin_total','LABEL_Lactate','LABEL_TroponinI','LABEL_SaO2','LABEL_Bilirubin_direct','LABEL_EtCO2','LABEL_Sepsis','LABEL_RRate','LABEL_ABPm','LABEL_SpO2','LABEL_Heartrate']

#load dataset
train_features_raw = pd.read_csv("train_features.csv" )
train_labels = pd.read_csv("train_labels.csv" )
test_features_raw = pd.read_csv("test_features.csv" )

imputer = KNNImputer(n_neighbors=5, weights="uniform")

print('Imputing train_features...')
train_features_imp = imputer.fit_transform(train_features_raw)
train_features = pd.DataFrame(train_features_imp)
train_features.to_csv("KNN_train_features.csv", index=False, header=head)

print('Imputing test_feature...')
test_features_imp = imputer.fit_transform(test_features_raw)
test_features = pd.DataFrame(test_features_imp)
test_features.to_csv("KNN_test_features.csv", index=False, header=head)

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

    svclassifier_task1 = SVC(kernel='rbf', gamma='scale', C=1.0, probability=True)

    svclassifier_task1.fit(X_train_task1, y_train_task1.ravel())
    pred_t1 = svclassifier_task1.predict_proba(X_test_task1)
    print(pred_t1[:,1])
    
    result_t1 = np.concatenate((result_t1, np.expand_dims(pred_t1[:,1], axis=1)), axis=1)
    
    k += 1

#print(result_t1)

######################################
#Task 2
label = ['LABEL_Sepsis']
X_train_task2 = train_features.pivot_table(index="pid", columns="Time").to_numpy()
y_train_task2 = train_labels.pivot_table(index="pid", values=label).to_numpy()
X_test_task2 = test_features.pivot_table(index="pid", columns="Time").to_numpy()

svclassifier_task2 = SVC(kernel='rbf', gamma='scale', C=1.0, probability=True)

svclassifier_task2.fit(X_train_task2, y_train_task2.ravel())
pred_t2 = svclassifier_task2.predict_proba(X_test_task2)
print(pred_t2[:,1])

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
    
    svregression_task3 = SVR(kernel='rbf', gamma='scale', C=1.0) 

    svregression_task3.fit(X_train_task3, y_train_task3.ravel())
    pred_t3 = svregression_task3.predict(X_test_task3)
    print(pred_t3)

    result_t3 = np.concatenate((result_t3, np.expand_dims(pred_t3, axis=1)), axis=1)
    k += 1

final_result = np.concatenate((np.expand_dims(head, axis=0), result_t3), axis=0)
DF = pd.DataFrame(final_result)
DF.to_csv("sumbission.csv", index=False, header=False, float_format='%.3f')