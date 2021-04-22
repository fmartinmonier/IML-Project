import pandas as pd 
from sklearn.svm import SVC, SVR
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer

imputer = KNNImputer(n_neighbors=2, weights="uniform")
imputer1 = KNNImputer(n_neighbors=2, weights="uniform")

#Defining classification and regression models for tasks 1,2 and 3
#svclassifier_task1 = SVC(kernel='rbf', class_weight={1: 10}) 
#svclassifier_task2 = SVC(kernel='sigmoid', probability=True)
#svregression_task3 = SVR(kernel='sigmoid')


vitals = ['RRate', 'ABPm', 'SpO2', 'Heartrate']
VITALS = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']

#add Base_excess before submission
tests = ['BaseExcess', 'Fibrinogen', 'AST', 'Alkalinephos', 'Bilirubin_total', 'Lactate', 'TroponinI', 'SaO2', 'Bilirubin_total','EtCO2']
TESTS = ['LABEL_BaseExcess','LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total','LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2','LABEL_Bilirubin_direct', 'LABEL_EtCO2']
head = [ 'pid','LABEL_BaseExcess','LABEL_Fibrinogen','LABEL_AST','LABEL_Alkalinephos','LABEL_Bilirubin_total','LABEL_Lactate','LABEL_TroponinI','LABEL_SaO2','LABEL_Bilirubin_direct','LABEL_EtCO2','LABEL_Sepsis','LABEL_RRate','LABEL_ABPm','LABEL_SpO2','LABEL_Heartrate']

#load dataset
train_features = pd.read_csv("train_features_imp.csv" )
train_labels = pd.read_csv("train_labels.csv" )
test_features = pd.read_csv("test_features_imp.csv" )

'''features = train_features.columns.values

# average over 12h for each patient
# use mean over all patients if no values are availible for any of the 12h
def imp(df_in):
    df_in_med_np = np.empty(df_in.shape)
    
    i = 0
    for pid in df_in['pid'].unique():
        imp_mean = SimpleImputer(strategy='mean')
        patient = df_in.loc[df_in['pid'] == pid]
        
        nonempty_columns =  []
        temp = pd.DataFrame(columns = features)
        
        for columns in features:
            # all the values for this feature are null
            if sum(patient[columns.isnull()) != patient.shape[0]:
                nonempty_columns.append(columns)
        
        imp_mean_patient = imp_mean.fit_transform(patient)
        imp_mean_patient = pd.DataFrame(data=b, columns = nonempty_columns)
        temp = temp.merge(imp_mean_patient, how='outer')
        temp = temp[features]
        
        c = temp.to_numpy(copy=False, na_value=np.nan)
        df_in_med_np[12*i:12*i+12,:] = c
        
        i += 1
    df_in_med = pd.DataFrame(data=df_in_med_np, columns = features)
    
    # second impute -> mean from all patients if one test has never been done
    # for a given patient
    imp_mean = SimpleImputer(strategy='mean')
    df_in_imp_np = imp_mean.fit_transform(df_in_med)
    df_in_imp = pd.DataFrame(
        data=df_in_imp_np, columns = features)
    
    return df_in_imp


print('Imputing train_features')
train_features_imp = imp(train_features)
print('Imputing test_features')
test_features_imp = imp(test_features)

train_features_imp.to_csv('train_features_imp.csv', index=False, float_format='%.3f')
test_features_imp.to_csv('test_features_imp.csv', index=False, float_format='%.3f')
train_features = pd.read_csv('train_features_imp.csv', delimiter=',')
test_features = pd.read_csv('test_features_imp.csv', delimiter=',')
'''

#Replacing NaN values with means of their respective columns (mean computed over all patient data)
#train_features = train_features.fillna(train_features.mean())
#test_features = test_features.fillna(test_features.mean())

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

#Merging both training and testin datasets on the patient id number
all_training_data = pd.merge_ordered(train_features, train_labels, on='pid')

############################################################################
#TRAINING
print(len(pid_list))
######################################

#Task 1 --> For-loop commented out due to long execution time

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
    pred_t1 = pred_t1[:,1]
    print(pred_t1)
    result_t1 = np.concatenate((result_t1, np.expand_dims(pred_t1, axis=1)), axis=1)
    k += 1


###########################################################################
######################################
#Task 2

label = ['LABEL_Sepsis']
X_train_task2 = train_features.pivot_table(index="pid", columns="Time").to_numpy()
y_train_task2 = train_labels.pivot_table(index="pid", values=label).to_numpy()
X_test_task2 = test_features.pivot_table(index="pid", columns="Time").to_numpy()

svclassifier_task2 = SVC(kernel='rbf', gamma='scale', C=1.0, probability=True)

svclassifier_task2.fit(X_train_task2, y_train_task2.ravel())
pred_t2 = svclassifier_task2.predict_proba(X_test_task2)
pred_t2 = pred_t2[:,1]
print(pred_t2)

result_t2 = np.concatenate((result_t1, np.expand_dims(pred_t2, axis=1)), axis=1)


###########################################################################
######################################
#Task 3 

k = 0 #integer increment at each iterator of the for loop to iterate on the "vitals" list
result_t3 = result_t2
for vital_label in VITALS:
    vital_type = vitals[k]
    print('Current vital being trained:', vital_type)
    X_train_task3 = train_features.pivot_table(index="pid", columns="Time", values=vital_type).to_numpy()
    y_train_task3 = train_labels.pivot_table(index="pid", values=vital_label).to_numpy()
    X_test_task3 = test_features.pivot_table(index="pid", columns="Time", values=vital_type).to_numpy()

    svregression_task3 = SVR(kernel='rbf', gamma='scale', C=1.0) 

    svregression_task3.fit(X_train_task3, y_train_task3.ravel())
    pred_t3 = svregression_task3.predict(X_test_task3)
    print(pred_t3)
    result_t3 = np.concatenate((result_t3, np.expand_dims(pred_t3, axis=1)), axis=1)
    k += 1


final_result = np.concatenate((np.expand_dims(head, axis=0), result_t3), axis=0)
DF = pd.DataFrame(final_result)
compression_opts = dict(method='zip', archive_name='predictions.csv')
DF.to_csv('prediction.zip', index=False, header=False, float_format='%.3f', compression=compression_opts)