import pandas as pd 
import numpy as np
import os

import tensorflow as tf 
from tensorflow import keras

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

#import matplotlib.pyplot as plt

#sources: https://www.bmc.com/blogs/keras-neural-network-classification/
#        https://medium.com/nerd-for-tech/how-to-train-neural-networks-for-image-classification-part-1-21327fe1cc1
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #to suppress a pseky error message 

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

################################################
#Creating a neural network model

def define_model():
    model = keras.Sequential(
        [
            keras.layers.Dense(512, activation="relu", input_shape=(80,)),
            keras.layers.Dense(512, activation="relu"),
            #keras.layers.Dense(20, activation="relu", kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1/15, seed=None)),
            #keras.layers.Dense(10, activation="relu", kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1/20, seed=None)),
            keras.layers.Dropout(.1),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model

#define the model
model = define_model()
model.summary()

#opt = optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

################################################
#Training the neural network model
#weights = {0:1, 1:5}
print('Training the model')
#history = model.fit(encoded_X_train, Y_train, class_weight=weights, epochs=200, batch_size=2048, verbose=1)
model.fit(encoded_X_train, Y_train, epochs=200, batch_size=2048, verbose=1)
print('Finished training the model')

#pd.DataFrame(history.history)
#plt.grid(True)
#plt.gca().set_ylim(0, 1)
#plt.show()

predictions = model.predict_classes(encoded_X_test)

DF = pd.DataFrame(predictions)
DF.to_csv('submission.csv', index=False, header=False)