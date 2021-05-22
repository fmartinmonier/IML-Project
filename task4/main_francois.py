import csv
import sys
import requests
import skimage.io
import os
import glob
import pickle 
import time
import helper
import shutil

import pandas as pd 
import numpy as np
import scipy.sparse as sp
import skimage.io

from PIL import Image
from IPython.display import display, HTML
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input 
from keras.preprocessing import image as kimage
from tqdm import tqdm
import matplotlib.pyplot as plt

#Sources:
# https://www.ethanrosenthal.com/2016/12/05/recasketch-keras/
# https://www.kaggle.com/bulentsiyah/dogs-vs-cats-classification-vgg16-fine-tuning


##########################################
#Preprocessing

#Import image and print out shape
rand_img = np.random.choice(glob.glob('food/00000.jpg'))
img = skimage.io.imread(rand_img)
print("Shape of input image is: ", img.shape)

#Preprocess image for inference in VGG16 model 
img = kimage.load_img(rand_img,target_size=(224,224)) #images must be resized to (224x224) for inference through VGG16
x = kimage.img_to_array(img)
x = np.expand_dims(x,axis=0) #extra channel added because Keras expects to receive mutliple models
x = preprocess_input(x)
print("Shape of preprocessed image is: ", x.shape)


#Load VGG16 model and feed image through
#We add include_top=False to remove the final fully connected layers.
#This allows us to forego the image recognition step and only output a feature map of the input image
model = VGG16(include_top=False, weights='imagenet') 

# Get feature map of input image 
tic = time.time()
pred = model.predict(x)
print(pred.shape )
print(pred.ravel().shape)
print("time required to infer one image is ", time.time()-tic, "seconds")

IMG_PATH = 'food_mini/'
IMG_NUM = len(os.listdir(IMG_PATH))
fnames = []
# split the data by train/val/test
for (n, FILE_NAME) in enumerate(os.listdir(IMG_PATH)):
    fnames.append(IMG_PATH + FILE_NAME)

batch_size = 4
min_idx = 0
max_idx = min_idx + batch_size
n_dims = pred.ravel().shape[0]
px = 224

# Initialize predictions matrix
preds = sp.lil_matrix((IMG_NUM, n_dims))

while min_idx < IMG_NUM - 1:
    t0 = time.time()
    
    X = np.zeros(((max_idx - min_idx), px, px, 3))
    
    # For each file in batch, 
    # load as row into X
    for i in range(min_idx, max_idx):
        fname = fnames[i]
        img = kimage.load_img(fname, target_size=(px, px))
        img_array = kimage.img_to_array(img)
        X[i - min_idx, :, :, :] = img_array
        if i % 200 == 0 and i != 0:
            t1 = time.time()
            print('{}: {}'.format(i, (t1 - t0) / i))
            t0 = time.time()
    max_idx = i
    t1 = time.time()
    print('{}: {}'.format(i, (t1 - t0) / i))
    
    print('Preprocess input')
    t0 = time.time()
    X = preprocess_input(X)
    t1 = time.time()
    print('{}'.format(t1 - t0))
    
    print('Predicting')
    t0 = time.time()
    these_preds = model.predict(X)
    shp = ((max_idx - min_idx) + 1, n_dims)
    
    # Place predictions inside full preds matrix.
    preds[min_idx:max_idx + 1, :] = these_preds.reshape(shp)
    t1 = time.time()
    print('{}'.format(t1 - t0))
    
    min_idx = max_idx
    max_idx = np.min((max_idx + batch_size, IMG_NUM))

def cosine_similarity(ratings):
    sim = ratings.dot(ratings.T)
    if not isinstance(sim, np.ndarray):
        sim = sim.toarray()
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)

preds = preds.tocsr()
sim = cosine_similarity(preds)
print(sim)
DF = pd.DataFrame(sim)
DF.to_csv('predictions.csv', index=False, header=False)

sample_test = test_df.sample(n=9).reset_index()
sample_test.head()
plt.figure(figsize=(12, 12))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = kimage.load_img("../input/test1/test1/"+filename, target_size=(256, 256))
    plt.subplot(3, 3, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')')
plt.tight_layout()
plt.show()