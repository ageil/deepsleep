#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 10:45:00 2017

Imagenet implementation using the plug & play Keras VGGnet for the convolutional
layers, with dense layers on top.

@author: allin
"""
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
from skimage import io

# Number of output classes
num_classes = 5
# Name of sensor used
sensors='fpz'
# Path to spectrograms
impath = '../imdata/'

# CONVOLUTIONAL LAYERS: use the weights from ImageNet
base_model = VGG16(weights='imagenet', input_shape=(224, 224, 3), include_top=False)

# TOP LAYERS: Trainable dense layers with random initialization
dense_model = Sequential()
# Input layer
dense_model.add(Flatten(input_shape=base_model.output_shape[1:]))
# FC6
dense_model.add(Dense(4096, activation='relu'))
# FC6 Dropout
dense_model.add(Dropout(rate=0.5))
# FC7
dense_model.add(Dense(4096, activation='relu'))
# FC7 Dropout
dense_model.add(Dropout(rate=0.5))
# FC8
dense_model.add(Dense(num_classes, activation=None))
# Softmax
dense_model.add(Dense(num_classes, activation='softmax'))



# CREATE THE FULL MODEL (stack the dense layers on the convolutional layers)
model = Sequential()
# Add convolutional layers
for layer in base_model.layers:
    model.add(layer)
# Fix weights in the convolutional layers
for layer in model.layers:
    layer.trainable = False
# Add fully conncected layers
for layer in dense_model.layers:
    model.add(layer)


# COMPILE THE MODEL
# Optimizer (can't figure out what learning rate Albert used)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

    
# print summary
model.summary()


# TRY PASSING AN IMAGE through the network 
img_path = '/Users/allin/02456-deep-learning/DeepSleep/imdata/sub1_n1_img_fpz/img_1.png'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
features = model.predict(x)

# LOAD SPECTROGRAMS (copy-pasted from Albert)
def load_spectrograms(subject_id, night_id):
    labels = np.loadtxt(impath +'sub'+str(subject_id)+'_n'+str(night_id)+'_img_'+sensors+'/labels.txt',dtype='str')
    num_images = np.size(labels)
    
    targets = np.zeros((num_images), dtype='uint8')
    targets[:]=-1    
    targets[labels=='W'] = 0
    targets[labels=='1'] = 1
    targets[labels=='2'] = 2
    targets[labels=='3'] = 3
    targets[labels=='4'] = 4
    targets[labels=='R'] = 5
    
    targets = targets[targets!=-1]
    num_images = np.size(targets)
    
    inputs = np.zeros((num_images,3,224,224),dtype='uint8')

    for idx in range(1,num_images+1):    
        rawim = io.imread(impath + 'sub'+str(subject_id)+'_n'+str(night_id)+'_img_'+sensors+'/img_'+ np.str(idx) +'.png')
        rawim = rawim[:,:,0:3]
        
        h, w, _ = rawim.shape
        if not (h==224 and w==224):
            rawim = skimage.transform.resize(rawim, (224, 224), preserve_range=True)
        
        # Shuffle axes to c01
        im = np.transpose(rawim,(2,0,1))
        
        im = im[np.newaxis]        
        inputs[idx-1,:,:,:]=im
    
    
    return inputs, targets

#TODO: Define the training with LOO as in Albert's code