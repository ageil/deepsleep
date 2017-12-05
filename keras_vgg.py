#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 10:45:00 2017

Imagenet implementation using the plug & play Keras VGGnet for the convolutional
layers, with dense layers on top.

@author: allin
"""

import keras.initializers as initializers
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import Callback
from keras import backend as K
from keras import regularizers

import os
import pickle
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from skimage import io
import skimage.transform
from sklearn.metrics import confusion_matrix

# test: 0:1 validation: 2:5, train: 6:19
folds = np.array([[10,14,1,13,15,19,12,17,5,9,2,4,16,3,11,18,8,20,7,6],
[20,18,13,9,3,11,16,1,19,5,4,14,7,6,15,17,2,12,8,10],
[16,13,2,9,15,8,18,14,7,17,19,6,11,20,3,1,4,5,10,12],
[17,4,1,20,5,16,8,7,2,18,10,15,14,19,13,12,9,11,3,6],
[2,11,6,15,8,12,5,4,14,13,7,19,3,10,9,18,16,17,1,20],
[19,6,10,2,11,15,7,3,9,13,5,20,17,4,1,14,8,18,16,12],
[15,9,5,6,13,19,2,17,18,3,11,12,4,20,1,10,14,16,7,8],
[5,12,9,1,17,13,15,11,20,19,18,7,6,14,4,8,10,16,3,2],
[8,3,10,16,12,1,6,15,20,13,19,18,4,14,9,11,7,2,17,5],
[1,7,3,15,10,6,18,4,5,12,19,20,2,13,16,14,8,9,11,17]])
folds = folds-1


# Number of output classes
num_classes = 5
# Number of subjects
num_subjects = 20
# Number of units in dense layers
num_dense = 4096
# Number of units in LSTM layers
num_lstm = 128
# Name of sensor used
sensors = 'fpz'
# Path to spectrograms
impath = '../imdata/'
init_seed = 3
n_epochs = 2
batch_size = 75



def build_model(init_seed=None):
    # CONVOLUTIONAL LAYERS: use the pretrained weights from ImageNet
    base_model = VGG16(weights='imagenet', input_shape=(224, 224, 3), include_top=False)
    
    # TOP LAYERS: Trainable dense layers with random initialization
    dense_model = Sequential()
    # Input layer
    dense_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    # FC6
    dense_model.add(Dense(num_dense, activation='relu', kernel_initializer=initializers.glorot_normal(seed=init_seed), kernel_regularizer=regularizers.l2(0.01)))
    # FC6 Dropout
    dense_model.add(Dropout(rate=0.5))
    # FC7
    dense_model.add(Dense(num_dense, activation='relu', kernel_initializer=initializers.glorot_normal(seed=init_seed), kernel_regularizer=regularizers.l2(0.01)))
    # FC7 Dropout
    dense_model.add(Dropout(rate=0.5))
    # Softmax
    dense_model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_normal(seed=init_seed), kernel_regularizer=regularizers.l2(0.01)))

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
    adam = Adam(lr = 0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    return model


# LOAD SPECTROGRAMS (copy-pasted from Albert)
def load_spectrograms(subject_id, night_id):
    labels = np.loadtxt(impath +'sub'+str(subject_id)+'_n'+str(night_id)+'_img_'+sensors+'/labels.txt',dtype=bytes).astype(str)
    num_images = np.size(labels)
    
    targets = np.zeros((num_images), dtype='int')
    targets[:]=-1    
    targets[labels=='W'] = 0
    targets[labels=='1'] = 1
    targets[labels=='2'] = 2
    targets[labels=='3'] = 3
    targets[labels=='4'] = 3
    targets[labels=='R'] = 4
    
    if max(targets)>4:
        print('Error in reading targets')
    
    targets = targets[targets!=-1]
    num_images = np.size(targets)
    
    inputs = np.zeros((num_images,224,224,3),dtype='uint8')

    for idx in range(1,num_images+1):    
        rawim = io.imread(impath + 'sub'+str(subject_id)+'_n'+str(night_id)+'_img_'+sensors+'/img_'+ np.str(idx) +'.png')
        rawim = rawim[:,:,0:3]
        
        h, w, _ = rawim.shape
        if not (h==224 and w==224):
            rawim = skimage.transform.resize(rawim, (224, 224), preserve_range=True)
               
        inputs[idx-1,:,:,:]=rawim
    
    return inputs, targets


def get_subjects_list():
    subjects_list = []
    for i in range(1,num_subjects+1):
        print("Loading subject %d..." %(i))
        inputs_night1, targets_night1  = load_spectrograms(i,1)
        if(i!=20):
            inputs_night2, targets_night2  = load_spectrograms(i,2)
        else:
            inputs_night2 = np.empty((0,224,224,3),dtype='uint8')
            targets_night2 = np.empty((0,),dtype='uint8')           
        
        current_inputs = np.concatenate((inputs_night1,inputs_night2),axis=0)
        current_targets = np.concatenate((targets_night1, targets_night2),axis=0)
        subjects_list.append([current_inputs,current_targets])
    
    return subjects_list


# Split the dataset into training, validation and test set
def split_dataset(subjects_list, idx): 

    # Split the list in 3 subsets
    idx_test = idx[0:1]
    idx_val = idx[2:5]
    idx_train = idx[6:19]
    
    # Training
    train_data = [subjects_list[i] for i in idx_train]
    inputs_train = np.empty((0,224,224,3),dtype='uint8')  
    targets_train = np.empty((0,),dtype='uint8') 
    for item in train_data:
        inputs_train = np.concatenate((inputs_train,item[0]),axis=0)
        targets_train = np.concatenate((targets_train,item[1]),axis=0)
    # Convert labels to categorical one-hot encoding
    targets_train = to_categorical(targets_train, num_classes=5)
            
    # Validation
    val_data = [subjects_list[i] for i in idx_val]
    inputs_val = np.empty((0,224,224,3),dtype='uint8')  
    targets_val = np.empty((0,),dtype='uint8')
    for item in val_data:
            inputs_val = np.concatenate((inputs_val,item[0]),axis=0)
            targets_val = np.concatenate((targets_val,item[1]),axis=0)
    # Convert labels to categorical one-hot encoding
    targets_val = to_categorical(targets_val, num_classes=5)
            
    # Test
    test_data = [subjects_list[i] for i in idx_test]
    inputs_test = np.empty((0,224,224,3),dtype='uint8')  
    targets_test = np.empty((0,),dtype='uint8')
    for item in test_data:
        inputs_test = np.concatenate((inputs_test,item[0]),axis=0)
        targets_test = np.concatenate((targets_test,item[1]),axis=0)  
    # Convert labels to categorical one-hot encoding
    targets_test = to_categorical(targets_test, num_classes=5)
    
    return inputs_train, targets_train, inputs_val, targets_val, inputs_test, targets_test

     
# Weight the classes according to their frequency in the dataset
def get_class_weights(targets_train):
    
    n_samples = np.sum(targets_train, axis=0)
    n_total = np.sum(n_samples)
    #
    w = [n_total/n_class for n_class in n_samples]
    wsum = np.sum(w)
    w = [10*wclass/wsum for wclass in w]
        
    class_weights = {}
    keys = range(5)
    for i in keys:
        class_weights[i] = w[i]
    
    return class_weights


# Function that plots training and validation error/accuracy
def plot_training_history(history, fold, plotpath, show=False):
                
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        filename = plotpath+'/fold' + str(fold) + '_acc.png'
        plt.savefig(filename, dpi=72)
        if show:
            plt.show()
        plt.close()
        
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        filename = plotpath+'/fold' + str(fold) + '_loss.png'
        plt.savefig(filename, dpi=72)
        if show:
            plt.show()
        plt.close()

 
# Custom Keras Callback class to continuously calculate test error.
# Error is only calculated if the epoch has lower validation error than all previous epochs. 
# Test error and accuracy can be accessed through TestOnBest.history       
class TestOnBest(Callback):
    # Initialize dict to store test loss and accuracy, and some tracker variables
    def __init__(self, test_data):
        self.test_data = test_data
        self.best_loss = float('inf')
        self.current_loss = float('inf')
        self.best_weights = []
        self.confusion_matrix = []
        self.history = {'test_acc': []}

    # At the end of epoch, check if validation loss is the best so far    
    def on_epoch_end(self, batch, logs={}):
        self.current_loss = logs.get('val_loss')
        # If validation error is lowest, compute test error and store in dict
        if (self.current_loss < self.best_loss):
            self.best_loss = self.current_loss
            x, y = self.test_data
            
            # Get predictions
            y_pred = self.model.predict(x)
            # probability to hard class assignment
            y_pred = to_categorical(np.argmax(y_pred, axis=1), num_classes=num_classes)
            
            # Calculate accuracy (checked: equals accuracy obtained from model.fit())
            acc = y_pred + y
            acc = np.sum(acc==2, dtype='float32')/np.shape(acc)[0]
            print ('Test acc: {}\n'.format(acc))
            self.history['test_acc'].append(acc)
            
            # Calculate confusion matrix
            # Convert from one-hot to integer labels
            y_pred = np.argmax(y_pred, axis=1)
            y = np.argmax(y, axis=1)
            m = confusion_matrix(y, y_pred, labels=None, sample_weight=None)
            self.confusion_matrix = m
            # Get model weights and store in temporary variable
            self.best_weights = model.get_weights()
            
        else:
            self.history['test_acc'].append(float('nan'))

            
if __name__ == '__main__':    

    # Make sure output paths are in place
    plotpath = '../plots'
    if not os.path.exists(plotpath):
            os.makedirs(plotpath)   
    outpath = '../outputs'
    if not os.path.exists(outpath):
            os.makedirs(outpath)


    # Read in all the data
    subjects_list = get_subjects_list()
    
    # Build model, save initial weights (TODO: Save/load only trainable weights?)
    model = build_model(init_seed)
    # print summary
    model.summary()
    # save initial weights
    Winit = model.get_weights()
    
   
    for fold in range(10):
        
        print("Fold num %d" %(fold))
        #f = open('outputs/sleep5_fold'+str(fold), 'w').close()
        
        # reset weights to initial (common initialization for all folds)
        model.set_weights(Winit)
        
        # Get training, validation, test set
        inputs_train, targets_train, inputs_val, targets_val, inputs_test, targets_test = split_dataset(subjects_list, folds[fold])
    
        # Get class weights for the current training set
        class_weights = get_class_weights(targets_train)     
        
        # Call testing history callback
        test_history = TestOnBest((inputs_test, targets_test))
        # Run training
        history = model.fit(inputs_train, targets_train, epochs=n_epochs, batch_size=batch_size, class_weight=class_weights,
        validation_data = (inputs_val, targets_val), callbacks=[test_history], verbose=2)
        
        # Retreive test set statistics and merge to training statistics log
        history.history.update(test_history.history)
    
        # Save training history
        fn = '../outputs/train_hist_fold'+str(fold)+'.pickle'
        pickle_out = open(fn,'wb')
        pickle.dump(history.history, pickle_out)
        pickle_out.close()
        fn_hist = fn
        
        # Save confusion matrix
        fn = '../outputs/confusion_matrix_fold'+str(fold)+'.pickle'
        pickle_out = open(fn,'wb')
        pickle.dump(test_history.confusion_matrix, pickle_out)
        pickle_out.close()
        fn_mat = fn
        
        # Save weights after training is finished
        model.set_weights(test_history.best_weights)
        fn = '../outputs/weights_fold'+str(fold)+'.hdf5'
        model.save_weights(fn)
        
        # Plot loss and accuracy by epoch
        plot_training_history(history, fold, plotpath)
                    
        fold+=1  
    
    
    # Clear session
    K.clear_session()
    
