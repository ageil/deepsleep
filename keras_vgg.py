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
from keras.layers import Dense, Dropout, Flatten, LSTM
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import Callback
from keras import backend as K

import os
import pickle
import matplotlib.pyplot as plt
import random
import numpy as np
from skimage import io
import skimage.transform
from sklearn.model_selection import LeaveOneOut


# Number of output classes
num_classes = 5
# Number of subjects
num_subjects = 10
# Number of units in dense layers
num_dense = 4096
# Number of units in LSTM layers
num_lstm = 128
# Name of sensor used
sensors = 'fpz'
# Path to spectrograms
impath = '../imdata/'
init_seed = 3
n_epochs = 50
batch_size = 75



def build_model(init_seed=None):
    # CONVOLUTIONAL LAYERS: use the pretrained weights from ImageNet
    base_model = VGG16(weights='imagenet', input_shape=(224, 224, 3), include_top=False)
    
    # TOP LAYERS: Trainable dense layers with random initialization
    dense_model = Sequential()
    # Input layer
    dense_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    # FC6
    dense_model.add(Dense(num_dense, activation='relu', kernel_initializer=initializers.glorot_normal(seed=init_seed)))
    # FC6 Dropout
    dense_model.add(Dropout(rate=0.5))
    # FC7
    dense_model.add(Dense(num_dense, activation='relu', kernel_initializer=initializers.glorot_normal(seed=init_seed)))
    # FC7 Dropout
    dense_model.add(Dropout(rate=0.5))
    # Softmax
    dense_model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_normal(seed=init_seed)))
    
    # RNN LAYERS
    rnn_model = Sequential()
    # Input layer
    rnn_model.add(Flatten(input_shape=dense_model.output_shape[1:]))
    # LSTM1
    rnn_model.add(LSTM(num_lstm, activate='relu'))
    # LSTM2
    rnn_model.add(LSTM(num_lstm, activate='relu'))
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
    # Add LSTM layers
    for layer in rnn_model.layers:
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
    labels = np.loadtxt(impath +'sub'+str(subject_id)+'_n'+str(night_id)+'_img_'+sensors+'/labels.txt',dtype='str')
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
            inputs_night2 = np.empty((0,3,224,224),dtype='uint8')
            targets_night2 = np.empty((0,),dtype='uint8')           
        
        current_inputs = np.concatenate((inputs_night1,inputs_night2),axis=0)
        current_targets = np.concatenate((targets_night1, targets_night2),axis=0)
        subjects_list.append([current_inputs,current_targets])
    
    return subjects_list


# Split the dataset into training, validation and test set
def split_dataset(subjects_list, idx_tmp, idx_test): 

    # Shuffle indices
    random.shuffle(idx_tmp)
    idx_train = idx_tmp[0:15]
    idx_val = idx_tmp[15:19]
#    num_subjects_train = np.size(idx_train)
#    num_subjects_val = np.size(idx_val)
#    num_subjects_test = np.size(idx_test)
    
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
        self.history = {'test_loss': [], 'test_acc': []}

    # At the end of epoch, check if validation loss is the best so far    
    def on_epoch_end(self, batch, logs={}):
        self.current_loss = logs.get('val_loss')
        # If validation error is lowest, compute test error and store in dict
        if (self.current_loss < self.best_loss):
            self.best_loss = self.current_loss
            x, y = self.test_data
            loss, acc = self.model.evaluate(x, y, verbose=0)
            self.history['test_loss'].append(loss)
            self.history['test_acc'].append(acc)
            print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
        # Else store NaNs (to keep correct epoch indexing)
        else:
            self.history['test_loss'].append(float('nan'))
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
    
    
    loo=LeaveOneOut()
    fold=1
    for idx_tmp, idx_test in loo.split(range(num_subjects)):
        
        print("Fold num %d\tSubject id %d" %(fold, idx_test+1))
        #f = open('outputs/sleep5_fold'+str(fold), 'w').close()
        
        # reset weights to initial (common initialization for all folds)
        model.set_weights(Winit)
        
        # Get training, validation, test set
        inputs_train, targets_train, inputs_val, 
        targets_val, inputs_test, targets_test = split_dataset(subjects_list, idx_tmp, idx_test)
    
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
        fn = outpath+'/train_hist_fold'+str(fold)+'.pickle'
        pickle_out = open(fn,'wb')
        pickle.dump(history.history, pickle_out)
        pickle_out.close()
        
        # Save weights after training is finished
        fn = outpath+'/weights_fold'+str(fold)+'.hdf5'
        model.save_weights(fn)
        
        # Plot loss and accuracy by epoch
        plot_training_history(history, fold, plotpath)
                    
        fold+=1  
    
    
    # Clear session
    K.clear_session()
    
