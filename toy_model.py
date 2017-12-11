#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 11:24:12 2017

@author: allin
"""

# Visualize training history
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy
from keras.callbacks import Callback
import pickle
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
import numpy as np

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
        print(model.folds)

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
            y_pred = to_categorical(np.argmax(y_pred, axis=1), num_classes=2)
            
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
            
            
def plot_training_history(history, fold, show=False):
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        filename = '../plots/fold' + str(fold) + '_acc.png'
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
        filename = '../plots/fold' + str(fold) + '_loss.png'
        plt.savefig(filename, dpi=72)
        if show:
            plt.show()
        plt.close()


# fix random seed for reproducibility
seed = 7
fold = seed

numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
Y = to_categorical(Y, num_classes=2)
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', kernel_regularizer=regularizers.l2(0.01), activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', kernel_regularizer=regularizers.l2(0.01), activation='relu'))
model.add(Dense(2, kernel_initializer='uniform', kernel_regularizer=regularizers.l2(0.01), activation='sigmoid'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model

model.folds = 'hej'

# TRAIN MODEL
test_history = TestOnBest((X,Y))
history = model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=2, callbacks=[test_history])

history.history.update(test_history.history)
# list all data in history
print(history.history.keys())


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
plot_training_history(history, fold)

# Try and load the history log
try_load_hist = pickle.load(open(fn_hist, 'rb'))
try_load_mat = pickle.load(open(fn_mat, 'rb'))
