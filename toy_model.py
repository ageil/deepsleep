#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 11:24:12 2017

@author: allin
"""

# Visualize training history
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy
from keras.callbacks import Callback
import pickle

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
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model


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

# Save weights after training is finished
fn = '../outputs/weights_fold'+str(fold)+'.hdf5'
model.save_weights(fn)

# Plot loss and accuracy by epoch
plot_training_history(history, fold)

# Try and load the history log
try_load_hist = pickle.load(open(fn_hist, 'rb'))

