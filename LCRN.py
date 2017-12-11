import keras.backend as K
import keras.initializers as initializers
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop, Adadelta
from keras.utils import to_categorical
from keras.callbacks import Callback
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.utils.data_utils import get_file
from keras import regularizers

import pickle
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
import sys
import numpy as np
import glob
from skimage import io
import skimage.transform
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight

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

class_weights = {0: 1.69217121,
 1: 2.76249095,
 2: 0.43409,
 3: 1.36469326,
 4: 0.98949553}

# Get fold range from terminal
fmin = int(sys.argv[1]) 
fmax = int(sys.argv[2]) 

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
# Model name (for output filenames)
mdname = 'LCRN1'
init_seed = 3
n_epochs = 2
batch_size = 70
tsteps = 3



def build_model(init_seed=None, cnn_softmax=False, timesteps=tsteps, droprate=0.5):
    # load weights data
    wpath = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    wpath_notop = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                                        wpath,
                                        cache_subdir='models',
                                        file_hash='64373286793e3c8b2b4e3219cbf3544b')
    weights_path_no_top = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                        wpath_notop,
                                        cache_subdir='models',
                                        file_hash='6d6bbae143d832006294945121d1f1fc')

    # CNN (TimeDistributed VGG16)
    cnn = Sequential()

    # Block 1
    cnn.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', trainable=False), name='block1_tdist_conv1', input_shape=(timesteps, 224, 224, 3))) # timesteps x 224x224x64
    cnn.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', trainable=False), name='block1_tdist_conv2')) # timesteps x 224x224x64
    cnn.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)), name='block1_tdist_pool')) # timesteps x 112x112x64

    # Block 2
    cnn.add(TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same', trainable=False), name='block2_tdist_conv1')) # timesteps x 112x112x128
    cnn.add(TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same', trainable=False), name='block2_tdist_conv2')) # timesteps x 112x112x128
    cnn.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)), name='block2_tdist_pool')) # timesteps x 56x56x128

    # Block 3
    cnn.add(TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', trainable=False), name='block3_tdist_conv1')) # timesteps x 56x56x256
    cnn.add(TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', trainable=False), name='block3_tdist_conv2')) # timesteps x 56x56x256
    cnn.add(TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', trainable=False), name='block3_tdist_conv3')) # timesteps x 56x56x256
    cnn.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)), name='block3_tdist_pool')) # timesteps x 28x28x256

    # Block 4
    cnn.add(TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', trainable=False), name='block4_tdist_conv1')) # timesteps x 28x28x512
    cnn.add(TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', trainable=False), name='block4_tdist_conv2')) # timesteps x 28x28x512
    cnn.add(TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', trainable=False), name='block4_tdist_conv3')) # timesteps x 28x28x512
    cnn.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)), name='block4_tdist_pool')) # timesteps x 14x14x512

    # Block 5
    cnn.add(TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', trainable=False), name='block5_tdist_conv1')) # timesteps x 14x14x512
    cnn.add(TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', trainable=False), name='block5_tdist_conv2')) # timesteps x 14x14x512
    cnn.add(TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', trainable=False), name='block5_tdist_conv3')) # timesteps x 14x14x512
    cnn.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)), name='block5_tdist_pool')) # timesteps x 7x7x512


    # OPTIONAL: Classification block (see param cnn_softmax)
    #cnn_top = Sequential()
    #xavier = initializers.glorot_normal(seed=init_seed)
    #cnn_top.add(TimeDistributed(Flatten(), input_shape=cnn.output_shape[1:], name="block6_tdist_flatten")) # timesteps x 25088
    #cnn_top.add(TimeDistributed(Dense(4096, activation='relu', kernel_initializer=xavier, kernel_regularizer=regularizers.l2(0.01)), name="block6_tdist_dense1")) # timesteps x 4096
    #cnn_top.add(TimeDistributed(Dropout(rate=droprate), name="block6_tdist_dropout1")) # timesteps x 4096
    #cnn_top.add(TimeDistributed(Dense(4096, activation='relu', kernel_initializer=xavier, kernel_regularizer=regularizers.l2(0.01)), name="block6_tdist_dense2")) # timesteps x 4096
    #cnn_top.add(TimeDistributed(Dropout(rate=droprate), name="block6_tdist_dropout2")) # timesteps x 4096
    #cnn_top.add(TimeDistributed(Dense(num_classes, activation='softmax', kernel_initializer=xavier, kernel_regularizer=regularizers.l2(0.01)), name="block6_tdist_softmax")) # timesteps x num_classes


    # RNN
    rnn = Sequential()
    rnn.add(TimeDistributed(Flatten(), input_shape=cnn.output_shape[1:], name="block7_tdist_flatten")) # timesteps x 25088; skip this line if including block 6
    # rnn.add(Bidirectional(LSTM(32, return_sequences=True), name="block7_bidir_lstm1", input_shape=(5, 4096))) # timesteps x (2x32); add this line if including block 6
    rnn.add(Bidirectional(LSTM(32, return_sequences=True), name="block7_bidir_lstm1")) # timesteps x (2x32)
    rnn.add(Dropout(rate=droprate, name="block7_dropout1")) # timesteps x (2x32)
    rnn.add(Bidirectional(LSTM(64, return_sequences=True), name="block7_bidir_lstm2")) # timesteps x (2x64)
    rnn.add(Dropout(rate=droprate, name="block7_dropout2")) # timesteps x (2x64)
    rnn.add(LSTM(num_classes, activation='softmax', return_sequences=False, name="block7_lstm3")) # 1 x num_classes

    # alternative classification using FC layers
#    rnn.add(LSTM(128, return_sequences=True, name="block7_bidir_lstm3")) # timesteps x 128
#    rnn.add(Flatten(name="block7_flatten")) # 1 x 640
#    rnn.add(Dense(num_classes, activation='softmax', kernel_initializer=xavier, name="block7_softmax")) # 1 x num_classes


    # Combine to LCRN
    lcrn = Sequential()
    # add VGG 16 base layers
    for layer in cnn.layers:
        lcrn.add(layer)
    # load VGG16 imagenet weights
    lcrn.load_weights(weights_path_no_top) # only base layers
    # add cnn classification layers
    # if cnn_softmax: # include softmax
        # for layer in cnn_top.layers:
            # lcrn.add(layer)
    # else: # exclude softmax
        # for layer in cnn_top.layers[:-1]: # skip last layer
            # lcrn.add(layer)
    # add rnn layers
    for layer in rnn.layers:
        lcrn.add(layer)


    # COMPILE THE MODEL
    # Optimizer (can't figure out what learning rate Albert used)
    adam = Adam(lr = 0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    lcrn.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    return lcrn

def data_gen(paths):
    while 1:
        for file in paths:
            with open(file, 'rb') as f:
                x, y = pickle.load(f)
            y_cat = np.argmax(y, axis=1)
            sample_weights = class_weight.compute_sample_weight(class_weights, y_cat, indices=None)
            yield x, y, sample_weights

def data_gen_test(paths):
    while 1:
        for file in paths:
            with open(file, 'rb') as f:
                x, y = pickle.load(f)
            yield x, y

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
            test_paths = self.test_data

            y_pred = np.empty((1,))
            y = np.empty((1,))
            # Get predictions
            for file in test_paths:
                with open(file, 'rb') as f:
                    x, y_batch = pickle.load(f)
                ypred_batch = self.model.predict_on_batch(self, x)
                np.column_stack((y_pred, ypred_batch))
                np.column_stack((y, y_batch))
            y_pred = y_pred[1:]
            y = y[1:]
            
            # probability to hard class assignment
            y_pred = to_categorical(np.argmax(y_pred, axis=1), num_classes=num_classes)

            # Calculate accuracy (checked: equals accuracy obtained from model.fit())
            acc = y_pred + y
            acc = np.sum(acc==2, dtype='float32')/np.shape(acc)[0]
            print('Test acc: {}\n'.format(acc))

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

    
    # Build model, save initial weights (TODO: Save/load only trainable weights?)
    model = build_model(init_seed)
    # print summary
    model.summary()
    # save initial weights
    Winit = model.get_weights()

    for fold in range(fmin, fmax+1):
    # for idx_tmp, idx_test in loo.split(range(num_subjects)):
        
        print("Fold num %d" %(fold))
        #f = open('outputs/sleep5_fold'+str(fold), 'w').close()
        
        # reset weights to initial (common initialization for all folds)
        model.set_weights(Winit)
        

        tr_paths = []
        for sub in folds[fold, 6:]:
            tr_paths = tr_paths + glob.glob("../databatches/sub" + str(sub+1) + "*.pickle")
        steps_per_ep = len(tr_paths)
            
        val_paths = []
        for sub in folds[fold, 2:6]:
            val_paths = val_paths + glob.glob("../databatches/sub" + str(sub+1) + "*.pickle")
        val_steps = len(val_paths)
            
        test_paths = []
        for sub in folds[fold, 0:2]:
            test_paths = test_paths + glob.glob("../databatches/sub" + str(sub+1) + "*.pickle")
    
        
        # Call testing history callback
        test_history = TestOnBest(test_paths)
        # Run training
        #history = model.fit(inputs_train, targets_train, epochs=n_epochs, batch_size=batch_size, sample_weight=sample_weights_tr,
        #validation_data = (inputs_val, targets_val, sample_weights_val), callbacks=[test_history], verbose=2)
        
        history = model.fit_generator(generator=data_gen(tr_paths), steps_per_epoch=steps_per_ep, epochs=3, verbose=2, callbacks=[test_history], 
                                      validation_data=data_gen(val_paths), validation_steps=val_steps, shuffle=False)
        
        # Retreive test set statistics and merge to training statistics log
        history.history.update(test_history.history)
    
        # Save training history
        fn = '../outputs/'+mdname+'train_hist_fold'+str(fold)+'.pickle'
        pickle_out = open(fn,'wb')
        pickle.dump(history.history, pickle_out)
        pickle_out.close()
        fn_hist = fn

        # Save confusion matrix
        fn = '../outputs/'+mdname+'confusion_matrix_fold'+str(fold)+'.pickle'
        pickle_out = open(fn, 'wb')
        pickle.dump(test_history.confusion_matrix, pickle_out)
        pickle_out.close()
        fn_mat = fn
        
        # Save weights after training is finished
        model.set_weights(test_history.best_weights)
        fn = '../outputs/'+mdname+'weights_fold'+str(fold)+'.hdf5'
        model.save_weights(fn)
        
        # Plot loss and accuracy by epoch
        plot_training_history(history, fold, plotpath)
                    
        fold+=1  
    
    
    # Clear session
    K.clear_session()
