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
num_subjects = 1 # 10
# Path to spectrograms
impath = '/Users/anders1991/imdata/'
init_seed = 3
n_epochs = 50
batch_size = 75
timesteps = 5

def build_model(init_seed=None):
    # Input layer
    img_input = Input(shape=(None, 5, 224, 224, 3)) # 224x224x3

    # CNN (TimeDistributed VGG16)
    cnn = Sequential()

    # Block 1
    cnn.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'), name='block1_tdist_conv1', input_shape=(timesteps, 224, 224, 3))) # 224x224x64
    cnn.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'), name='block1_tdist_conv2')) # 224x224x64
    cnn.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)), name='block1_tdist_pool')) # 112x112x64

    # Block 2
    cnn.add(TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'), name='block2_tdist_conv1')) # 112x112x128
    cnn.add(TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'), name='block2_tdist_conv2')) # 112x112x128
    cnn.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)), name='block2_tdist_pool')) # 56x56x128

    # Block 3
    cnn.add(TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same'), name='block3_tdist_conv1')) # 56x56x256
    cnn.add(TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same'), name='block3_tdist_conv2')) # 56x56x256
    cnn.add(TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same'), name='block3_tdist_conv3')) # 56x56x256
    cnn.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)), name='block3_tdist_pool')) # 28x28x256

    # Block 4
    cnn.add(TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same'), name='block4_tdist_conv1')) # 28x28x512
    cnn.add(TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same'), name='block4_tdist_conv2')) # 28x28x512
    cnn.add(TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same'), name='block4_tdist_conv3')) # 28x28x512
    cnn.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)), name='block4_tdist_pool')) # 14x14x512

    # Block 5
    cnn.add(TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same'), name='block5_tdist_conv1')) # 14x14x512
    cnn.add(TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same'), name='block5_tdist_conv2')) # 14x14x512
    cnn.add(TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same'), name='block5_tdist_conv3')) # 14x14x512
    cnn.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)), name='block5_tdist_pool')) # 7x7x512


    # OPTIONAL: Classification block
    xavier = initializers.glorot_normal(seed=init_seed)
    cnn_top = Sequential()
    cnn_top.add(TimeDistributed(Flatten(), input_shape=cnn.output_shape[1:], name="block6_tdist_flatten"))
    cnn_top.add(TimeDistributed(Dense(4096, activation='relu', kernel_initializer=xavier), name="block6_tdist_dense1"))
    cnn_top.add(TimeDistributed(Dropout(rate=0.5), name="block6_tdist_dropout1"))
    cnn_top.add(TimeDistributed(Dense(4096, activation='relu', kernel_initializer=xavier), name="block6_tdist_dense2"))
    cnn_top.add(TimeDistributed(Dropout(rate=0.5), name="block6_tdist_dropout2"))
    cnn_top.add(TimeDistributed(Dense(num_classes, activation='softmax', kernel_initializer=xavier), name="block6_tdist_softmax"))


    # LCRN
    lcrn = Sequential()
    # add VGG 16 base layers
    for layer in cnn.layers:
        lcrn.add(layer)

    # load weights
    WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                                        WEIGHTS_PATH,
                                        cache_subdir='models',
                                        file_hash='64373286793e3c8b2b4e3219cbf3544b')
    weights_path_no_top = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                        WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models',
                                        file_hash='6d6bbae143d832006294945121d1f1fc')
    lcrn.load_weights(weights_path_no_top)
    
    # freeze VGG16 weights
    for layer in lcrn.layers:
        layer.trainable = False
    # add VGG 16 classification layers
    for layer in cnn_top.layers:
        lcrn.add(layer)


    # COMPILE THE MODEL
    # Optimizer (can't figure out what learning rate Albert used)
    adam = Adam(lr = 0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    lcrn.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    return lcrn


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


# LOAD SPECTROGRAMS (copy-pasted from Albert)
def load_spectrograms(subject_id, night_id):
    labels = np.loadtxt(impath +'sub'+str(subject_id)+'_n'+str(night_id)+'_img_'+ 'fpz' +'/labels.txt',dtype='str')
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
        rawim = io.imread(impath + 'sub'+str(subject_id)+'_n'+str(night_id)+'_img_'+'fpz'+'/img_'+ np.str(idx) +'.png')
        rawim = rawim[:,:,0:3]
        
        h, w, _ = rawim.shape
        if not (h==224 and w==224):
            rawim = skimage.transform.resize(rawim, (224, 224), preserve_range=True)
               
        inputs[idx-1,:,:,:]=rawim
    
    return inputs, targets


if __name__ == '__main__':           

    # Read in all the data
    # subjects_list = get_subjects_list()
    
    # Build model, save initial weights (TODO: Save/load only trainable weights?)
    model = build_model(init_seed)
    print(model.summary())

    # save initial weights
    Winit = model.get_weights()