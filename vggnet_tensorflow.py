#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 13:15:22 2017

@author: allin
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout

from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix
import skimage.transform
from skimage import io
import pickle
import imp
# utils = imp.load_source('utils', 'utils.py')

#Constants
num_epochs = 50
minibatch_size = 75 #250
lr = 0.00001
num_subjects = 10 #num_subjects_train + num_subjects_val + num_subjects_test
sensors='fpz'
train_features = False
pre_trained_weights = True


# Build the network
height = 224
width = 224
nchannels = 3
num_classes = 5

def build_model():
    tf.reset_default_graph()
    
    kernel_sz = (3,3)
    pool_sz = (2,2)
    stride = (1,1)
    pad = 'valid'
    train = False
    
    x_pl = tf.placeholder(tf.float32, [None, height, width, nchannels], name='xPlaceholder')
    y_pl = tf.placeholder(tf.float64, [None, num_classes], name='yPlaceholder')
    y_pl = tf.cast(y_pl, tf.float32)
    

    with tf.variable_scope('conv1_1'):
        ## TODO: Don't flip filters?
        print('x_pl \t\t', x_pl.get_shape())
        conv1_1 = Conv2D(filters=64, kernel_size=kernel_sz, strides=stride, padding=pad, activation='relu', trainable=train)
        x = conv1_1(x_pl)
        print('conv1_1 \t\t', x.get_shape())
        
    with tf.variable_scope('conv1_2'):
        conv1_2 = Conv2D(filters=64, kernel_size=kernel_sz, strides=stride, padding=pad, activation='relu', trainable=train)
        x = conv1_2(x)
        print('conv1_2 \t\t', x.get_shape())
        
    with tf.variable_scope('pool1'):
        pool1 = MaxPooling2D(pool_size=pool_sz, strides=None, padding=pad)
        x = pool1(x)
        print('pool1 \t\t', x.get_shape())
        
    with tf.variable_scope('conv2_1'):
        conv2_1 = Conv2D(filters=128, kernel_size=kernel_sz, strides=stride, padding=pad, activation='relu', trainable=train)
        x = conv2_1(x)
        print('conv2_1 \t\t', x.get_shape())
        
    with tf.variable_scope('conv2_2'):
        conv2_2 = Conv2D(filters=128, kernel_size=kernel_sz, strides=stride, padding=pad, activation='relu', trainable=train)
        x = conv2_2(x)
        print('conv2_2 \t\t', x.get_shape())
        
    with tf.variable_scope('pool2'):
        pool2 = MaxPooling2D(pool_size=pool_sz, strides=None, padding=pad)
        x = pool2(x)
        print('pool2 \t\t', x.get_shape())
        
    with tf.variable_scope('conv3_1'):
        conv3_1 = Conv2D(filters=256, kernel_size=kernel_sz, strides=stride, padding=pad, activation='relu', trainable=train)
        x = conv3_1(x)
        print('conv3_1 \t\t', x.get_shape())
        
    with tf.variable_scope('conv3_2'):
        conv3_2 = Conv2D(filters=256, kernel_size=kernel_sz, strides=stride, padding=pad, activation='relu', trainable=train)
        x = conv3_2(x)
        print('conv3_2 \t\t', x.get_shape())
        
    with tf.variable_scope('conv3_3'):
        conv3_3 = Conv2D(filters=256, kernel_size=kernel_sz, strides=stride, padding=pad, activation='relu', trainable=train)
        x = conv3_3(x)
        print('conv3_3 \t\t', x.get_shape())
        
    with tf.variable_scope('pool3'):
        pool3 = MaxPooling2D(pool_size=pool_sz, strides=None, padding=pad)
        x = pool3(x) 
        print('pool3 \t\t', x.get_shape())
        
    with tf.variable_scope('conv4_1'):
        conv4_1 = Conv2D(filters=512, kernel_size=kernel_sz, strides=stride, padding=pad, activation='relu', trainable=train)
        x = conv4_1(x)
        print('conv4_1 \t\t', x.get_shape())
        
    with tf.variable_scope('conv4_2'):
        conv4_2 = Conv2D(filters=512, kernel_size=kernel_sz, strides=stride, padding=pad, activation='relu', trainable=train)
        x = conv4_2(x)
        print('conv4_2 \t\t', x.get_shape())
        
    with tf.variable_scope('conv4_3'):
        conv4_3 = Conv2D(filters=512, kernel_size=kernel_sz, strides=stride, padding=pad, activation='relu', trainable=train)
        x = conv4_3(x)
        print('conv4_3 \t\t', x.get_shape())
        
    with tf.variable_scope('pool4'):
        pool4 = MaxPooling2D(pool_size=pool_sz, strides=None, padding=pad)
        x = pool4(x)
        print('pool4 \t\t', x.get_shape())
        
    with tf.variable_scope('conv5_1'):
        conv5_1 = Conv2D(filters=512, kernel_size=kernel_sz, strides=stride, padding=pad, activation='relu', trainable=train)
        x = conv5_1(x)
        print('conv5_1 \t\t', x.get_shape())
        
    with tf.variable_scope('conv5_2'):
        conv5_2 = Conv2D(filters=512, kernel_size=kernel_sz, strides=stride, padding=pad, activation='relu', trainable=train)
        x = conv5_2(x)
        print('conv5_2 \t\t', x.get_shape())
        
    with tf.variable_scope('conv5_3'):
        conv5_3 = Conv2D(filters=512, kernel_size=kernel_sz, strides=stride, padding=pad, activation='relu', trainable=train)
        x = conv5_3(x)
        print('conv5_3 \t\t', x.get_shape())
        
    with tf.variable_scope('pool5'):
        pool5 = MaxPooling2D(pool_size=pool_sz, strides=None, padding=pad)
        x = pool5(x)
        print('pool5 \t\t', x.get_shape())
        
    with tf.variable_scope('fc6'):
        fc6 = Dense(4096, activation='relu')
        x = fc6(x)
        print('fc6 \t\t', x.get_shape())
        
    with tf.variable_scope('fc6_dropout'):
        drop6 = Dropout(rate=0.5)
        x = drop6(x)
        print('fc6_dropout \t\t', x.get_shape())
        
    with tf.variable_scope('fc7'):
        fc7 = Dense(4096, activation='relu')
        x = fc7(x)
        print('fc7 \t\t', x.get_shape())
        
    with tf.variable_scope('fc7_dropout'):
        drop7 = Dropout(rate=0.5)
        x = drop7(x)
        print('fc7_dropout \t\t', x.get_shape())
        
    with tf.variable_scope('fc8'):
        fc8 = Dense(num_classes, activation='relu')
        x = fc8(x)
        print('fc8 \t\t', x.get_shape())
        
    with tf.variable_scope('prob'):
        soft = Dense(num_classes, activation='softmax')
        y_out = soft(x)
        print('y_out \t\t', y_out.get_shape())
        
        
        