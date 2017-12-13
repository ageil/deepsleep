#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 13:33:59 2017

@author: allin
"""

import numpy as np
from skimage import io
import skimage.transform
import pickle
import sys

# Name of sensor used
sensors = 'fpz'
# Path to spectrograms
impath = '../imdata/'
init_seed = 3

num_subjects = 20

timesteps = 3
batch_size = 16


# Get subject range from terminal
smin = 1 # int(sys.argv[1]) 
smax = 20 #int(sys.argv[2]) 


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
        if i != 20:
            inputs_night2, targets_night2  = load_spectrograms(i,2)
        else:
            inputs_night2 = np.empty((0,224,224,3),dtype='uint8')
            targets_night2 = np.empty((0,),dtype='uint8')           
        
        current_inputs = np.concatenate((inputs_night1,inputs_night2),axis=0)
        current_targets = np.concatenate((targets_night1, targets_night2),axis=0)
        subjects_list.append([current_inputs,current_targets])
    
    return subjects_list


def timebatcher(timesteps, i, batch_size):

    idx_start = int(np.floor(timesteps/2))
    idx_end = int(np.ceil(timesteps/2))


    # load spectrograms for subject
    inputs_night1, targets_night1  = load_spectrograms(i,1)
    if i != 20:
        inputs_night2, targets_night2  = load_spectrograms(i,2)
    else:
        inputs_night2 = np.empty((0,224,224,3),dtype='uint8')
        targets_night2 = np.empty((0,),dtype='uint8')
    inputs = np.concatenate((inputs_night1,inputs_night2),axis=0)
    targets = np.concatenate((targets_night1, targets_night2),axis=0)
    
    print("\tPacking data for subject", i, "...")
    
    num_batches = int(np.ceil((inputs.shape[0]-timesteps)/batch_size))
    timedata = np.empty((batch_size, timesteps, 224, 224, 3), dtype='uint8')
    labels = np.empty((batch_size,), dtype='uint8')
    
    for b in range(num_batches):

        idx_low = idx_start + b * batch_size
        if b == (num_batches-1):
            idx_high = inputs.shape[0] - idx_end
        else:
            idx_high = idx_start + (b+1) * batch_size
        
        # pack subject data in batches of size (timesteps x 224 x 224 x 3)
        for idx in range(idx_low, idx_high):
            # get the index of current example in batch
            batch_idx = idx - (b * batch_size + idx_start)
            # center batch on labelled spectrogram
            timebatch = inputs[idx-idx_start:idx+idx_end]
            timedata[batch_idx] = timebatch
            labels[batch_idx] = targets[idx]

        if (idx_high -idx_low < batch_size):
            timedata = timedata[:(idx_high - idx_low),:,:,:,:]
            labels = labels[:(idx_high - idx_low),]

        # one-hot encode labels
        onehot_labels = np.zeros((labels.shape[0], 5))
        for idx, el in enumerate(labels):
            onehot_labels[idx, el] = 1
        
        name = "sub" + str(i) + "_batch" + str(b) + ".pickle"
        with open("../databatches/" + name, 'wb') as f:
            pickle.dump((timedata, onehot_labels), f)



if __name__ == '__main__': 
    
    for subject in range(smin, smax+1):
        print('Processing subject ', subject, '...')
        timebatcher(timesteps, subject, batch_size)
        

