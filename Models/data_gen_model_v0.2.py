# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 16:39:04 2021

@author: Lewis Jones & Marko Pancic
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import glob
import re
import math
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import make_multilabel_classification
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
# import tensorflow_hub as hub
from sklearn.metrics import mean_absolute_error,mean_squared_error
from tensorflow.keras.models import load_model
from model import model

weight_dir = 'weights_C3D_sports1M_tf.h5'


#_____________________________________________________________________________

def calculate_mean_std(x, channels_first=False, verbose=0):
    """
    Calculates channel-wise mean and std
    
    Parameters
    ----------
    x : array
        Array representing a collection of images (frames) or
        collection of collections of images (frames) - namely video
    channels_first : bool, optional
        Leave False, by default False
    verbose : int, optional
        1-prints out details, 0-silent mode, by default 0
    
    Returns
    -------
    array of shape [2, num_channels]
        Array with per channel mean and std for all the frames
    """
    ndim = x.ndim
    assert ndim in [5,4]
    assert channels_first == False
    all_mean = []
    all_std = []    
    num_channels = x.shape[-1]
    
    for c in range(0, num_channels):
        if ndim ==5: # videos
            mean = x[:,:,:,:,c].mean()
            std = x[:,:,:,:,c].std()
        elif ndim ==4: # images rgb or grayscale
            mean = x[:,:,:,c].mean()
            std = x[:,:,:,c].std()
        if verbose:
            print("Channel %s mean before: %s" % (c, mean))   
            print("Channel %s std before: %s" % (c, std))
            
        all_mean.append(mean)
        all_std.append(std)
    
    return np.stack((all_mean, all_std))


def preprocess_input(x, mean_std, divide_std=False, channels_first=False, verbose=0):
    """
    Channel-wise substraction of mean from the input and optional division by std
    
    Parameters
    ----------
    x : array
        Input array of images (frames) or videos
    mean_std : array
        Array of shape [2, num_channels] with per-channel mean and std
    divide_std : bool, optional
        Add division by std or not, by default False
    channels_first : bool, optional
        Leave False, otherwise not implemented, by default False
    verbose : int, optional
        1-prints out details, 0-silent mode, by default 0
    
    Returns
    -------
    array
        Returns input array after applying preprocessing steps
    """
    x = np.asarray(x, dtype=np.float32)    
    ndim = x.ndim
    assert ndim in [5,4]
    assert channels_first == False
    num_channels = x.shape[-1]
    
    for c in range(0, num_channels):  
        if ndim ==5: # videos
            x[:,:,:,:,c] -= mean_std[0][c]
            if divide_std:
                x[:,:,:,:,c] /= mean_std[1][c]
            if verbose:
                print("Channel %s mean after preprocessing: %s" % (c, x[:,:,:,:,c].mean()))    
                print("Channel %s std after preprocessing: %s" % (c, x[:,:,:,:,c].std()))
        elif ndim ==4: # images rgb or grayscale
            x[:,:,:,c] -= mean_std[0][c]
            if divide_std:
                x[:,:,:,c] /= mean_std[1][c]   
            if verbose:        
                print("Channel %s mean after preprocessing: %s" % (c, x[:,:,:,c].mean()))    
                print("Channel %s std after preprocessing: %s" % (c, x[:,:,:,c].std()))            
    return x

#_____________________________________________________________________________


print(tf.version.VERSION)
# i3d = hub.load("https://tfhub.dev/deepmind/i3d-kinetics-400/1").signatures['default']

files = glob.glob('training_arr/*.avi')
print(str(files[5]))

training_labels = []
training_files = []


for file in files:
    label = re.findall('[A-Za-z]+[0-9]',str(file))[0][:-1]
    training_labels.append(label)
    training_files.append(str(file))
    
# print(training_labels)
# print(training_files)

training_data = pd.DataFrame({'filename':training_files,'training_labels':training_labels})
print(training_data)

print(training_data)
label_encoder = LabelEncoder().fit_transform(training_data['training_labels'])
training_data['encoded_labels'] = label_encoder
print(training_data)


### testing numpy frame aggregating

videos_data = []


for file in training_data['filename'].iloc[0:50]:
    
    training_data_practice = training_data.iloc[0:50]
    
    resize=(112, 112)
    
    print(str(file))
    cap = cv2.VideoCapture(str(file))
    ret = True
    frames=[]

    while ret == True:
        ret,frame = cap.read()
        print(ret)
        try:
            print(frame.shape)
        except:
            continue
        if ret == True:
            frame = cv2.resize(frame,resize)
            frames.append(frame)
        
    video = np.stack(frames,axis=0)
    frames,length,width,channels = video.shape


    video = video[list(np.linspace(0,frames-1,16,dtype=int))]
    
    #print(f'video before {video[0,:,:,0]}')
    mean_std = calculate_mean_std(video, channels_first=False, verbose=0)
    #print(f'std_mean shape: {mean_std.shape}')
    video = preprocess_input(video, mean_std, divide_std=False, channels_first=False, verbose=0)
    #print(f'video after {video[0,:,:,0]}')
    videos_data.append(video)
    cap.release()
    
print(np.array(videos_data).shape)

#training_data_practice['data'] = videos_data
print(training_data_practice.head())

cv2.destroyAllWindows()


#-----------------------------------------------------------------------------

conv_model = model(weight_dir, trainable = True, freeze_layer = 0)

conv_model = conv_model.retrainable_model((3, 3, 3), (16, 112, 112, 3))


