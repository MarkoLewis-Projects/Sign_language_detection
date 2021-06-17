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
oh_labels = tf.constant(to_categorical(training_data['encoded_labels'],298,dtype=int))
#print(pd.DataFrame(data=oh_labels,columns=np.linspace(0,298,298,dtype=str)))
#training_data = training_data.append(oh_labels)
#print(training_data)


### testing numpy frame aggregating

cap = cv2.VideoCapture('training_arr/about10757_clipped.avi')
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
        frames.append(frame)
        
video = np.stack(frames,axis=0)
#print(video)
frames,length,width,channels = video.shape
print(np.linspace(0,11,16,dtype=int))

#print(video[1,1,1,1])

video = video[list(np.linspace(0,11,16,dtype=int))]
print(video.shape)
    


cap.release()
cv2.destroyAllWindows()

    


