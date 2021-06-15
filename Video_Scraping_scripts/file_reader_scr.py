# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 19:55:27 2021

@author:lewis
"""
import pandas as pd
import numpy as np
import os
import PIL
import cv2
import imutils
import ffmpeg
import glob
from time import time
import numpy as np


files = glob.glob('training_videos/*.mp4')
clean = []
for file in files:
    clean.append(file[16:].split('_')[0])
   
df = pd.read_csv('training_files.csv').sort_values(by='file_name',ascending=True)
df = df[df.file_name.isin(clean)]
# print(df.head(15))

boxes = df['box'].apply(lambda x: str(x.replace(' ',''))[1:-1].split(',')).to_list()
# print(boxes[:10])
# path = 'training_videos/'
# path_test = 'training_practice/'


def videocap(video_name):
    cap = cv2.VideoCapture(video_name)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = 640#int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = 360 #int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    
    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
    
    fc = 0
    ret = True
    
    while (fc < frameCount  and ret):
        ret, frame = cap.read()
        buf[fc] = cv2.resize(frame,dsize=(frameWidth,frameHeight))
        fc += 1
    
    cap.release()
    
    return buf

vid_list = []
labels = []
start = time()
for file in files[0:15]:
    label = file[15:].split('_')[0]
    vid = videocap(file)
    labels.append(label)
    vid_list.append(vid)
    
    # print(vid.shape)
    # print(label)
    
    
# print(time()-start)
    
    
def crop_ROI(frame, box):

  y, x = frame.shape[0:2]
  # print(box)
  # print(x)
  # print(float(box[0])*x)
  start_x = int(float(box[1])*x)
  start_y = int(float(box[0])*y)
  end_x = int(float(box[3])*x)
  end_y = int(float(box[2])*y)
  # frame = frame[start_y:end_y,start_x:end_x]
 
  return frame[start_y:end_y,start_x:end_x]

def load_video(path, ROI, max_frames=0, resize=(224, 224)):
  cap = cv2.VideoCapture(path)
  frames = []
  try:
    while True:
      ret, frame = cap.read()
      if not ret:
        break
      frame = crop_ROI(frame, ROI)
      frame = cv2.resize(frame, resize)
      frame = frame[:, :,[0,1,2]]#[2, 1, 0]]
      frames.append(frame)
      # cv2.imshow('djn',frame)
      # cv2.waitKey(0)
      out.write(frame)
      if len(frames) == max_frames:
        break
  finally:
    cap.release()
  return np.array(frames)# / 255.0 
   
size = (224,224)
 
for file, box in zip(files,boxes):
    fps = 24
    out = cv2.VideoWriter(f'training_arr/{file[15:-4]}.mp4', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    processed_vid = load_video(path = file, ROI = box)
    
    print(processed_vid.shape)
    out.release() 
    
    # fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    # fps = 30
    # video_filename = 'output.avi'
    # out = cv2.VideoWriter(video_filename, fourcc,
    #                       fps, (processed_vid.shape[0], processed_vid.shape[1]))
    #out.write(processed_vid)
    
    
    #np.savez_compressed(f'training_arr/{file[15:-4]}.npz', processed_vid)
    
    
    
    

   
    
