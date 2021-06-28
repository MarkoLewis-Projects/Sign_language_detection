# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 19:18:14 2021

@author: Lewis
"""

import os
import cv2
import numpy as np
import mediapipe as mp
import time
import uuid


vid_capture = cv2.VideoCapture(0)
mphands = mp.solutions.hands
hands = mphands.Hands(static_image_mode=False,max_num_hands=1,min_detection_confidence=0.5)
draw = mp.solutions.drawing_utils
IMAGE_PATH = 'H:\Digital_Futures_Projects\CapstoneProject\lewis_letters\Y'


while True:
    
    ret, frame = vid_capture.read()
    #box_creator((frame))
    h, w, c = frame.shape
    img_hstart, img_wstart, colorsstart = frame.shape
    img_height, img_width, colors = frame.shape
    rgbframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # ---------------------------
    frame = cv2.flip(frame,1) 
    result = hands.process(frame)
   
    
    frame.flags.writeable = True

    
    if result.multi_hand_landmarks:

            
        for handLMs in result.multi_hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * (w)), int(lm.y * (h))
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
                    
            #print(type(x_min))
            
            x_range = (x_max*1.1)-(x_min*0.8)
            
            coords = []
            coords.append(x_min*0.85)
            coords.append(y_min*0.85)
            coords.append((x_max*1.1)-(x_min*0.85))
            coords.append((y_max*1.1)-(y_min*0.85))
            cv2.rectangle(frame, (int(x_min*0.85), int(y_min*0.85)), (int(x_max*1.1), int(y_max*1.1)), (0, 255, 0), 2)
           
            
            x,y,w,h = coords
            hand_img = frame[int(y) : int(y+h), int(x) : int(x+w)]
           
            
    cv2.imshow('MediaPipe hand Detection', frame)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(2) & 0xFF == ord('a'):
        img_resize = cv2.resize(hand_img,(28,28))
        img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
        img_name = IMAGE_PATH+('\Y{}.bmp'.format(str(uuid.uuid4())))
        cv2.imwrite(img_name, img_resize)
        print('img saved as {}'.format(img_name))
    
vid_capture.release()
cv2.destroyAllWindows()