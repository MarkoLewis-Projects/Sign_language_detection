# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 10:53:32 2021

@author: Lewis Jones + Marko Pancic
"""
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from statistics import mode
import mediapipe as mp
import imutils
import time
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transform
import torchvision.models as models



imsize = 200
loader = transform.Compose([transform.Resize(imsize), transform.ToTensor()])

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def image_loader(img):
    """load image, returns cuda tensor"""
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img)
    image = loader(image).float()
    image = image.unsqueeze(0)
    return image

#model classes
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))     
        
class Resnet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = models.resnet34(pretrained=True)
        features = self.network.fc.in_features
        self.network.fc = nn.Linear(features, 29)
    
    def forward(self, xb):
        return self.network(xb)
    
    def freeze(self):
        for param in self.network.parameters():
            param.require_grad = False
        for param in self.network.fc.parameters():
            param.require_grad = True
    
    def unfreeze(self):
        for param in self.network.parameters():
            param.require_grad = True

letter_output = []

device = torch.device('cpu')
model = Resnet()       
model.load_state_dict(torch.load('asl-model_GC.pth', map_location=device))
model.eval()
vid_capture = cv2.VideoCapture(0)
counter = 0
# Hands
mphands = mp.solutions.hands
hands = mphands.Hands(static_image_mode=False,max_num_hands=1,min_detection_confidence=0.5)
draw = mp.solutions.drawing_utils

img_height = 480
img_width = 640



def sign_preds(img,boxes):
    
    
    x,y,w,h = boxes
    hand_img = img[int(y) : int(y+h), int(x) : int(x+w)]

    try:
        cv2.imshow('test',hand_img)
    except:
        pass
  
    class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G',
                   'H', 'I','J','K', 'L', 'M', 'N',
                   'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                   'V', 'W', 'X', 'Y','Z','del','nothing','. ']
    
    
    image = image_loader(img)
    x = model(image)
    confidence = torch.nn.functional.softmax(x) 
    
    x = torch.argmax(x)
   
    confidence = round(float(confidence[0, x])*100,0)
    
    return class_names[x], confidence
          
     
start = time.time()

while True:
    
    ret, frame = vid_capture.read()
    h, w, c = frame.shape
    img_hstart, img_wstart, colorsstart = frame.shape
    img_height, img_width, colors = frame.shape
    #rgbframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                    
            
            x_range = (x_max*1.2)-(x_min*0.8)
            
            coords = []
            coords.append(x_min*0.8)
            coords.append(y_min*0.8)
            coords.append((x_max*1.2)-(x_min*0.8))
            coords.append((y_max*1.2)-(y_min*0.8))
            try:
                cv2.rectangle(frame, (int(x_min*0.8), int(y_min*0.8)), (int(x_max*1.2), int(y_max*1.2)), (0, 0, 255), 2)
            except:
                continue
            
            x, percent = sign_preds(frame,coords)
            
            print(x)
            
    try:
        cv2.rectangle(frame, (int(x_min*0.8), int(y_min*0.8)), (int(x_max*1.2), int(y_max*1.2)), (0, 0, 255), 2)
    except:
        pass
            
            
            
    if cv2.waitKey(10) & 0xFF == ord('l'):
        letter_output.append(x)
    elif cv2.waitKey(11) & 0xFF == ord('k'):
        letter_output = []    
    
    try:
        if result.multi_hand_landmarks:
            cv2.putText(frame,f'{x}:   {percent}%',(int(x_min*0.8),int(y_min*0.8)),fontFace=cv2.FONT_HERSHEY_SIMPLEX,color=(0,255,0),fontScale=1.2,thickness=2)
            
        words = "".join(letter_output)
        cv2.putText(frame,words,(20,475),fontFace=cv2.FONT_HERSHEY_SIMPLEX,color=(0,255,0),fontScale=2,thickness=3)
    except:
        pass        
            
    cv2.imshow('MediaPipe hand Detection', frame)
    
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
vid_capture.release()
cv2.destroyAllWindows()














