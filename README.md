# Sign_language_detection
Lewis + Marko projects

Detection algorithm for sign langauge

## Intro:
Deafness and sign language are the largest and hardest to overcome language barriers in the modern world. In this case it isn't feasible to "learn" or even attempt to bridge this gap through self-study. Here we aim to aid in crossing this gap through the creation of a lightweight and truely robust ASL(American Sign-Language) detection model that could take a live feed from a user's camera and translate word and alphabet level sign in real-time to english. The realisation of such a project would surely aid in day to day life in order to connect more people together through seemless conversation.

The repository itself contains 3 main folders:

#### Model 
- This folder contains all the model files and preprocessing done for the project

#### VideoFeed 
- This contains the final script that takes a live user's feed and translates to english in real-time as well as the script for us to take more photos

#### Video_scraping_scripts 
- This contains the python scripts used to obtain the video dataset from youtube

The project can be described as a two part system where we looked into creating two models that could work with one another to robustly recognise and translate ASL through both alphabet level and word level sign language.

## Project Overview

### ALPHABET LEVEL:

This was the easier of the two tasks due to the dimensionality, as classification with no regard for the temporal relations is a very light computation. Here we utilised both transfer learning and full learning utilising custom CNN architecture.  
The dataset used for this was comprised of two main sections. Firstly the image dataset found on kaggle.com (https://www.kaggle.com/grassknoted/asl-alphabet), and secondly we generated our own sign language images utilising the openCV module.  
The largest issue with this dataset was its lack of background and signer diversity dispite our efforts to diversify the data. This lead to suboptimal performance in real world scenarios making transfer learning the best solution due to models being previously trained on ImageNet and having an easier time recognising truely relevant features. 

The pretrained models considered were: 

#### MobileNetV2 
- Chosen for its light-weight architecture and mobile application compatibility.

#### ResNet34 
- Chosen for the amazing performance of residual networks in recent years and determined to have a worthy trade of processing time for prediction accuracy. 

#### DenseNet201 
- As a close relative of ResNet with fewer parameters DenseNet was another obvious consideration.

Whilst all models performed compitently when compared to the validation data, we found that ResNet34 was superior in real world translation scenarios. Most modelling attempts were an initial pass and further finetuning is available, yet due to the nature of transfer learning we believe there will be no significant changes in ranking of these models.

Realtime example:

![alt text](https://github.com/MarkoLewis-Projects/Sign_language_detection/blob/main/hand_detection_clipped.gif "Resnet34 detection")

### WORD LEVEL:

Due to the computational intensity of high accuracy video recognition we have opted to complete this section at a later date, however preliminary work can be seen through video aquisition/preprocessing in conjunction with preliminary transfer learning attempts with both i3D from deepmind trained with the kinetics 400 dataset (https://arxiv.org/abs/1705.07750), as well as the C3D model which was trained on the Sports1M dataset (https://arxiv.org/abs/1412.0767).

## Presentation

You can view the presentation of our findings here (reccomend turning down for 2nd speaker): 

[![Watch the video](http://img.youtube.com/vi/ooL-wb60CFE/hqdefault.jpg)](https://youtu.be/ooL-wb60CFE)
