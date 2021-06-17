from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Conv3D, MaxPooling3D, ZeroPadding3D
from keras.optimizers import SGD
import keras.backend as K

#https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.pdf

class model:
    
    self.weights = 0
    self.trainable = False
    self.freeze_layer = 0
    
    
	def __init__(self, weights, trainable = True, freeze_layer = 0, l2_norm = False):
    
        self.weights = None
        self.trainable = True
        self.freeze_layer = 0
        
    def make_model(self, Conv_kernel, input_shape):
        
        model = Sequential()
        
        #layer 1
        model.add(Conv3D(64, kernel, 
                    activation = 'relu', 
                    input_shape = input_shape, 
                    strides = (1, 1, 1), 
                    name = 'Conv3D 1a', 
                    padding = 'same'))
        model.add(MaxPooling3D(pool_size(1, 2, 2), 
                        strides(1, 1, 2),
                        name = 'MaxPool 1'
                        padding = 'valid') 
        
        #layer 2
        model.add(Conv3D(128, kernel, 
                    activation = 'relu',
                    strides = (1, 1, 1), 
                    name = 'Conv3D 2a', 
                    padding = 'same'))
        model.add(MaxPooling3D(pool_size(1, 2, 2), 
                        strides(1, 2, 2),
                        name = 'MaxPool 2'
                        padding = 'valid') 
                        
        #layer 3
        model.add(Conv3D(256, kernel, 
                    activation = 'relu', 
                    strides = (1, 1, 1), 
                    name = 'Conv3D 3a', 
                    padding = 'same'))
        model.add(Conv3D(256, kernel, 
                    activation = 'relu',  
                    strides = (1, 1, 1), 
                    name = 'Conv3D 3b', 
                    padding = 'same'))
        model.add(MaxPooling3D(pool_size(2, 2, 2), 
                        strides(2, 2, 2),
                        name = 'MaxPool 3'
                        padding = 'valid')                 
                        
        #layer 4
        model.add(Conv3D(512, kernel, 
                    activation = 'relu', 
                    strides = (1, 1, 1), 
                    name = 'Conv3D 4a', 
                    padding = 'same'))
        model.add(Conv3D(512, kernel, 
                    activation = 'relu',  
                    strides = (1, 1, 1), 
                    name = 'Conv3D 4b', 
                    padding = 'same'))
        model.add(MaxPooling3D(pool_size(2, 2, 2), 
                        strides(2, 2, 2),
                        name = 'MaxPool 4'
                        padding = 'valid')

        #layer 5
        model.add(Conv3D(512, kernel, 
                    activation = 'relu', 
                    strides = (1, 1, 1), 
                    name = 'Conv3D 5a', 
                    padding = 'same'))
        model.add(Conv3D(512, kernel, 
                    activation = 'relu',  
                    strides = (1, 1, 1), 
                    name = 'Conv3D 5b', 
                    padding = 'same'))
        model.add(ZeroPadding3D(padding = (0, 1, 1)))
        model.add(MaxPooling3D(pool_size(2, 2, 2), 
                        strides(2, 2, 2),
                        name = 'MaxPool 5'
                        padding = 'valid')
        model.add(Flatten())
        
        #Fully Connected Layers
        model.add(Dense(4096, activation = 'relu', name = 'fc6'))
        model.add(Dropout(0.5))
        
        model.add(Dense(4096, activation = 'relu', name = 'fc7'))
        model.add(Dropout(0.5))
        
        #final layer
        model.add(Dense(487, activation='softmax', name='fc8'))
        
        

        if self.weights:
            model.load_weights(self.weights)
            
        if freeze_layer > 0:
            for i in range(freeze_layer):
            model.pop()
            
        for layer in model.layers:
            layer.trainable = trainable
        
        return model
        
    def retrainable_model(self, Conv_kernel, input_shape)
        
        model = make_model(weights = self.weights, freeze_layer = 3)
        
        #Devide by magnitude squared
        model.add(Lambda(lambda  x: K.l2_normalize(x, axis=1)))
        
        #could add pca
        
        return model   
    