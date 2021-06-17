from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Lambda
from tensorflow.keras.layers import Conv3D, MaxPooling3D, ZeroPadding3D
#from tensorflow.keras.optimizers import SGD
import tensorflow.keras.backend as K

#https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.pdf

class model:
    
    def __init__(self, weights, trainable = True, freeze_layer = 0):
    
        self.weights = weights
        self.trainable = True
        
    def make_model(self, kernel, input_shape, freeze_layer):
        
        model = Sequential()
        
        #layer 1
        model.add(Conv3D(64, kernel, 
                    activation = 'relu', 
                    input_shape = input_shape, 
                    strides = (1, 1, 1), 
                    name = 'Conv1', 
                    padding = 'same'))
        model.add(MaxPooling3D(pool_size = (1, 2, 2), 
                        strides = (1, 2, 2),
                        name = 'Pool1',
                        padding = 'valid'))
        
        #layer 2
        model.add(Conv3D(128, kernel, 
                    activation = 'relu',
                    strides = (1, 1, 1), 
                    name = 'Conv2', 
                    padding = 'same'))
        model.add(MaxPooling3D(pool_size = (2, 2, 2), 
                        strides = (2, 2, 2),
                        name = 'Pool2',
                        padding = 'valid'))
                        
        #layer 3
        model.add(Conv3D(256, kernel, 
                    activation = 'relu', 
                    strides = (1, 1, 1), 
                    name = 'Conv3a', 
                    padding = 'same'))
        model.add(Conv3D(256, kernel, 
                    activation = 'relu',  
                    strides = (1, 1, 1), 
                    name = 'Conv3b', 
                    padding = 'same'))
        model.add(MaxPooling3D(pool_size = (2, 2, 2), 
                        strides = (2, 2, 2),
                        name = 'Pool3',
                        padding = 'valid'))                 
                        
        #layer 4
        model.add(Conv3D(512, kernel, 
                    activation = 'relu', 
                    strides = (1, 1, 1), 
                    name = 'Conv4a', 
                    padding = 'same'))
        model.add(Conv3D(512, kernel, 
                    activation = 'relu',  
                    strides = (1, 1, 1), 
                    name = 'Conv4b', 
                    padding = 'same'))
        model.add(MaxPooling3D(pool_size = (2, 2, 2), 
                        strides = (2, 2, 2),
                        name = 'Pool4',
                        padding = 'valid'))

        #layer 5
        model.add(Conv3D(512, kernel, 
                    activation = 'relu', 
                    strides = (1, 1, 1), 
                    name = 'Conv5a', 
                    padding = 'same'))
        model.add(Conv3D(512, kernel, 
                    activation = 'relu',  
                    strides = (1, 1, 1), 
                    name = 'Conv5b', 
                    padding = 'same'))
        model.add(ZeroPadding3D(padding = (0, 1, 1)))
        model.add(MaxPooling3D(pool_size = (2, 2, 2), 
                        strides = (2, 2, 2),
                        name = 'Pool5',
                        padding = 'valid'))
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
            layer.trainable = self.trainable
        
        return model
        
    def retrainable_model(self,kernel,input_shape):
        
         model = self.make_model(kernel = kernel, input_shape = input_shape, freeze_layer = 3)
         
         #Devide by magnitude squared
         model.add(Lambda(lambda  x: K.l2_normalize(x, axis=1)))
         
         #could add pca
        
         return model   
    