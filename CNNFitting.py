# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 10:36:29 2017

@author: Chuhan Wu
"""

import numpy as np  
np.random.seed(1337)  # for reproducibility  
from keras.models import Sequential  
from keras.layers import Dense, Dropout, Activation, Flatten  
from keras.layers import  MaxPooling2D  
from keras.utils import np_utils   
from keras.layers import Conv2D
 
###########################################################################################################  
def classnumber(document):
    Doctemp=document.flatten()
    temp=[]
    x=0
    while x<len(Doctemp):
        if Doctemp[x] not in temp:
            temp.append(temp)
        x=x+1
    length=len(temp)
    return length

###########################################################################################################  
def ConvLayer(Model,ConvLay,ConvAct,Filter,kernel_size):
    x=2
    while x<=ConvLay:
            Model.add(Conv2D(Filter, (kernel_size[0], kernel_size[1])))
            Model.add(Activation(ConvAct))
            x=x+1
                
###########################################################################################################          
def DensLayer(Model,DensLay,DensUnit,DensAct,Drop_perc):
    x=1
    while x<=DensLay:
        Model.add(Dense(DensUnit))                     
        Model.add(Activation(DensAct))
        Model.add(Dropout(Drop_perc))
        x=x+1

###########################################################################################################  
def ModelBuild(Model,input_shape,ConvLay,ConvAct,Filter,kernel_size,pool_size,DensLay,DensUnit,DensAct,Drop_perc,OutAct,num1):
    Model.add(Conv2D(Filter, (kernel_size[0], kernel_size[1]),  
                        padding='same',  
                        input_shape=input_shape))  
    Model.add(Activation(ConvAct))
    ConvLayer(Model,ConvLay,ConvAct,Filter,kernel_size)             #Convolution-layer
    Model.add(MaxPooling2D(pool_size=pool_size))                    #MaxPooling-layer
    Model.add(Dropout(Drop_perc))                                   #Dropout-layer 
    Model.add(Flatten())                                            #flatten-layer
    DensLayer(Model,DensLay,DensUnit,DensAct,Drop_perc)             #Fully-collected-layer
    Model.add(Dense(num1))   
    Model.add(Activation(OutAct))
    
###########################################################################################################
###########################################################################################################
def CNNTraining(docx_train,docy_train,docx_CV,docy_CV,docx_test,docy_test,nb_classes = 2,epoch=10,batch_size=128,ConvLay=3,ConvAct='relu',Filter=32,kernel_size=(3,3),pool_size=(2,2),Drop_perc=0.45,DensLay=2,DensUnit=128,DensAct='relu',OutAct='softmax'):      
    
    img_rows=docx_train.shape[1] 
    img_cols=docx_train.shape[2]
    X_train = docx_train.reshape(docx_train.shape[0], img_rows, img_cols, 1)  
    X_test = docx_test.reshape(docx_test.shape[0], img_rows, img_cols, 1)  
    X_CV = docx_CV.reshape(docx_CV.shape[0], img_rows, img_cols, 1)  
    in_shape= (img_rows, img_cols, 1)    
###########################################################################################################
    Y_train = np_utils.to_categorical(docy_train, nb_classes)
    Y_CV = np_utils.to_categorical(docy_CV, nb_classes)    
    Y_test = np_utils.to_categorical(docy_test, nb_classes)  
  
########################################################################################################### 
    model = Sequential()  
    ModelBuild(model,in_shape,ConvLay,ConvAct,Filter,kernel_size,pool_size,DensLay,DensUnit,DensAct,Drop_perc,OutAct,nb_classes)

###########################################################################################################   
    model.compile(loss='categorical_crossentropy',  
                  optimizer='adadelta',  
                  metrics=['accuracy'])  
    
########################################################################################################### 
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epoch,  
              verbose=1)#, validation_data=(X_CV, Y_CV))  

########################################################################################################### 
    print('\n################    The Detail of the Training     ###################')
    print(model.summary())
    score = model.evaluate(X_test, Y_test, verbose=0)  
    print('\n######################################################################\n')
    print('\nThe final Lost value of learning is :        ',score[0])
    print('The final accuracy of learning is   :        ',score[1])
    return score
    
########################################################################################################### 
A=CNNTraining(X_train,Y_train,X_CrossValidation,Y_CrossValidation,X_CrossValidation,Y_CrossValidation,epoch=12,batch_size=4,ConvLay=3,ConvAct='relu',Filter=32,kernel_size=(3,3),pool_size=(2,2),Drop_perc=0.25,DensLay=2,DensUnit=128,DensAct='sigmoid',OutAct='softmax')