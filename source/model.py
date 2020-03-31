# -*- coding: utf-8 -*-
"""Model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ot3-36M2QRdtBlcITI3reMi3aDhyQ5Dp
"""

#%tensorflow_version 2.0
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import tensorflow as tf

import keras
from keras import backend as K
from keras import layers
from keras.layers import Input, Multiply, Add, Dense, Activation, Maximum, ZeroPadding2D, BatchNormalization,Conv2D, MaxPooling2D,DepthwiseConv2D,SeparableConv2D
from keras.models import Model, load_model
from keras.layers.core import Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model
from keras.initializers import glorot_uniform
from keras.layers.convolutional import *
from keras.models import Model
from keras.applications.xception import Xception
from keras.models import Model
import keras.utils
from sklearn.metrics import confusion_matrix
import itertools
import os
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
import sys
import gc
from IPython.display import SVG
import scipy.misc
print (keras.__version__)
print (tf.__version__)





def spatial_feature_aggregation_block(X, num_channel, base):
    
    X_shortcut = X
    
    X = BatchNormalization(axis=-1, name=base + '/Branch1/bn_1')(X)
    X = Activation('relu', name=base + '/Branch1/relu_1')(X)
    X = Conv2D(num_channel, kernel_size=(3,3), strides=(1,1), padding='same', name=base + '/Branch1/conv_1', kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis=-1, name=base + '/Branch1/bn_2')(X)
    X= Activation('relu', name=base + '/Branch1/relu_2')(X)
    X = DepthwiseConv2D((3,3), strides=(1,1), padding='same', name=base + '/Branch1/depthconv_2', kernel_initializer=glorot_uniform(seed=0))(X)
    
    
    ##### Branch2 Start #####
    X_shortcut = BatchNormalization(axis=-1, name=base + '/Branch2/bn_1')(X_shortcut)
    X_shortcut = Activation('relu', name=base + '/Branch2/relu_1')(X_shortcut)
    X_shortcut = SeparableConv2D(num_channel, (3,3), strides=(1,1), dilation_rate=1, padding='same', name=base + '/Branch2/SepConv_1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    
   ##### Branch2 --- 1 #####
    X_shortcut_1 = X_shortcut
    X = Add(name=base + '/Add_X_shortcut_1')([X, X_shortcut_1])

    ##### Branch2 --- 2 #####
    X_shortcut_2 = BatchNormalization(axis=-1, name=base + '/Branch2_sc2/bn_1')(X_shortcut)
    X_shortcut_2 = Activation('relu', name=base + '/Branch2_sc2/relu_1')(X_shortcut_2)
    X_shortcut_2 = Conv2D(num_channel, kernel_size=(3,3), strides=(1,1),dilation_rate=2, padding='same', name=base + '/Branch2_sc2/conv_1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut_2)
     
    X_shortcut_2 = BatchNormalization(axis=-1, name=base + '/Branch2_sc2/bn_2')(X_shortcut_2)
    X_shortcut_2= Activation('relu', name=base + '/Branch2_sc2/relu_2')(X_shortcut_2)
    X_shortcut_2 = DepthwiseConv2D((3,3), strides=(1,1), padding='same', name=base + '/Branch2_sc2/depthconv_2', kernel_initializer=glorot_uniform(seed=0))(X_shortcut_2)
    
    X = Add(name=base + '/Add_X_shortcut_2')([X, X_shortcut_2])

    ##### Branch2 --- 3 ##### 
    X_shortcut_3 = BatchNormalization(axis=-1, name=base + '/Branch3/bn')(X_shortcut)
    X_shortcut_3 = Activation('relu', name=base + '/Branch3/relu')(X_shortcut_3)
    X_shortcut_3 = SeparableConv2D(num_channel, (3,3), strides=(1,1), dilation_rate=2, padding='same', name=base + '/Branch2/SepConv', kernel_initializer=glorot_uniform(seed=0))(X_shortcut_3)
     
    X = Add(name=base + '/Add_X_shortcut_3')([X, X_shortcut_3])
    
    return X


def spa_pooling(X, num_channel, base):

 ##### Branch1 Start ##### 
  X_shortcut1 = BatchNormalization(axis=-1, name=base + '/Branch1/bn_1')(X)
  X_shortcut1 = Activation('relu', name=base + '/Branch1/relu_1')(X_shortcut1)
  X_shortcut1 = Conv2D(num_channel, kernel_size=(2,2), strides=(2,2), padding='valid', name=base + '/Branch1/conv_1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut1)
  ##### Branch2 Start ##### 
  X_shortcut2 = BatchNormalization(axis=-1, name=base + '/Branch2/bn_1')(X)
  X_shortcut2 = Activation('relu', name=base + '/Branch2/relu_1')(X_shortcut2)
  X_shortcut2 = SeparableConv2D(num_channel, (2,2), strides=(2,2), dilation_rate=1, padding='same', name=base + '/Branch2/SepConv_1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut2)

  X = Add(name=base + '/Add_X_shortcut_1_2')([X_shortcut1, X_shortcut2])
  
  X = BatchNormalization(axis=-1, name=base + '/Branch1_2/bn_2')(X)
  X = Activation('relu', name=base + '/Branch1_2/relu_2')(X)
  X = DepthwiseConv2D((3,3), strides=(1,1), padding='same', name=base + '/Branch1_2/depthconv_2', kernel_initializer=glorot_uniform(seed=0))(X)
    
  X = Add(name=base + '/Add_X_shortcut_1_and_X')([X, X_shortcut1])
  X = Add(name=base + '/Add_X_shortcut_2_and_X')([X, X_shortcut2])

  return X  


def SPANet(img_size=224,channels=3):
 
    input_shape = (img_size,img_size,channels)
    X_input= Input(input_shape)
    X= spatial_feature_aggregation_block(X_input,64,'spa_block1_1')
    X= spatial_feature_aggregation_block(X,64,'spa_block1_2')
    X= spa_pooling(X,128,'spa_pool1')
    
    X= spatial_feature_aggregation_block(X,128,'spa_block2_1')
    X= spatial_feature_aggregation_block(X,128,'spa_block2_2')
    X= spa_pooling(X,256,'spa_pool2')

    X= spatial_feature_aggregation_block(X,256,'spa_block3_1')
    X= spa_pooling(X,512,'spa_pool3')

    X= spatial_feature_aggregation_block(X,512,'spa_block4_1')
    X= spa_pooling(X,512,'spa_pool4')

    X= spatial_feature_aggregation_block(X,512,'spa_block5_1')
    X= spa_pooling(X,1024,'spa_pool5')
    X= GlobalAveragePooling2D()(X)
    X = Dense(128, name='Dense_1')(X)
    #X = Dropout(0.2)(X)
    X = Dense(16, name='Dense_2')(X)
    #X = Dropout(0.2)(X)
    X = Dense(2, name='Dense_3')(X)
    Output = Activation('softmax', name='classifier')(X)
       
   # predictions = Dense(2, activation='softmax')(X)
    model = Model(inputs=X_input, outputs=Output)
    model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model