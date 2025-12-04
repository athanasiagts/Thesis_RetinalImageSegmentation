###################################################
#   - Define the neural network
##################################################
from __future__ import division
import tensorflow.keras
# from tensorflow.keras.models import Model
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model as plot
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import *
from tensorflow.keras import regularizers, losses

import configparser as cfp
import sys
sys.path.insert(0, './lib/')


METRICS = [
      tensorflow.keras.metrics.TruePositives(name='tp'),
      tensorflow.keras.metrics.FalsePositives(name='fp'),
      tensorflow.keras.metrics.TrueNegatives(name='tn'),
      tensorflow.keras.metrics.FalseNegatives(name='fn'), 
      tensorflow.keras.metrics.BinaryAccuracy(name='accuracy'),
      tensorflow.keras.metrics.Precision(name='precision'),
      tensorflow.keras.metrics.Recall(name='recall'),
      tensorflow.keras.metrics.AUC(name='auc'),
      tensorflow.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

def NO_PLAIN_net(input_size):
    inputs = Input(input_size)
    # s = Lambda(lambda x: x / 255)(inputs)
    num_classes = 2

    conv1 = Conv2D(64, 4, activation='relu', strides=1, padding='valid', kernel_initializer='glorot_normal')(inputs)
    conv2 = Conv2D(64, 3, activation='relu', strides=1, padding='same', kernel_initializer='glorot_normal')(conv1)

    conv3 = Conv2D(128, 3, activation='relu', strides=1, padding='same', kernel_initializer='glorot_normal')(conv2)
    conv4 = Conv2D(128, 3, activation='relu', strides=1, padding='same', kernel_initializer='glorot_normal')(conv3)

    flat1 = Flatten()(conv4)
    dense1 = Dense(512, activation='relu', kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
                   bias_initializer='zeros', kernel_regularizer=regularizers.l2(5e-4))(flat1)
    drop1 = Dropout(0.5)(dense1)

    # model.add(Dense(64, kernel_initializer=initializers.random_normal(stddev=0.01)))
    dense2 = Dense(512, activation='relu', kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
                   bias_initializer='zeros', kernel_regularizer=regularizers.l2(5e-4))(drop1)
    drop2 = Dropout(0.5)(dense2)

    dense3 = Dense(num_classes, activation='softmax', kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01, seed=None),
                   bias_initializer='zeros')(drop2)
    # create optimizer parameters etc.

    model = Model(inputs=inputs, outputs=dense3)
    model.compile(optimizer=SGD(lr=1e-3, decay=1e-1, momentum=0.9, nesterov=True), loss='binary_crossentropy', metrics=['accuracy'])
    # or categorical crossentropy?
    return model

##########################################################################################

def PLAIN_net(input_size):  # = (27,27,1) / [ (27,27,3) or (565,565,1) ? ]
    inputs = Input(input_size)
    # s = Lambda(lambda x: x / 255)(inputs)
    num_classes = 2

    # batch0 = BatchNormalization()(inputs)
    conv1 = Conv2D(64, (4,4), activation='relu', strides=1, padding='valid', kernel_initializer=GlorotNormal())(inputs)
    conv2 = Conv2D(64, (3,3), activation='relu', strides=1, padding='same', kernel_initializer=GlorotNormal())(conv1)
    batch1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2), strides=2, padding='valid')(batch1)

    conv3 = Conv2D(128, (3,3), activation='relu', strides=1, padding='same', kernel_initializer=GlorotNormal())(pool1)
    conv4 = Conv2D(128, (3,3), activation='relu', strides=1, padding='same', kernel_initializer=GlorotNormal())(conv3)
    batch2 = BatchNormalization()(conv4)
    pool2 = MaxPooling2D(pool_size=(2,2), strides=2, padding='valid')(batch2)

    flat1 = Flatten()(pool2)  # flatten the input to FC layer
    dense1 = Dense(512, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
                   bias_initializer='zeros', kernel_regularizer=regularizers.l2(5e-4))(flat1)
    batch3 = BatchNormalization()(dense1)
    drop1 = Dropout(0.5)(batch3)

    dense2 = Dense(512, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
                   bias_initializer='zeros', kernel_regularizer=regularizers.l2(5e-4))(drop1)
    batch4 = BatchNormalization()(dense2)
    drop2 = Dropout(0.5)(batch4)

    dense3 = Dense(2, activation='softmax', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
                   bias_initializer='zeros')(drop2)

    model = Model(inputs=inputs, outputs=dense3)
    model.compile(optimizer=SGD(lr=1e-3, momentum=0.9, nesterov=True), loss='binary_crossentropy', metrics=METRICS)        #['accuracy']
    # or categorical crossentropy? for more than 2 classes
    return model
