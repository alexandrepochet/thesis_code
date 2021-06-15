import numpy as np
import pandas as pd
import copy
import pdb
import keras
from keras import backend as K
from sklearn.metrics import accuracy_score
from keras.layers.core import Reshape
from keras.layers import concatenate
from keras.backend import binary_crossentropy


class callback():

    def __init__(self, ratio=1.2, max_accuracy_training=0.65,
                 patience=5):
        self.ratio = ratio
        self.patience = patience
        self.wait = 0
        self.stopped_epoch = 0
        self.stop_training = False
        self.monitor_op = np.less
        self.max_accuracy_training = max_accuracy_training

    def EarlyStopping(self, training, validation):
        length = len(training)
        self.stopped_epoch = length
        for i in range(0, length):
            if self.stop_training is False:
                current_train = training[i]
                current_val = validation[i]
                if (self.monitor_op(np.divide(current_train,current_val),self.ratio) and self.monitor_op(current_train,self.max_accuracy_training)) or (current_val<0.5 and self.monitor_op(current_train,self.max_accuracy_training)):
                    self.wait = 0
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        self.stopped_epoch = i + 1
                        self.stop_training = True
            else:
                break
        return self.stopped_epoch

def calc_custom_loss(y_true, y_pred):
    returns = y_true[:,1]
    returns = Reshape((1,))(returns)
    tcosts = y_true[:,2]
    tcosts = Reshape((1,))(tcosts)
    loss =  -K.mean(returns*K.tanh(10*(y_pred - 0.5)) - tcosts*K.abs(K.tanh(10*(tf_diff_axis_0(y_pred)))), axis=-1)
    return loss

def tf_diff_axis_0(a):
    const = Reshape((1,))(K.constant([1]))
    return concatenate([a[1:]-a[:-1], const], axis=0)

def custom_accuracy(y_true, y_pred):
    y = y_true[:,0]  
    y = Reshape((1,))(y)
    accuracy = K.mean(K.equal(y, K.round(y_pred)))#
    return accuracy