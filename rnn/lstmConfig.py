seed_value= 0
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)
# 4. Set `tensorflow` pseudo-random generator at a fixed value
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#tf.random.set_seed(seed_value) # tensorflow 2.x
tf.set_random_seed(seed_value) # tensorflow 1.x
# 5. Configure a new global `tensorflow` session
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Masking
from keras.layers import Dropout, Flatten
from keras.utils import plot_model
from keras.optimizers import SGD, RMSprop, Adam
from keras.models import load_model
from rnn.KerasBatchGenerator import KerasBatchGenerator
from keras.layers import BatchNormalization
from keras_layer_normalization import LayerNormalization
from matplotlib import pyplot as plt
from utils.utils import reset_keras
from utils.algorithm import algorithm
from rnn.CustomEarlyStopping import CustomEarlyStopping
from keras.callbacks import ModelCheckpoint
import numpy as np
import math
import numpy.ma as ma
import pandas as pd
import pdb
from sklearn.preprocessing import StandardScaler
from keras.regularizers import L1L2
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from keras.backend import binary_crossentropy
from keras.callbacks import TensorBoard


class lstmConfig():

    model_build = False
    scaler = None
    path = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/src/model/"

    def __init__(self, optimizer="RMSprop"):
        super().__init__() 
        self.optimizer = optimizer
        # Initialising the RNN
        self.model = None
        self.history = None
        self.scaler = None
        self.indices = None
        self.history_index = None
        self.mean_missing = None


    def _build_network(self, X_train, lstm_layers, keep_prob, kernel_regularizer, lstm_activation, dense_layers, dense_activation, learning_rate):
        self.model = Sequential()
        self.model_build = True
        length_lstm = len(lstm_layers)
        if length_lstm==1:
            self._lstm_cell(lstm_layers[0], lstm_activation, keep_prob[0], kernel_regularizer, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))
        else:
            for i in range(0, length_lstm):
                if i==0:
                    self._lstm_cell(lstm_layers[0], lstm_activation, keep_prob[0], kernel_regularizer, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))
                else:
                    self._lstm_cell(lstm_layers[i], lstm_activation, keep_prob[i], kernel_regularizer, return_sequences=True)
        length_dense = len(dense_layers)
        for i in range(0, length_dense):
            if i==0 and length_lstm==0:
                self.model.add(Flatten(input_shape=(X_train.shape[1], X_train.shape[2]))) 
                self._dense_cell(dense_layers[i], dense_activation, keep_prob[length_lstm+i], kernel_regularizer)#, input_shape=(X_train.shape[1], X_train.shape[2]))
            else:
                self._dense_cell(dense_layers[i], dense_activation, keep_prob[length_lstm+i], kernel_regularizer)
        # Adding the output layer    
        self.model.add(BatchNormalization())       
        if X_train.ndim == 3 and len(lstm_layers)!=0:
            self.model.add(Flatten())
        self.model.add(Dense(units=1, activation='sigmoid'))
        # Compiling the RNN
        if self.optimizer=='SGD':
            opt = SGD  (lr=learning_rate)
        elif self.optimizer=='RMSprop':
            opt = RMSprop(lr=learning_rate, clipnorm=1.0)
        else:
            opt = Adam(lr=learning_rate)
        #print(self.model.summary())
        #self.model.compile(optimizer=opt, loss=calc_custom_loss2, metrics=[custom_accuracy])
        #self.model.compile(optimizer=opt, loss=binary_crossentropy, metrics=['accuracy'])
        self.model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    def _lstm_cell(self, units, lstm_activation, keep_prob, kernel_regularizer, return_sequences=True, input_shape=None):
        if input_shape is None:
            self.model.add(LSTM(units=units, activation=lstm_activation, return_sequences=return_sequences, 
                                kernel_regularizer=L1L2(l1=kernel_regularizer[0], l2=kernel_regularizer[1]), unroll=True))
        else:
            self.model.add(LSTM(units=units, activation=lstm_activation, return_sequences=return_sequences, 
                                kernel_regularizer=L1L2(l1=kernel_regularizer[0], l2=kernel_regularizer[1]), unroll=True, input_shape=input_shape))
        #self.model.add(LayerNormalization())
        self.model.add(Dropout(keep_prob))

    def _dense_cell(self, units, dense_activation, keep_prob, kernel_regularizer, input_shape=None):
        if input_shape is None:
            self.model.add(Dense(units=units, activation=dense_activation, kernel_regularizer = L1L2(l1=kernel_regularizer[0], l2=kernel_regularizer[1])))
        else:
            self.model.add(Dense(units=units, activation=dense_activation, kernel_regularizer = L1L2(l1=kernel_regularizer[0], l2=kernel_regularizer[1]), input_shape=input_shape))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(keep_prob))

    def fit(self, X_train, Y_train, X_val=None, Y_val=None, X_test=None, Y_test=None, nb_lags=0, batch_size=32, lstm_layers=[64], lstm_activation="relu", dense_layers=None, 
            dense_activation='relu', keep_prob=[0, 0], kernel_regularizer=[0, 0], learning_rate=0.01, epochs=100):
        try:
            if dense_layers is None:
                dense_layers = [int(lstm_layers[1]/2)]
            self.init_scale(X_train)
            #X_train_prime = X_train
            X_train_prime = self.scale(X_train)
            X_train_prime = self.reshape(X_train_prime, nb_lags)
            self._build_network(X_train_prime, lstm_layers, keep_prob, kernel_regularizer, lstm_activation, dense_layers, dense_activation, learning_rate)
            
            if X_val is not None and Y_val is not None:
                X_val_prime = np.concatenate((X_train[X_train.shape[0]-nb_lags:X_train.shape[0]], X_val), axis=0)
                X_val_prime = self.scale(X_val_prime)
                if self.history_index is not None and self.indices is not None:
                    X_val_prime = self.reshape(X_val_prime, nb_lags, val=True)
                else:
                    X_val_prime = self.reshape(X_val_prime, nb_lags)
                X_val_prime = X_val_prime[nb_lags:]
                Y_train, Y_val = self.prepare_targets(Y_train, Y_val)
                if self.optimizer=='SGD':
                    self.history = self.model.fit(X_train_prime, Y_train, epochs=epochs, shuffle=False, verbose=0, validation_data=(X_val_prime, Y_val))
                else:
                    tb = TensorBoard(histogram_freq=1, write_grads=True)
                    self.history = self.model.fit(X_train_prime, Y_train, epochs=epochs, batch_size=batch_size, shuffle=False, verbose=0, 
                                                      validation_data=(X_val_prime, Y_val), callbacks=[tb])
            else:
                Y_train = self.prepare_targets(Y_train)
                if self.optimizer=='SGD':
                    self.history = self.model.fit(X_train_prime, Y_train, epochs=epochs, shuffle=False, verbose=0)
                else:   
                    self.history = self.model.fit(X_train_prime, Y_train, epochs=epochs, batch_size=batch_size, shuffle=False, verbose=0)
            return self.history, None
        except: 
            return self.history, "issue"

    def predict(self, X_test):
        y_hat = self.model.predict_classes(X_test)
        return y_hat

    def evaluate(self, X_test, Y_test):
        y_hat = self.predict(X_test)
        if y_hat is None:
            score = 0
        else:
            score = accuracy_score(Y_test, y_hat)
        return score
     
    def init_scale(self, X):
        self.scaler = StandardScaler()
        X_train = X    
        if len(X_train.shape) == 1:
            X_train = np.reshape(X_train, (X_train.shape[0],1))
        self.scaler.fit(X_train)

    def scale(self, X):
        X_train = self.scaler.transform(X)
        return X_train

    def reshape(self, X, nb_lags, val=False):
        length = len(X)
        if len(X.shape) == 1:
             X = np.reshape(X, (X.shape[0],1))
        X_train = np.empty((length, nb_lags+1, len(X[0])))
        X_train[:] = np.nan
        for i in range(0, X.shape[0]):
            for j in range(0, X.shape[1]):
                X_train[i,nb_lags,j] = X[i,j]
        if self.indices is not None and self.history_index is not None:
            for i in range(0, length):
                for j in range(1, min(i+1, nb_lags+1)):
                    date = self.indices[i]
                    if self._check_index(date, j):
                        X_train[i,nb_lags-j,:] = X[i-j,:]
                    else:
                        X_train[i,nb_lags-j,:] = np.nan
        else:
            for i in range(0, length):
                for j in range(1, min(i+1, nb_lags+1)):
                    X_train[i,nb_lags-j,:] = X[i-j,:]
        if not isinstance(X_train, np.ndarray):
            X_train = np.array(X_train)
        if val is False:
            X_train = np.where(np.isnan(X_train), ma.array(X_train, mask=np.isnan(X_train)).mean(axis=0), X_train)
            self.mean_missing = ma.array(X_train, mask=np.isnan(X_train)).mean(axis=0)
        else:
             X_train = np.where(np.isnan(X_train), self.mean_missing, X_train)
        return X_train

    def _check_index(self, date, lag):
        index = self.history_index.get_loc(date) - lag
        if index < 0:
            return False
        time = self.history_index[index]
        if time in self.indices:
            return True
        else:
            return False

    def set_indices(self, indices, history):
        self.indices = indices
        self.history_index = history
    # prepare target
    def prepare_targets(self, y_train, y_test=None):
        if y_train.shape[1]==1:
            le = LabelEncoder()
            le.fit(y_train)
            y_train_enc = le.transform(y_train)
            if y_test is not None:
                y_test_enc = le.transform(y_test)
                return y_train_enc, y_test_enc
            else:
                return y_train_enc
        else:
            le = LabelEncoder()
            le.fit(y_train[:,0])
            y_train_enc = np.vstack((le.transform(y_train[:,0]),  y_train[:,1], y_train[:,2])).transpose()
            if y_test is not None:
                y_test_enc = np.vstack((le.transform(y_test[:,0]),  y_test[:,1], y_test[:,2])).transpose()
                return y_train_enc, y_test_enc
            else:
                return y_train_enc

    def plot_graph(self, file):
        plot_model(model, to_file = file + '.png')

    def plot_history(self):
        # Plot training & validation accuracy values
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        # Plot training & validation loss values
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    def get_model(self):
        if self.model_build is False:
            print("Model has not been build yet, please call function train")
            return
        else:
            return self.model

    def get_optimizer(self):
        return self.optimizer

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def get_history(self):
        return self.history

    def get_history_index(self):
        return self.history_index

    def get_indices(self):
        return self.indices

    def get_mean_missing(self):
        return self.mean_missing
