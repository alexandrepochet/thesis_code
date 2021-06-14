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
import tensorflow as tf
#tf.random.set_seed(seed_value) # tensorflow 2.x
tf.set_random_seed(seed_value) # tensorflow 1.x
# 5. Configure a new global `tensorflow` session
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout, Flatten
from keras.utils import plot_model
from utils.parallel import parallel
from keras.optimizers import SGD, RMSprop, Adam
from rnn.KerasBatchGenerator import KerasBatchGenerator
from matplotlib import pyplot as plt
import gensim
import numpy as np
import pandas as pd
import pdb
from sklearn.preprocessing import StandardScaler
from keras.regularizers import L1L2
from sklearn.model_selection import KFold, TimeSeriesSplit
# Seed value (can actually be different for each attribution step)


class lstmConfig(parallel):

    model_build = False
    scaler = None
    path = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/src/model/"

    def __init__(self, batch_size=32, lstm_layers=[8], lstm_activation="relu", dense_layers=[4], 
                 dense_activation='relu', keep_prob=[0, 0], kernel_regularizer=[0, 0], learning_rate=0.01, optimizer="Adam"):
        self.lstm_layers = lstm_layers
        self.dense_layers = dense_layers
        self.num_layers = len(lstm_layers) + len(dense_layers)
        self.keep_prob = keep_prob
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.lstm_activation = lstm_activation
        self.dense_activation = dense_activation
        self.kernel_regularizer = kernel_regularizer
        # Initialising the RNN
        self.model = Sequential()
        self.history = None

    def build_network(self, X_train):
        self.model_build = True
        length_lstm = len(self.lstm_layers)
        if length_lstm==1:
            self._lstm_cell(self.lstm_layers[0], self.keep_prob[0], self.kernel_regularizer, input_shape=(X_train.shape[1], X_train.shape[2]))
        else:
            for i in range(0, length_lstm):
                if i==0:
                    self._lstm_cell(self.lstm_layers[0], self.keep_prob[0], self.kernel_regularizer, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))
                #elif i==self.num_layers-1:
                #    self._lstm_cell(self.lstm_layers[i], self.keep_prob[i], self.kernel_regularizer)
                else:
                    self._lstm_cell(self.lstm_layers[i], self.keep_prob[i], self.kernel_regularizer, return_sequences=True)
        length_dense = len(self.dense_layers)
        for i in range(0, length_dense):
            if i==0 and length_lstm==0:
                self._dense_cell(self.dense_layers[i], self.keep_prob[length_lstm+i], self.kernel_regularizer, input_shape=(X_train.shape[1], X_train.shape[2]))
            else:
                self._dense_cell(self.dense_layers[i], self.keep_prob[length_lstm+i], self.kernel_regularizer)
        # Adding the output layer            
        if X_train.ndim == 3:
            self.model.add(Flatten())
        self.model.add(Dense(units=1, activation='sigmoid'))
        # Compiling the RNN
        if self.optimizer=='SGD':
            opt = SGD  (lr=self.learning_rate)
        elif self.optimizer=='RMSprop':
            opt = RMSprop(lr=self.learning_rate)
        else:
            opt = Adam(lr=self.learning_rate)
        self.model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    def _lstm_cell(self, units, keep_prob, kernel_regularizer, return_sequences=True, input_shape=None):
        if input_shape is None:
            self.model.add(LSTM(units=units, activation=self.lstm_activation, return_sequences=return_sequences, kernel_regularizer=L1L2(l1=kernel_regularizer[0], l2=kernel_regularizer[1])))
        else:
            self.model.add(LSTM(units=units, activation=self.lstm_activation, return_sequences=return_sequences, kernel_regularizer=L1L2(l1=kernel_regularizer[0], l2=kernel_regularizer[1]), input_shape=input_shape))
        self.model.add(Dropout(keep_prob))

    def _dense_cell(self, units, keep_prob, kernel_regularizer, input_shape=None):
        if input_shape is None:
            self.model.add(Dense(units=units, activation=self.dense_activation, kernel_regularizer = L1L2(l1=kernel_regularizer[0], l2=kernel_regularizer[1])))
        else:
            self.model.add(Dense(units=units, activation=self.dense_activation, kernel_regularizer = L1L2(l1=kernel_regularizer[0], l2=kernel_regularizer[1]), input_shape=input_shape))
        self.model.add(Dropout(keep_prob))

    def get_accuracy(self, X_test, Y_test):
        scores = self.model.evaluate(X_test, Y_test, verbose=0)
        return scores
    
    def train_cv(self, X_train, Y_train, X_val, Y_val, nb_lags, epochs=10):
        train_data_generator = KerasBatchGenerator(X_train, Y_train, self.batch_size, nb_lags+1)
        val_data_generator = KerasBatchGenerator(X_val, Y_val, self.batch_size, nb_lags+1)
        # Fitting the RNN to the Training set
        if self.optimizer=='SGD':
            self.history = self.model.fit_generator(train_data_generator.generate(), len(X_train), epochs=epochs,
                                                    validation_data=val_data_generator.generate(), validation_steps=len(X_val))
        else:   
            self.history = self.model.fit_generator(train_data_generator.generate(), len(X_train)/(self.batch_size), epochs=epochs,
                                                    validation_data=val_data_generator.generate(), validation_steps=len(X_val)/(self.batch_size))

    def train(self, X_train, Y_train, nb_lags, epochs=10):
        train_data_generator = KerasBatchGenerator(X_train, Y_train, self.batch_size, nb_lags+1)
        # Fitting the RNN to the Training set
        if self.optimizer=='SGD':
            self.history = self.model.fit_generator(train_data_generator.generate(), len(X_train), epochs=epochs)
        else:   
            self.history = self.model.fit_generator(train_data_generator.generate(), len(X_train)/(self.batch_size), epochs=epochs)
    
    def train_tscv(self, X_train, Y_train, epochs=10, n_splits=5):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        acc_per_fold = []
        loss_per_fold = []
        fold_no = 1
        for train, val in tscv.split(X_train):
        # Fitting the RNN to the Training set
            self.history = self.model.fit(X_train[train], Y_train[train], epochs=epochs, batch_size=self.batch_size, verbose=1)
            scores = self.get_accuracy(X_train[val], Y_train[val])
            acc_per_fold.append(scores[1] * 100)
            loss_per_fold.append(scores[0])
            fold_no = fold_no + 1
        print('------------------------------------------------------------------------')
        print('Average scores for all folds:')
        print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
        print('------------------------------------------------------------------------')

    def predict(self, X):
        y_hat = self.model.predict(X)
        return y_hat
     
    def fit_standardization(self, X):
        self.scaler = StandardScaler()
        if isinstance(X, gensim.models.keyedvectors.Doc2VecKeyedVectors):
            length = len(X)
            X_train = np.zeros((length,len(X[0])))
            for i in range(length):
                X_train[i,:] = X[i]    
        else:
            X_train = X    
        if len(X_train.shape) == 1:
            X_train = np.reshape(X_train, (X_train.shape[0],1))
        self.scaler.fit(X_train)
        return X_train

    def standardize(self, X):
        X_train = self.scaler.transform(X)
        return X_train

    @staticmethod
    def reshape(X, nb_lags):
        length = len(X)
        #if isinstance(X, gensim.models.keyedvectors.Doc2VecKeyedVectors):
        if len(X.shape) == 1:
            X = np.reshape(X, (X.shape[0],1))
        X_train = np.zeros((length-nb_lags, nb_lags+1, len(X[0])))
        #else:
        #    X_train = []
        for i in range(nb_lags, length):
            if isinstance(X, pd.DataFrame):
                X_train.append(X.iloc[i-nb_lags:i].values)
            else:# isinstance(X, gensim.models.keyedvectors.Doc2VecKeyedVectors):
                for j in range(nb_lags+1):
                    X_train[i-nb_lags,j,:] = X[j+i-nb_lags]
            #else:
            #    X_train.append(X[(i-nb_lags):(length-1)], axis=1)
        if not isinstance(X_train, np.ndarray):
            X_train = np.array(X_train)
            #X_train = X_train.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
        #print (X_train.shape)
        return X_train

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

    def get_lstm_layers(self):
        return self.lstm_layers

    def get_lstm_activation(self):
        return self.lstm_activation

    def get_dense_layers(self):
        return self.dense_layers

    def get_dense_activation(self):
        return self.dense_activation

    def get_num_layers(self):
        return self.num_layers

    def get_keep_prob(self):
        return self.keep_prob

    def get_batch_size(self):
        return self.batch_size

    def get_epochs(self):
        return self.epochs

    def get_learning_rate(self):
        return self.learning_rate

    def get_optimizer(self):
        return self.optimizer

    def set_lstm_layers(self, lstm_layers):
        self.lstm_layers = lstm_layers

    def set_lstm_activation(self, lstm_activation):
        self.lstm_activation = lstm_activation

    def set_dense_layers(self, dense_layers):
        self.dense_layers = dense_layers

    def set_dense_activation(self, dense_activation):
        self.dense_activation = dense_activation

    def set_keep_prob(self, keep_prob):
        self.keep_prob = keep_prob

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_epochs(self, epochs):
        self.epochs = epochs

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
