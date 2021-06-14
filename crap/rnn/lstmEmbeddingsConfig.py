from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.utils import plot_model
from keras.optimizers import SGD, RMSprop, Adam
from matplotlib import pyplot as plt
from utils.parallel import parallel
from rnn.KerasBatchGenerator import KerasBatchGenerator
from keras.regularizers import L1L2
import numpy as np
import pandas as pd
import pdb
np.random.seed(42)

class lstmEmbeddingsConfig(parallel):

    model_build = False
    embeddings_index = dict()
    embedding_matrix = None
    Glove_file = ""
    tokenizer = None

    def __init__(self, embedding_dimension, vocabulary_size, maxlen, Glove):
        np.random.seed(42)
        self.embedding_dimension = embedding_dimension
        self.vocabulary_size = vocabulary_size
        if Glove is True:
            self.Glove = True    
            if embedding_dimension == 50:
                self.Glove_file = "C:/Users/alexa/OneDrive/Desktop/Thesis/Glove/glove.6B.50d.txt"
            elif embedding_dimension == 100:
                self.Glove_file = "C:/Users/alexa/OneDrive/Desktop/Thesis/Glove/glove.6B.100d.txt"
            elif embedding_dimension == 200:
                self.Glove_file = "C:/Users/alexa/OneDrive/Desktop/Thesis/Glove/glove.6B.200d.txt"
            else:
                self.Glove_file = "C:/Users/alexa/OneDrive/Desktop/Thesis/Glove/glove.6B.300d.txt"
                self.embedding_dimension = 300
            # Initialising the Glove embedding 
            self._Glove_initializer()
        else:
            self.Glove = False
        # Initialising the RNN
        self.model = Sequential()
        self.history = None
        self.maxlen = maxlen
        self.optimizer = None

    def get_model(self):
        if self.model_build is False:
            print("Model has not been build yet, please call function train")
            return
        else:
            return self.model
    def get_maxlen(self):
        return self.maxlen

    def set_maxlen(self, maxlen):
        self.maxlen = maxlen

    def get_vocabulary_size(self):
        return self.vocabulary_size

    def set_vocabulary_size(self, vocabulary_size):
        self.vocabulary_size = vocabulary_size

    def get_embedding_dimension(self):
        return self.embedding_dimension

    def set_embedding_dimension(self, embedding_dimension):
        self.embedding_dimension = embedding_dimension
        if self.Glove is True:
            if embedding_dimension == 50:
                self.Glove_file = "C:/Users/alexa/OneDrive/Desktop/Thesis/Glove/glove.6B.50d.txt"
            elif embedding_dimension == 100:
                self.Glove_file = "C:/Users/alexa/OneDrive/Desktop/Thesis/Glove/glove.6B.100d.txt"
            elif embedding_dimension == 200:
                self.Glove_file = "C:/Users/alexa/OneDrive/Desktop/Thesis/Glove/glove.6B.200d.txt"
            else:
                self.Glove_file = "C:/Users/alexa/OneDrive/Desktop/Thesis/Glove/glove.6B.300d.txt"
                self.embedding_dimension = 300
            # Initialising the Glove embedding
            self._Glove_initializer(self)

    def get_Glove(self):
        return self.Glove

    def set_Glove(self, Glove):
        if self.Glove is True:
            self.Glove = Glove
        else:
            if Glove is True:
                self.Glove = True
                if self.embedding_dimension == 50:
                    self.Glove_file = "C:/Users/alexa/OneDrive/Desktop/Thesis/Glove/glove.6B.50d.txt"
                elif self.embedding_dimension == 100:
                    self.Glove_file = "C:/Users/alexa/OneDrive/Desktop/Thesis/Glove/glove.6B.100d.txt"
                elif self.embedding_dimension == 200:
                    self.Glove_file = "C:/Users/alexa/OneDrive/Desktop/Thesis/Glove/glove.6B.200d.txt"
                else:
                    self.Glove_file = "C:/Users/alexa/OneDrive/Desktop/Thesis/Glove/glove.6B.300d.txt"
            # Initialising the Glove embedding
                self._Glove_initializer(self)

    def _train_tokenizer(self, X_train):
        self.tokenizer = Tokenizer(num_words= self.vocabulary_size)
        self.tokenizer.fit_on_texts(X_train)

    def _create_sequence(self, X):
        sequences = self.tokenizer.texts_to_sequences(X)
        X = pad_sequences(sequences, maxlen=self.maxlen)
        return X

    def _Glove_initializer(self):
        f = open(self.Glove_file, encoding="utf8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
        f.close()

    def _create_embeddings(self):
        # create a weight matrix for words in training docs
        self.embedding_matrix = np.zeros((self.vocabulary_size, self.embedding_dimension))
        for word, index in self.tokenizer.word_index.items():
            if index > self.vocabulary_size - 1:
                break
            else:
                embedding_vector = self.embeddings_index.get(word)
                if embedding_vector is not None:
                    self.embedding_matrix[index] = embedding_vector

    def build_network(self, X_train,  optimizer='Adam', learning_rate=0.01, lstm_units=32, dense_units=32, dropout=0.2):
        #Formatting data
        self. _train_tokenizer(X_train)
        if self.Glove is True:
            self._create_embeddings()
        self.model_build = True
        if self.Glove is True:
            self.model.add(Embedding(self.vocabulary_size, self.embedding_dimension, input_length=self.maxlen, weights=[self.embedding_matrix], trainable=False))
        else:
            self.model.add(Embedding(self.vocabulary_size, self.embedding_dimension, input_length=self.maxlen))
        self.model.add(LSTM(lstm_units, activation='relu', bias_regularizer = L1L2(l1=0.1, l2=0.1)))
        # Adding the output layer          
        self.model.add(Dropout(dropout))
        self.model.add(Dense(units=dense_units, activation='relu', bias_regularizer = L1L2(l1=0.1, l2=0.1)))
        if X_train.ndim == 3:
            self.model.add(Flatten())
        self.model.add(Dense(units=1, activation='sigmoid'))
        # Compiling the RNN
        self.optimizer = optimizer
        if optimizer=='SGD':
            opt = SGD  (lr=learning_rate)
        elif optimizer=='RMSprop':
            opt = RMSprop(lr=learning_rate)
        else:
            opt = Adam(lr=learning_rate)
        self.model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    def get_accuracy(self, X_test, Y_test):
        X_test = self._create_sequence(X_test)
        scores = self.model.evaluate(X_test, Y_test, verbose=0)
        return scores

    def predict(self, X):
        X = self._create_sequence(X)
        y_hat = self.model.predict(X)
        return y_hat

    def train(self, X_train, Y_train, X_val, Y_val, epochs=10, batch_size=32):
        #Building the network
        if self.model_build is True:
            if self.optimizer == "SGD":
                batch_size = 1
            X_train = self._create_sequence(X_train)
            X_val = self._create_sequence(X_val)
            train_data_generator = KerasBatchGenerator(X_train, Y_train, batch_size, self.maxlen)
            val_data_generator = KerasBatchGenerator(X_val, Y_val, batch_size, self.maxlen)
             # Fitting the RNN to the Training set
            self.history = self.model.fit_generator(train_data_generator.generate(), len(X_train)/(batch_size), epochs=epochs,
                                                    validation_data=val_data_generator.generate(), validation_steps=len(X_val)/(batch_size))
        else:
            print("Build model first")

    def plot_graph(self, file):
        plot_model(self.model, to_file = file + '.png')

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
