import numpy as np
import pandas as pd
import math
import warnings
import time
import matplotlib.pyplot as plt 
import data.data_tweets as dat
import rnn.lstmEmbeddingsConfig as l
import pdb
import ast
from termcolor import colored
import itertools


def run_experiment(batch_sizes, learning_rates, lstm_units, dense_units, dropouts, X_train, Y_train, X_val, Y_val, maxlen, epochs):
    optimizer = "Adam"
    Glove = False
    vocabulary_size = 50000
    embedding_dimension = 200
    model = l.lstmEmbeddingsConfig(embedding_dimension, vocabulary_size, maxlen, Glove)
    for batch_sizes_elem, learning_rates_elem, lstm_units_elem, dense_units_elem, dropouts_elem in itertools.product(batch_sizes, learning_rates, lstm_units, dense_units, dropouts):
        print ('Building and training model...\n')
        model = l.lstmEmbeddingsConfig(embedding_dimension, vocabulary_size, maxlen, Glove)
        print (batch_sizes_elem, learning_rates_elem, lstm_units_elem, dense_units_elem,dropouts_elem)
        model.build_network(X_train,  optimizer=optimizer, learning_rate=learning_rates_elem, lstm_units=lstm_units_elem, dense_units=dense_units_elem, dropout=dropouts_elem)
        model.train(X_train, Y_train, X_val, Y_val, epochs, batch_sizes_elem)

def main():
    """
    Execute matching action for testing
    """
    start = time.time()

    print('\n')
    print(colored('------------------------------------------------------------------', 'red'))
    print(colored('                      RNN for sentiment 140                      |', 'red'))
    print(colored('------------------------------------------------------------------', 'red'))
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/sentiment140.txt"
    freq = 's'
    print('\n')
    print(colored('preprocessing...\n', 'magenta'))
    data = dat.data_tweets(fname, freq)
    data.shuffle()
    size = data.get_length()
    
    data_copy = data.slice(0, int(0.2*size))
    size = data_copy.get_length()
    X = data_copy.get_text().values.ravel()
    for i in range(0, len(X)):
        temp = ast.literal_eval(X[i])
        X[i] = " ".join(temp)
    Y = data_copy.df.sentiment.values
    Y = np.reshape(Y, (Y.shape[0], 1))
    Y = np.where(Y==4, 1, Y) 

    X_train = X[0:int(0.98*size)]
    Y_train = Y[0:int(0.98*size)]
    X_val = X[int(0.98*size):int(0.99*size)]
    Y_val = Y[int(0.98*size):int(0.99*size)]
    maxlen = max(len(ast.literal_eval(ele))for ele in data.get_text())
    print('max length is ' + str(maxlen))
    print ('\n')
    
    print(colored('------------------------------------------------------------------', 'red'))
    print(colored('                         Embedding                               |', 'red'))
    print(colored('------------------------------------------------------------------', 'red'))    
    print(colored('------------------------------------------------------------------', 'red'))
    print(colored('                     Parameters tuning                           |', 'red'))
    print(colored('------------------------------------------------------------------', 'red'))
    epochs = 4
    batch_sizes = [32, 64, 128]
    optimizer = "Adam"
    learning_rates = [0.001, 0.0005, 0.0001] 
    lstm_units = [64, 128]
    dropouts = [0.1, 0.3, 0.5]
    dense_units = [16, 32]
    run_experiment(batch_sizes, learning_rates, lstm_units, dense_units, dropouts, X_train, Y_train, X_val, Y_val, maxlen, epochs)
    print ('done...\n')
    
    print(colored('------------------------------------------------------------------', 'red'))
    print(colored('                         Testing                                  |', 'red'))
    print(colored('------------------------------------------------------------------', 'red'))
    optimizer = "Adam"
    Glove = False
    vocabulary_size = 50000
    embedding_dimension = 200
    epochs = 2
    batch_size = 32
    optimizer = "Adam"
    learning_rate = 0.001
    lstm_units = 128
    dropout = 0.3
    dense_units = 32

    size = data.get_length()
    X = data.get_text().values.ravel()
    for i in range(0, len(X)):
        temp = ast.literal_eval(X[i])
        X[i] = " ".join(temp)
    Y = data.df.sentiment.values
    Y = np.reshape(Y, (Y.shape[0], 1))
    Y = np.where(Y==4, 1, Y) 

    X_train = X[0:int(0.98*size)]
    Y_train = Y[0:int(0.98*size)]
    X_val = X[int(0.98*size):int(0.99*size)]
    Y_val = Y[int(0.98*size):int(0.99*size)]
    X_test = X[int(0.99*size):int(size)]
    Y_test = Y[int(0.99*size):int(size)]
    
    model = l.lstmEmbeddingsConfig(embedding_dimension, vocabulary_size, maxlen, Glove)
    model.build_network(X_train, optimizer=optimizer, learning_rate=learning_rate, lstm_units=lstm_units, dense_units=dense_units, dropout=dropout)
    model.train(X_train, Y_train, X_val, Y_val, epochs, batch_size)
    accuracy = model.get_accuracy(X_test, Y_test)[1]
    print("accuracy on the test set: " + str(np.round(accuracy*100,2)) + "%")
    end = time.time()
    print(end - start)

if __name__ == '__main__':
    main()