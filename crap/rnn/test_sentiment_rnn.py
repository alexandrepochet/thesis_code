import pandas as pd
import numpy as np
import data.data_sentiments as d
from crap.rnn.lstmConfig import lstmConfig
from  crap.rnn.currencyPredict import currencyPredict
from  utils.utils import reset_keras
import pdb
import time
import warnings
import ast
import itertools



def main(load=1): 
    """  
    Execute matching action for testing  
    """

    warnings.filterwarnings("ignore")
    start = time.time()

    ##Daily data
    #file where all the sentiment time series are stored
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/daily_sentiment_series.txt"
    freq = 'D'
    print ('preprocessing...\n')
    data = d.data_sentiments(fname, freq)
    size = data.get_length()
    cols = data.df.columns
    cols = cols[4:]
    for col in cols:
        data.df[col] = data.df[col].shift(1)
    #data.df.Return = data.df.Return.shift(1)
    data.df = data.df[1:(size)]
    data_train_val = data.slice(0, int(0.85*size))
    X_train_val = data_train_val.df.iloc[:,4:]
    #X_train_val = data_train_val.df.Return
    X_train_val = X_train_val.values
    Y_train_val = data_train_val.get_direction_num().values
    print ('done...\n')

    #First try an easy model
    print ('Building and training lstm model...\n')
    nbs_lags = [0, 1, 2, 5, 10]
    learning_rates = [0.0001, 0.001, 0.01]
    kernel_regularizers = [0.0001, 0.001]
    dense_layers = [[8, 2],  [16, 4], [64, 16]]
    batch_sizes = [1, 8, 32]
    epochs = 10
    n_splits = 5
    for nb_lags, learning_rate, kernel_regularizer, dense_layer, batch_size in itertools.product(nbs_lags, learning_rates, kernel_regularizers, dense_layers, batch_sizes):
        reset_keras()
        Y_train_val_reshape = Y_train_val[(nb_lags):]
        print("\n training model with " + str(nb_lags) + " lags, " + str(learning_rate)+ " learning rate, " + str(kernel_regularizer) + " kernel, " + str(dense_layer) + " dense layers, " + str(batch_size) + " batch size... \n")
        lstm_model = lstmConfig(batch_size=batch_size, learning_rate=learning_rate, kernel_regularizer=[kernel_regularizer,kernel_regularizer], lstm_layers=[], dense_layers=dense_layer)
        X_train_val = lstm_model.fit_standardization(X_train_val)
        X_train_val_std = lstm_model.standardize(X_train_val)
        X_train_val_reshape = lstmConfig.reshape(X_train_val_std, nb_lags)
        lstm_model.build_network(X_train_val_reshape)
        lstm_model.train_tscv(X_train_val_reshape, Y_train_val_reshape, epochs, n_splits)

    end = time.time()
    print(end - start)
    pdb.set_trace()
    ##Hourly data
    #file where all the sentiment time series are stored
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/hourly_sentiment_series.txt"
    freq = 'H'
    print ('preprocessing...\n')
    data = d.data_sentiments(fname, freq)
    size = data.get_length()
    cols = data.df.columns
    cols = cols[4:]
    for col in cols:
        data.df[col] = data.df[col].shift(1)
    #data.df.Return = data.df.Return.shift(1)
    data.df = data.df[1:(size)]
    data_train_val = data.slice(0, int(0.85*size))
    X_train_val = data_train_val.df.iloc[:,4:]
    #X_train_val = data_train_val.df.Return
    X_train_val = X_train_val.values
    Y_train_val = data_train_val.get_direction_num().values
    print ('done...\n')

    #First try an easy model
    print ('Building and training lstm model...\n')
    nbs_lags = [0, 1, 2, 5, 10]
    learning_rates = [0.0001, 0.001, 0.01]
    kernel_regularizers = [0.0001, 0.001]
    dense_layers = [[8, 2],  [16, 4], [64, 16]]
    batch_sizes = [1, 8, 32]
    epochs = 5
    n_splits = 5
    for nb_lags, learning_rate, kernel_regularizer, dense_layer, batch_size in itertools.product(nbs_lags, learning_rates, kernel_regularizers, dense_layers, batch_sizes):
        reset_keras()
        Y_train_val_reshape = Y_train_val[(nb_lags):]
        print("\n training model with " + str(nb_lags) + " lags, " + str(learning_rate)+ " learning rate, " + str(kernel_regularizer) + " kernel, " + str(dense_layer) + " dense layers, " + str(batch_size) + " batch size... \n")
        lstm_model = lstmConfig(batch_size=batch_size, learning_rate=learning_rate, kernel_regularizer=[kernel_regularizer,kernel_regularizer], lstm_layers=[], dense_layers=dense_layer)
        X_train_val = lstm_model.fit_standardization(X_train_val)
        X_train_val_std = lstm_model.standardize(X_train_val)
        X_train_val_reshape = lstmConfig.reshape(X_train_val_std, nb_lags)
        lstm_model.build_network(X_train_val_reshape)
        lstm_model.train_tscv(X_train_val_reshape, Y_train_val_reshape, epochs, n_splits)

    end = time.time()
    print(end - start)


if __name__ == '__main__':
    main()