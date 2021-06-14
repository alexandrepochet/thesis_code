import pandas as pd
import numpy as np
from rnn.Doc2Vec import Doc_2_Vec
import data.data_tweets as d
from rnn.lstmConfig import lstmConfig
from  rnn.currencyPredict import currencyPredict
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

    ##Daily data
    warnings.filterwarnings("ignore")
    start = time.time()
     
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_sentiment_D.txt"
    freq = 'D'
    print ('preprocessing...\n')
    data = d.data_tweets(fname, freq)
    size = data.get_length()
    data.df["count"] = data.df["count"].shift(1)
    data.df.text = data.df.text.shift(1)
    data.df = data.df[1:len(data.df)]
    data_train = data.slice(0, int(0.7*size))
    data_val = data.slice(int(0.7*size), int(0.85*size)) 
    #All the models ran
    window_sizes = [2, 5, 7] 
    vec_sizes = [50, 200, 500]
    path1 = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/src/model/"
    path2 = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/src/Doc2Vec_data/"
    print ('done...\n')
    for window_size in window_sizes:
        for vec_size in vec_sizes:
            print ('opening model with dimension' + str(vec_size) + ' and window ' + str(window_size) + '...\n')
            model_0 = str(path1) + "Doc2Vec_D_0_" + str(vec_size) + "_" + str(window_size) + ".model"
            model_1 = str(path1) + "Doc2Vec_D_1_" + str(vec_size) + "_" + str(window_size) + ".model"
            Doc2Vec_model_0 = Doc_2_Vec()
            Doc2Vec_model_0.load(model_0)
            X_train_0 = Doc2Vec_model_0.get_vectors()
            Doc2Vec_model_1 = Doc_2_Vec()
            Doc2Vec_model_1.load(model_1)
            X_train_1 = Doc2Vec_model_1.get_vectors()
            X_train = np.concatenate((Doc_2_Vec.keyedvectors_to_array(X_train_0), Doc_2_Vec.keyedvectors_to_array(X_train_1)), axis=1) 
            outfile_0 = str(path2) + "X_val_D_0_" + str(vec_size) + "_" + str(window_size) + ".npy"
            outfile_1 = str(path2) + "X_val_D_1_" + str(vec_size) + "_" + str(window_size) + ".npy" 
            X_val_0 = np.load(outfile_0)
            X_val_1 = np.load(outfile_1)
            X_val = np.concatenate((X_val_0, X_val_1), axis=1)
            X_train = np.concatenate((X_val, X_train), axis=0)
            Y_val = data_val.get_direction_num().values
            Y_train = data_train.get_direction_num().values
            Y_train = np.concatenate((Y_val, Y_train), axis=0)
            print ('done...\n')
            print ('Building and training lstm model...\n')
            nbs_lags = [0]
            learning_rates = [0.001, 0.0001]
            kernel_regularizers = [0.00001, 0.0001]
            dense_layers = [[32], [64], [128]]
            batch_sizes = [1]
            epochs = 3
            n_splits = 5
            for nb_lags, learning_rate, kernel_regularizer, dense_layer, batch_size in itertools.product(nbs_lags, learning_rates, kernel_regularizers, dense_layers, batch_sizes):
                reset_keras()
                print("\n training model with " + str(nb_lags) + " lags, " + str(learning_rate)+ " learning rate, " + str(kernel_regularizer) + " kernel, " + str(dense_layer) + " dense layers, " + str(batch_size) + " batch size... \n")
                Y_train_reshape = Y_train[(nb_lags):]
                lstm_model = lstmConfig(batch_size=batch_size, learning_rate=learning_rate, kernel_regularizer=[kernel_regularizer,kernel_regularizer], lstm_layers=[], dense_layers=dense_layer)
                #X_train = lstm_model.fit_standardization(X_train)
                #X_train_std = lstm_model.standardize(X_train)
                X_train_reshape = lstmConfig.reshape(X_train, nb_lags)
                lstm_model.build_network(X_train_reshape)
                lstm_model.train_tscv(X_train_reshape, Y_train_reshape, epochs, n_splits)
            
    #best results for window 5, dimension 200, 0.001 learning rate, 1 kernel, [32] dense layers
    print ('opening model with dimension 200 and window 2...\n')
    model_0 = str(path1) + "Doc2Vec_D_0_200_5.model"
    model_1 = str(path1) + "Doc2Vec_D_1_200_5.model"
    Doc2Vec_model_0 = Doc_2_Vec()
    Doc2Vec_model_0.load(model_0)
    X_train_0 = Doc2Vec_model_0.get_vectors()
    Doc2Vec_model_1 = Doc_2_Vec()
    Doc2Vec_model_1.load(model_1)
    X_train_1 = Doc2Vec_model_1.get_vectors()
    X_train = np.concatenate((Doc_2_Vec.keyedvectors_to_array(X_train_0), Doc_2_Vec.keyedvectors_to_array(X_train_1)), axis=1) 
    print ('loading test data...\n')
    #Opening test data
    outfile_0 = str(path2) + "X_test_D_0_200_5.npy"
    outfile_1 = str(path2) + "X_test_D_1_200_5.npy"
    X_test_0 = np.load(outfile_0)
    X_test_1 = np.load(outfile_1)
    X_test = np.concatenate((X_test_0, X_test_1), axis=1)
    X_test = X_test[0:(len(X_test)-1)]
    data_test = data.slice(int(0.85*size), int(size)) 
    Y_test = data_test.get_direction_num().values
    Y_train = data_train.get_direction_num().values
    nb_lags = 0
    learning_rate = 0.001
    kernel_regularizer = 0.00001
    lstm_layer = []
    dense_layer = [32]
    batch_size = 1
    epochs = 2

    Y_train_reshape = Y_train[nb_lags:]
    Y_test_reshape = Y_test[nb_lags:]
    reset_keras()
    lstm_model = lstmConfig(batch_size=batch_size, learning_rate=learning_rate, kernel_regularizer=[kernel_regularizer,kernel_regularizer], lstm_layers=[], dense_layers=dense_layer)
    X_train_reshape = lstmConfig.reshape(X_train, nb_lags)
    X_test_reshape = lstmConfig.reshape(X_test, nb_lags)
    lstm_model.build_network(X_train_reshape)
    lstm_model.train(X_train_reshape, Y_train_reshape, nb_lags, epochs)
    print("\n Accuracy on test data for daily data: " + str(lstm_model.get_accuracy(X_test_reshape, Y_test_reshape)))
    market_returns = data_test.df.Return
    bid_ask = data_test.df.close_bid_ask
    currency_predict = currencyPredict(lstm_model, Y_test_reshape, market_returns, bid_ask)
    y_pred = currency_predict.predict(X_test_reshape)
    currency_predict.plot_return("Doc2Vec_daily", dash=True)
 
    ##hourly data
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_sentiment_H.txt"
    freq = 'H'
    print ('preprocessing...\n')
    data = d.data_tweets(fname, freq)
    size = data.get_length()
    data.df["count"] = data.df["count"].shift(1)
    data.df.text = data.df.text.shift(1)
    data.df = data.df[1:len(data.df)]
    data_train = data.slice(0, int(0.7*size))
    data_val = data.slice(int(0.7*size), int(0.85*size)) 
    #All the models ran
    window_sizes = [2, 5, 7] 
    vec_sizes = [50, 200, 500]
    path1 = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/src/model/"
    path2 = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/src/Doc2Vec_data/"
    
    for window_size in window_sizes:
        for vec_size in vec_sizes:
            print ('opening model with dimension' + str(vec_size) + ' and window ' + str(window_size) + '...\n')
            model_0 = str(path1) + "Doc2Vec_H_0_" + str(vec_size) + "_" + str(window_size) + ".model"
            model_1 = str(path1) + "Doc2Vec_H_1_" + str(vec_size) + "_" + str(window_size) + ".model"
            Doc2Vec_model_0 = Doc_2_Vec()
            Doc2Vec_model_0.load(model_0)
            X_train_0 = Doc2Vec_model_0.get_vectors()
            Doc2Vec_model_1 = Doc_2_Vec()
            Doc2Vec_model_1.load(model_1)
            X_train_1 = Doc2Vec_model_1.get_vectors()
            X_train = np.concatenate((Doc_2_Vec.keyedvectors_to_array(X_train_0), Doc_2_Vec.keyedvectors_to_array(X_train_1)), axis=1) 
            outfile_0 = str(path2) + "X_val_H_0_" + str(vec_size) + "_" + str(window_size) + ".npy"
            outfile_1 = str(path2) + "X_val_H_1_" + str(vec_size) + "_" + str(window_size) + ".npy" 
            X_val_0 = np.load(outfile_0)
            X_val_1 = np.load(outfile_1)
            X_val = np.concatenate((X_val_0, X_val_1), axis=1)
            X_train = np.concatenate((X_val, X_train), axis=0)
            Y_val = data_val.get_direction_num().values
            Y_train = data_train.get_direction_num().values
            Y_train = np.concatenate((Y_val, Y_train), axis=0)
            print ('done...\n')
            print ('Building and training lstm model...\n')
            nbs_lags = [0]
            learning_rates = [0.001, 0.0001]
            kernel_regularizers = [0.00001, 0.0001]
            dense_layers = [[32], [64], [128], [256]]
            batch_sizes = [1]
            epochs = 2
            n_splits = 5
            for nb_lags, learning_rate, kernel_regularizer, dense_layer, batch_size in itertools.product(nbs_lags, learning_rates, kernel_regularizers, dense_layers, batch_sizes):
                reset_keras()
                print("\n training model with " + str(nb_lags) + " lags, " + str(learning_rate)+ " learning rate, " + str(kernel_regularizer) + " kernel, " + str(dense_layer) + " dense layers, " + str(batch_size) + " batch size... \n")
                Y_train_reshape = Y_train[(nb_lags):]
                lstm_model = lstmConfig(batch_size=batch_size, learning_rate=learning_rate, kernel_regularizer=[kernel_regularizer,kernel_regularizer], lstm_layers=[], dense_layers=dense_layer)
                #X_train = lstm_model.fit_standardization(X_train)
                #X_train_std = lstm_model.standardize(X_train)
                X_train_reshape = lstmConfig.reshape(X_train, nb_lags)
                lstm_model.build_network(X_train_reshape)
                lstm_model.train_tscv(X_train_reshape, Y_train_reshape, epochs, n_splits)
    
    #best results for window 2, dimension 200, 0.001 learning rate, 1 kernel, [32] dense layers
    print ('opening model with dimension 200 and window 2...\n')
    model_0 = str(path1) + "Doc2Vec_H_0_200_2.model"
    model_1 = str(path1) + "Doc2Vec_H_1_200_2.model"
    Doc2Vec_model_0 = Doc_2_Vec()
    Doc2Vec_model_0.load(model_0)
    X_train_0 = Doc2Vec_model_0.get_vectors()
    Doc2Vec_model_1 = Doc_2_Vec()
    Doc2Vec_model_1.load(model_1)
    X_train_1 = Doc2Vec_model_1.get_vectors()
    X_train = np.concatenate((Doc_2_Vec.keyedvectors_to_array(X_train_0), Doc_2_Vec.keyedvectors_to_array(X_train_1)), axis=1) 
    print ('loading test data...\n')
    #Opening test data
    outfile_0 = str(path2) + "X_test_H_0_200_2.npy"
    outfile_1 = str(path2) + "X_test_H_1_200_2.npy"
    X_test_0 = np.load(outfile_0)
    X_test_1 = np.load(outfile_1)
    X_test = np.concatenate((X_test_0, X_test_1), axis=1)
    X_test = X_test[0:(len(X_test)-1)]
    data_test = data.slice(int(0.85*size), int(size)) 
    Y_test = data_test.get_direction_num().values
    Y_train = data_train.get_direction_num().values
    nb_lags = 0
    learning_rate = 0.001
    kernel_regularizer = 0.00001
    lstm_layer = []
    dense_layer = [32]
    batch_size = 1
    epochs = 2

    Y_train_reshape = Y_train[nb_lags:]
    Y_test_reshape = Y_test[nb_lags:]
    reset_keras()
    lstm_model = lstmConfig(batch_size=batch_size, learning_rate=learning_rate, kernel_regularizer=[kernel_regularizer,kernel_regularizer], lstm_layers=[], dense_layers=dense_layer)
    X_train_reshape = lstmConfig.reshape(X_train, nb_lags)
    X_test_reshape = lstmConfig.reshape(X_test, nb_lags)
    lstm_model.build_network(X_train_reshape)
    lstm_model.train(X_train_reshape, Y_train_reshape, nb_lags, epochs)
    print("\n Accuracy on test data for hourly data: " + str(lstm_model.get_accuracy(X_test_reshape, Y_test_reshape)))
    market_returns = data_test.df.Return
    bid_ask = data_test.df.close_bid_ask
    currency_predict = currencyPredict(lstm_model, Y_test_reshape, market_returns, bid_ask)
    y_pred = currency_predict.predict(X_test_reshape)
    currency_predict.plot_return("Doc2Vec_hourly", dash=False)

    end = time.time()
    print(end - start)
            
if __name__ == '__main__':
    main()