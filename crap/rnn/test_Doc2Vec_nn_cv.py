import pandas as pd
import numpy as np
from rnn.Doc2Vec import Doc_2_Vec
import data.data_tweets as d
from rnn.lstmConfig import lstmConfig
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
    data.df.Direction = data.df.Direction.shift(-1)
    data.df = data.df[0:(len(data.df)-1)]
    data_train = data.slice(0, int(0.7*size))
    Y_train = data_train.get_direction_num().values
    data_val = data.slice(int(0.7*size), int(0.85*size)) 
    Y_val = data_val.get_direction_num().values
    #All the models ran
    window_sizes = [2, 5, 7] 
    vec_sizes = [50, 200, 500]
    path1 = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/src/model/"
    path2 = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/src/Doc2Vec_data/"
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
            X_train = X_train[0:len(Y_train)]
            outfile_0 = str(path2) + "X_val_D_0_" + str(vec_size) + "_" + str(window_size) + ".npy"
            outfile_1 = str(path2) + "X_val_D_1_" + str(vec_size) + "_" + str(window_size) + ".npy" 
            X_val_0 = np.load(outfile_0)
            X_val_1 = np.load(outfile_1)
            X_val = np.concatenate((X_val_0, X_val_1), axis=1)
            print ('done...\n')
            print ('Building and training lstm model...\n')
            nbs_lags = [0]
            learning_rates = [0.001, 0.0001]
            bias_regularizers = [0.1, 1]
            lstm_layers =[[]]# [[16], [32], [64]]
            dense_layers = [[32], [64], [128], [256]]
            batch_sizes = [1]
            epochs = 5
            for nb_lags, learning_rate, bias_regularizer, dense_layer, batch_size in itertools.product(nbs_lags, learning_rates, bias_regularizers, dense_layers, batch_sizes):
                print("\n training model with " + str(nb_lags) + " lags, " + str(learning_rate)+ " learning rate, " + str(bias_regularizer) + " bias, " + str(dense_layer) + " dense layers, "  + str(batch_size) + " batch... \n")
                Y_train_reshape = Y_train[(nb_lags):]
                Y_val_reshape = Y_val[(nb_lags):]
                lstm_model = lstmConfig(batch_size=batch_size, learning_rate=learning_rate, bias_regularizer=[bias_regularizer,bias_regularizer], lstm_layers=[], dense_layers=dense_layer)
                X_train = lstm_model.fit_standardization(X_train)
                X_train_std = lstm_model.standardize(X_train)
                X_val_std = lstm_model.standardize(X_val)
                X_train_reshape = lstmConfig.reshape(X_train_std, nb_lags)
                X_val_reshape = lstmConfig.reshape(X_val_std, nb_lags)
                lstm_model.build_network(X_train_reshape)
                lstm_model.train_cv(X_train_reshape, Y_train_reshape, X_val_reshape, Y_val_reshape, nb_lags, epochs)
                
    pdb.set_trace()


if __name__ == '__main__':
    main()