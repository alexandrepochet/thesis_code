import pandas as pd
import numpy as np
import data.data_sentiments as d
from rnn.lstmConfig import lstmConfig
from utils.GridSearchCV2 import GridSearchCV2
from utils.ShiftingWindowSplit import ShiftingWindowSplit
from utils.ExpandingWindowSplit import ExpandingWindowSplit
from utils.currencyReturn import currencyReturn
from utils.utils import reset_keras
from silence_tensorflow import silence_tensorflow
import itertools
import time
import pdb
import warnings
import math


def main(load=1): 
    """  
    Execute matching action for testing  
    """

    warnings.filterwarnings("ignore")
    silence_tensorflow()
    start = time.time()

    ##Daily data
    #file where all the sentiment time series are stored
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/daily_sentiment_series2.txt"
    freq = 'D'
    print ('preprocessing...\n')
    data = d.data_sentiments(fname, freq)
    size = data.get_length()
    X = data.get_df()
    cols = X.columns
    cols = cols[4:]
    for col in cols:
        X[col] = X[col].shift(1)
    X = X.iloc[1:]
    X = X.drop(['Direction', 'close_bid_ask', 'Return', 'close_bid_ask_previous', 'close'], axis=1)
    X = np.asarray(X)
    Y = data.get_direction_num()
    Y = np.asarray(Y.values)

    nb_lags = [5, 10]
    batch_sizes = [32,64,128]
    keep_probs=[[0.2,0.2,0.2], [0.4,0.4,0.4]] 
    kernel_regularizers=[[0.001,0.001],[0.01,0.01]]
    learning_rates = [0.0001,0.001,0.01]
    lstm_layers=[[64, 32], [32,16], [16,8]]
    # Test
    epochs = 100
    tscv_outer = ExpandingWindowSplit(test_size=50, train_size_init=500)
    acc_per_fold = np.zeros((20, len(nb_lags), len(batch_sizes), len(keep_probs), len(kernel_regularizers), len(learning_rates), len(lstm_layers), epochs))
    fold_no = 1
    for train, test in tscv_outer.split(X):
        lag = 0
        start = time.time()
        print("fold no: " + str(fold_no))
        for nb_lag in nb_lags:
            batch = 0
            for batch_size in batch_sizes:
                prob = 0 
                for keep_prob in keep_probs:
                    kernel = 0
                    for kernel_regularizer in kernel_regularizers:
                        learning = 0
                        for learning_rate in learning_rates:
                            lstm = 0
                            for lstm_layer in lstm_layers:
                                reset_keras()
                                model = lstmConfig()
                                X_train = X[train]
                                Y_train = Y[train]
                                X_test = X[test]
                                Y_test = Y[test]
                                history, message = model.fit(X_train, Y_train, X_val=X_test, Y_val=Y_test, nb_lags=nb_lag, batch_size=batch_size, lstm_layers=lstm_layer,
                                                             keep_prob=keep_prob, kernel_regularizer=kernel_regularizer, learning_rate=learning_rate, epochs=epochs)
                                if message is None:
                                    acc_per_fold[fold_no-1,lag, batch, prob, kernel, learning, lstm, :] = history.history['val_accuracy']
                                    pdb.set_trace()
                                lstm += 1
                            learning += 1
                        kernel += 1
                    prob += 1
                batch += 1
            lag += 1
        fold_no = fold_no + 1
        end = time.time()
        print(end - start)
    pdb.set_trace()

if __name__ == '__main__':
    main()