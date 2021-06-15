import pandas as pd
import numpy as np
import copy
import data.data_sentiments as d
from rnn.lstmConfig_custom import lstmConfig
from utils.GridSearchCV_custom import GridSearchCV_custom
from utils.ShiftingWindowSplit import ShiftingWindowSplit
from utils.ExpandingWindowSplit import ExpandingWindowSplit
from utils.currencyReturn import currencyReturn
from utils.utils import reset_keras
from silence_tensorflow import silence_tensorflow
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
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/daily_sentiment_series.txt"
    freq = 'D'
    print ('preprocessing...\n')
    data = d.data_sentiments(fname, freq)
    size = data.get_length()
    X = data.get_df()
    X = X.drop(['Direction', 'close_bid_ask', 'Return', 'close_bid_ask_previous', 'close'], axis=1)
    print('Drop McLoughran')
    X = X.drop(['assoc_fin_pos_positive', 'assoc_fin_pos_negative',
       'assoc_fin_pos_litigious', 'assoc_fin_pos_constraining',
       'assoc_fin_pos_uncertainty', 'assoc_fin_pos_strong_modal',
       'assoc_fin_pos_moderate_modal', 'assoc_fin_pos_weak_modal',
       'assoc_fin_positive', 'assoc_fin_negative', 'assoc_fin_litigious',
       'assoc_fin_constraining', 'assoc_fin_uncertainty',
       'assoc_fin_strong_modal', 'assoc_fin_moderate_modal',
       'assoc_fin_weak_modal'], axis=1)
    X = np.asarray(X)
    Y = data.get_direction_num()
    Y = np.asarray(Y.values)
    pdb.set_trace()
    nb_lags = [5, 10]
    batch_sizes = [32,64]
    keep_probs=[[0.2,0.2,0.2], [0.4,0.4,0.4]] 
    kernel_regularizers=[[0,0], [0.001,0.001],[0.0001,0.0001]]
    learning_rates = [0.001]
    lstm_layers=[[64, 32], [128,64], [32,16]]#
    space = {"nb_lags": nb_lags, "batch_size": batch_sizes, "keep_prob":keep_probs, "kernel_regularizer":kernel_regularizers, 
             "learning_rate":learning_rates, "lstm_layers":lstm_layers, "epochs":[100]}
    # Test
    tscv_outer = ExpandingWindowSplit(test_size=50, train_size_init=500)
    acc_per_fold = []
    fold_no = 1
    predicted_directions = []
    best_accuracy_per_fold = []
    best_loss_per_fold = []
    previous = None
    previous_training = None
    previous_accuracy = None
    previous_accuracy_training = None
    
    for train, test in tscv_outer.split(X):
        start = time.time()
        print("fold no: " + str(fold_no))
        reset_keras()
        model = lstmConfig()
        X_train = X[train]
        Y_train = Y[train]
        X_test = X[test]
        Y_test = Y[test]
        tscv_inner = ExpandingWindowSplit(test_size=50, train_size_init=len(train)-150)
        search = GridSearchCV_custom(model, space, tscv_inner)
        return_vec = data.get_return().iloc[train].values.ravel()
        dim = return_vec.shape[0]
        return_vec_train = np.reshape(data.get_return().iloc[train].values.ravel(), (dim,1))
        
        bid_ask = data.get_close_bid_ask().iloc[train].values.ravel()
        bid_ask_previous = data.get_close_bid_ask_previous().iloc[train].values.ravel()
        bid_ask = np.reshape(bid_ask, (1, dim))
        bid_ask_previous = np.reshape(bid_ask_previous, (1, dim))
        tcost_vec = copy.deepcopy(bid_ask)
        tcost_vec[0, 0] = tcost_vec[0, 0] + bid_ask_previous[0, 0]/2
        tcost_vec[0, dim-1] = bid_ask[0,dim-1]/2
        tcost_vec_train = np.reshape(tcost_vec, (dim,1)) 
        Y_train = np.concatenate((Y_train, return_vec_train, tcost_vec_train), axis=1)

        return_vec = data.get_return().iloc[test].values.ravel()
        dim = return_vec.shape[0]
        return_vec_test = np.reshape(data.get_return().iloc[test].values.ravel(), (dim,1))
        
        bid_ask = data.get_close_bid_ask().iloc[test].values.ravel()
        bid_ask_previous = data.get_close_bid_ask_previous().iloc[test].values.ravel()
        bid_ask = np.reshape(bid_ask, (1, dim))
        bid_ask_previous = np.reshape(bid_ask_previous, (1, dim))
        tcost_vec = copy.deepcopy(bid_ask)
        tcost_vec[0, 0] = tcost_vec[0, 0] + bid_ask_previous[0, 0]/2
        tcost_vec[0, dim-1] = bid_ask[0,dim-1]/2
        tcost_vec_test = np.reshape(tcost_vec, (dim,1)) 

        Y_test = np.concatenate((Y_test, return_vec_test, tcost_vec_test), axis=1)

        best_model, best_loss, best_accuracy, best_params, loss_per_fold_training_total, loss_per_fold_total, acc_per_fold_training_total, acc_per_fold_total = search.fit(X_train, Y_train, 
                                                                                                                                                                previous_training=previous_training,
                                                                                                                                                                previous=previous, previous_accuracy_training=previous_accuracy_training,
                                                                                                                                                                previous_accuracy=previous_accuracy, neural=True)
        
        print(best_params)
        previous = loss_per_fold_total
        previous_training = loss_per_fold_training_total
        previous_accuracy = acc_per_fold_total
        previous_accuracy_training = acc_per_fold_training_total
        best_loss_per_fold.append(best_loss)
        best_accuracy_per_fold.append(best_accuracy)
        nb_lags_best = best_params["nb_lags"]
        X_test_prime = np.concatenate((X_train[X_train.shape[0]-nb_lags_best:X_train.shape[0]], X_test), axis=0)
        X_test_prime = best_model.scale(X_test_prime)
        X_test_prime = best_model.reshape(X_test_prime, nb_lags_best)
        X_test_prime = X_test_prime[nb_lags_best:]
        pred_directions = best_model.predict(X_test_prime)
        predicted_directions.append(pred_directions)
        scores = best_model.evaluate(X_test_prime,best_model.prepare_targets(Y_test))
        acc_per_fold.append(scores)
        print(scores)
        fold_no = fold_no + 1
        end = time.time()
        print(end - start)
    print('------------------------------------------------------------------------')
    print('Average scores for all folds on validation sets:')
    print(f'> Accuracy: {np.mean(best_accuracy_per_fold*100)} (+- {np.std(best_accuracy_per_fold*100)})')
    print('------------------------------------------------------------------------')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold*100)} (+- {np.std(acc_per_fold*100)})')
    print('------------------------------------------------------------------------')

    # Back test
    predicted_directions = np.concatenate(predicted_directions).ravel()
    length = len(predicted_directions)
    history = len(X)
    y_test = data.get_direction_num().iloc[(history-length):]
    market_returns = data.get_return().iloc[(history-length):]
    bid_ask = data.get_close_bid_ask().iloc[(history-length):]
    bid_ask_previous = data.get_close_bid_ask_previous().iloc[(history-length):]
    currency = currencyReturn(y_test, market_returns, bid_ask, bid_ask_previous)
    currency.run(predicted_directions, t_cost=False)
    currency.plot_return("lstm_sentiments_daily_return", dash=True)
    stats = currency.output_summary_stats()
    print(stats)

    currency = currencyReturn(y_test, market_returns, bid_ask, bid_ask_previous)
    currency.run(predicted_directions, t_cost=True)
    currency.plot_return("lstm_sentiments_daily_cost_return", dash=True)
    stats = currency.output_summary_stats()
    print(stats)

    ##Hourly data
    #file where all the sentiment time series are stored
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/hourly_sentiment_series.txt"
    freq = 'H'
    print ('preprocessing...\n')
    data = d.data_sentiments(fname, freq)
    size = data.get_length()
    X = data.get_df()
    X = X.drop(['Direction', 'close_bid_ask', 'Return', 'close_bid_ask_previous', 'close'], axis=1)
    print('Drop McLoughran')
    X = X.drop(['assoc_fin_pos_positive', 'assoc_fin_pos_negative',
       'assoc_fin_pos_litigious', 'assoc_fin_pos_constraining',
       'assoc_fin_pos_uncertainty', 'assoc_fin_pos_strong_modal',
       'assoc_fin_pos_moderate_modal', 'assoc_fin_pos_weak_modal',
       'assoc_fin_positive', 'assoc_fin_negative', 'assoc_fin_litigious',
       'assoc_fin_constraining', 'assoc_fin_uncertainty',
       'assoc_fin_strong_modal', 'assoc_fin_moderate_modal',
       'assoc_fin_weak_modal'], axis=1)
    X = np.asarray(X)
    Y = data.get_direction_num()
    Y = np.asarray(Y.values)

    nb_lags = [5, 10]
    batch_sizes = [32,64]
    keep_probs=[[0.2,0.2,0.2], [0.4,0.4,0.4]] 
    kernel_regularizers=[[0,0], [0.001,0.001],[0.0001,0.0001]]
    learning_rates = [0.001]
    lstm_layers=[[64, 32], [128,64], [32,16]]#
    space = {"nb_lags": nb_lags, "batch_size": batch_sizes, "keep_prob":keep_probs, "kernel_regularizer":kernel_regularizers, 
             "learning_rate":learning_rates, "lstm_layers":lstm_layers, "epochs":[100]}
    # Test
    tscv_outer = ExpandingWindowSplit(test_size=500, train_size_init=5000)
    acc_per_fold = []
    fold_no = 1
    predicted_directions = []
    best_accuracy_per_fold = []
    best_loss_per_fold = []
    previous = None
    previous_training = None
    previous_accuracy = None
    previous_accuracy_training = None
    
    for train, test in tscv_outer.split(X):
        start = time.time()
        print("fold no: " + str(fold_no))
        reset_keras()
        model = lstmConfig()
        X_train = X[train]
        Y_train = Y[train]
        X_test = X[test]
        Y_test = Y[test]
        tscv_inner = ExpandingWindowSplit(test_size=500, train_size_init=len(train)-1500)
        search = GridSearchCV_custom(model, space, tscv_inner)
        return_vec = data.get_return().iloc[train].values.ravel()
        dim = return_vec.shape[0]
        return_vec_train = np.reshape(data.get_return().iloc[train].values.ravel(), (dim,1))
        
        bid_ask = data.get_close_bid_ask().iloc[train].values.ravel()
        bid_ask_previous = data.get_close_bid_ask_previous().iloc[train].values.ravel()
        bid_ask = np.reshape(bid_ask, (1, dim))
        bid_ask_previous = np.reshape(bid_ask_previous, (1, dim))
        tcost_vec = copy.deepcopy(bid_ask)
        tcost_vec[0, 0] = tcost_vec[0, 0] + bid_ask_previous[0, 0]/2
        tcost_vec[0, dim-1] = bid_ask[0,dim-1]/2
        tcost_vec_train = np.reshape(tcost_vec, (dim,1)) 
        Y_train = np.concatenate((Y_train, return_vec_train, tcost_vec_train), axis=1)

        return_vec = data.get_return().iloc[test].values.ravel()
        dim = return_vec.shape[0]
        return_vec_test = np.reshape(data.get_return().iloc[test].values.ravel(), (dim,1))
        
        bid_ask = data.get_close_bid_ask().iloc[test].values.ravel()
        bid_ask_previous = data.get_close_bid_ask_previous().iloc[test].values.ravel()
        bid_ask = np.reshape(bid_ask, (1, dim))
        bid_ask_previous = np.reshape(bid_ask_previous, (1, dim))
        tcost_vec = copy.deepcopy(bid_ask)
        tcost_vec[0, 0] = tcost_vec[0, 0] + bid_ask_previous[0, 0]/2
        tcost_vec[0, dim-1] = bid_ask[0,dim-1]/2
        tcost_vec_test = np.reshape(tcost_vec, (dim,1)) 

        Y_test = np.concatenate((Y_test, return_vec_test, tcost_vec_test), axis=1)

        best_model, best_loss, best_accuracy, best_params, loss_per_fold_training_total, loss_per_fold_total, acc_per_fold_training_total, acc_per_fold_total = search.fit(X_train, Y_train, 
                                                                                                                                                                previous_training=previous_training,
                                                                                                                                                                previous=previous, previous_accuracy_training=previous_accuracy_training,
                                                                                                                                                                previous_accuracy=previous_accuracy, neural=True)
        
        print(best_params)
        previous = loss_per_fold_total
        previous_training = loss_per_fold_training_total
        previous_accuracy = acc_per_fold_total
        previous_accuracy_training = acc_per_fold_training_total
        best_loss_per_fold.append(best_loss)
        best_accuracy_per_fold.append(best_accuracy)
        nb_lags_best = best_params["nb_lags"]
        X_test_prime = np.concatenate((X_train[X_train.shape[0]-nb_lags_best:X_train.shape[0]], X_test), axis=0)
        X_test_prime = best_model.scale(X_test_prime)
        X_test_prime = best_model.reshape(X_test_prime, nb_lags_best)
        X_test_prime = X_test_prime[nb_lags_best:]
        pred_directions = best_model.predict(X_test_prime)
        predicted_directions.append(pred_directions)
        scores = best_model.evaluate(X_test_prime,best_model.prepare_targets(Y_test))
        acc_per_fold.append(scores)
        print(scores)
        fold_no = fold_no + 1
        end = time.time()
        print(end - start)
    print('------------------------------------------------------------------------')
    print('Average scores for all folds on validation sets:')
    print(f'> Accuracy: {np.mean(best_accuracy_per_fold*100)} (+- {np.std(best_accuracy_per_fold*100)})')
    print('------------------------------------------------------------------------')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold*100)} (+- {np.std(acc_per_fold*100)})')
    print('------------------------------------------------------------------------')

    # Back test
    predicted_directions = np.concatenate(predicted_directions).ravel()
    length = len(predicted_directions)
    history = len(X)
    y_test = data.get_direction_num().iloc[(history-length):]
    market_returns = data.get_return().iloc[(history-length):]
    bid_ask = data.get_close_bid_ask().iloc[(history-length):]
    bid_ask_previous = data.get_close_bid_ask_previous().iloc[(history-length):]
    currency = currencyReturn(y_test, market_returns, bid_ask, bid_ask_previous)
    currency.run(predicted_directions, t_cost=False)
    currency.plot_return("lstm_sentiments_hourly_return", dash=True)
    stats = currency.output_summary_stats()
    print(stats)

    currency = currencyReturn(y_test, market_returns, bid_ask, bid_ask_previous)
    currency.run(predicted_directions, t_cost=True)
    currency.plot_return("lstm_sentiments_hourly_cost_return", dash=True)
    stats = currency.output_summary_stats()
    print(stats)

    end = time.time()
    print(end - start)


if __name__ == '__main__':
    main()