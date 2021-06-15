from utils.utils import reset_keras
import pandas as pd
import numpy as np
from data.data_currency import data_currency
import data.data_sentiments as d
from rnn.lstmConfig import lstmConfig
from utils.GridSearchCV import GridSearchCV
from utils.ShiftingWindowSplit import ShiftingWindowSplit
from utils.ExpandingWindowSplit import ExpandingWindowSplit
from utils.currencyReturn import currencyReturn
from silence_tensorflow import silence_tensorflow
from utils.utils import weighted_mean, weighted_std
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
           
    # Daily
    print('------------------------------------------------------------------------')
    print('Daily data ')
    print('------------------------------------------------------------------------')
    file = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/data_SVC_STD.txt"
    
    freq = 'D'
    data_curr = data_currency(file, 'D') 
    size = data_curr.get_length()
    X = data_curr.get_df()
    X = X.drop(['Direction', 'close_bid_ask', 'Return', 'close_bid_ask_previous', 'close'], axis=1)
    
    Y = data_curr.get_direction_num()
    Y = np.asarray(Y.values)

    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/daily_sentiment_series.txt"
    print ('preprocessing...\n')
    data = d.data_sentiments(fname, freq)
    size_sentiments = data.get_length()
    sentiments = data.get_df()
    sentiments = sentiments.drop(['close', 'close_bid_ask', 'close_bid_ask_previous', 'Return',
                                  'Direction', 'assoc_fin_pos_positive', 'assoc_fin_pos_negative',
                                  'assoc_fin_pos_litigious', 'assoc_fin_pos_constraining',
                                  'assoc_fin_pos_uncertainty', 'assoc_fin_pos_strong_modal',
                                  'assoc_fin_pos_moderate_modal', 'assoc_fin_pos_weak_modal',
                                  'assoc_fin_positive', 'assoc_fin_negative', 'assoc_fin_litigious',
                                  'assoc_fin_constraining', 'assoc_fin_uncertainty',
                                  'assoc_fin_strong_modal', 'assoc_fin_moderate_modal',
                                  'assoc_fin_weak_modal'], axis=1)
    X = pd.concat([X, sentiments], axis=1, join="inner")
    X = np.asarray(X)
    nb_lags = [5, 10]
    batch_sizes = [32,64]
    keep_probs=[[0.2,0.2,0.2], [0.4,0.4,0.4]] 
    kernel_regularizers=[[0,0], [0.001,0.001],[0.0001,0.0001]]
    learning_rates = [0.001]
    lstm_layers=[[64, 32], [16,8], [32,16]]#
    space = {"nb_lags": nb_lags, "batch_size": batch_sizes, "keep_prob":keep_probs, "kernel_regularizer":kernel_regularizers, 
             "learning_rate":learning_rates, "lstm_layers":lstm_layers, "epochs":[100]}
    # Test
    tscv_outer = ExpandingWindowSplit(test_size=50, train_size_init=500)
    acc_per_fold = []
    fold_no = 1
    predicted_directions = []
    weight = []
    length_tot = 0
    best_accuracy_per_fold = []
    weight = []
    length_tot = 0
    previous = None
    previous_training = None
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
        search = GridSearchCV(model, space, tscv_inner)
        best_model, best_accuracy, best_params, acc_per_fold_training_total, acc_per_fold_total = search.fit(X[train], Y[train], 
                                                                                                             previous_training=previous_training, previous=previous, neural=True)
        
        print(best_params)
        previous = acc_per_fold_total
        previous_training = acc_per_fold_training_total
        best_accuracy_per_fold.append(best_accuracy)
        nb_lags_best = best_params["nb_lags"]
        X_test_prime = np.concatenate((X_train[X_train.shape[0]-nb_lags_best:X_train.shape[0]], X_test), axis=0)
        X_test_prime = best_model.scale(X_test_prime)
        X_test_prime = best_model.reshape(X_test_prime, nb_lags_best)
        X_test_prime = X_test_prime[nb_lags_best:]
        pred_directions = best_model.predict(X_test_prime)
        predicted_directions.append(pred_directions)
        scores = best_model.evaluate(X_test_prime,best_model.prepare_targets(Y_test))
        length_tot += len(Y_test)
        weight.append(len(Y_test))
        acc_per_fold.append(scores)
        print(scores)
        fold_no = fold_no + 1
        end = time.time()
        print(end - start)
    weight = [number / length_tot for number in weight]
    print('------------------------------------------------------------------------')
    print('Average scores for all folds on validation sets:')
    print(f'> Accuracy: {np.mean(best_accuracy_per_fold*100)} (+- {np.std(best_accuracy_per_fold*100)})')
    print('------------------------------------------------------------------------')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {weighted_mean(weight, acc_per_fold*100)} (+- {weighted_std(weight, acc_per_fold*100)})')
    print('------------------------------------------------------------------------')

    # Back test
    predicted_directions = np.concatenate(predicted_directions).ravel()
    length = len(predicted_directions)
    history = len(X)
    y_test = data_curr.get_direction_num().iloc[(history-length):]
    market_returns = data_curr.get_return().iloc[(history-length):]
    bid_ask = data_curr.get_close_bid_ask().iloc[(history-length):]
    bid_ask_previous = data_curr.get_close_bid_ask_previous().iloc[(history-length):]
    currency = currencyReturn(y_test, market_returns, bid_ask, bid_ask_previous)
    currency.run(predicted_directions, t_cost=False)
    currency.plot_return("lstm_sentiments_daily_TA3", dash=True)
    stats = currency.output_summary_stats()
    print(stats)
    currency = currencyReturn(y_test, market_returns, bid_ask, bid_ask_previous)
    currency.run(predicted_directions, t_cost=True)
    currency.plot_return("lstm_sentiments_daily_TA_cost3", dash=True)
    stats = currency.output_summary_stats()
    print(stats)

    # Hourly
    print('------------------------------------------------------------------------')
    print('Hourly data ')
    print('------------------------------------------------------------------------')
    file = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/data_SVC_STH.txt"
    
    freq = 'H'
    data_curr = data_currency(file, 'H') 
    size = data_curr.get_length()
    X = data_curr.get_df()
    X = X.drop(['Direction', 'close_bid_ask', 'Return', 'close_bid_ask_previous', 'close'], axis=1)
    X = np.asarray(X)
    Y = data_curr.get_direction_num()
    Y = np.asarray(Y.values)

    nb_lags = [5, 10]
    batch_sizes = [32,64]
    keep_probs=[[0.2,0.2,0.2], [0.4,0.4,0.4]] 
    kernel_regularizers=[[0,0], [0,0.001],[0,0.0001]]#[[0,0], [0.001,0.001],[0.0001,0.0001]]
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
    weight = []
    length_tot = 0
    previous = None
    previous_training = None
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
        search = GridSearchCV(model, space, tscv_inner)
        best_model, best_accuracy, best_params, acc_per_fold_training_total, acc_per_fold_total = search.fit(X[train], Y[train], 
                                                                                                             previous_training=previous_training, previous=previous, neural=True)
        
        print(best_params)
        previous = acc_per_fold_total
        previous_training = acc_per_fold_training_total
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
        length_tot += len(Y_test)
        weight.append(len(Y_test))
        fold_no = fold_no + 1
        end = time.time()
        print(end - start)
    weight = [number / length_tot for number in weight]
    print('------------------------------------------------------------------------')
    print('Average scores for all folds on validation sets:')
    print(f'> Accuracy: {np.mean(best_accuracy_per_fold*100)} (+- {np.std(best_accuracy_per_fold*100)})')
    print('------------------------------------------------------------------------')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {weighted_mean(weight, acc_per_fold*100)} (+- {weighted_std(weight, acc_per_fold*100)})')
    print('------------------------------------------------------------------------')

    # Back test
    predicted_directions = np.concatenate(predicted_directions).ravel()
    length = len(predicted_directions)
    history = len(X)
    y_test = data_curr.get_direction_num().iloc[(history-length):]
    market_returns = data_curr.get_return().iloc[(history-length):]
    bid_ask = data_curr.get_close_bid_ask().iloc[(history-length):]
    bid_ask_previous = data_curr.get_close_bid_ask_previous().iloc[(history-length):]
    currency = currencyReturn(y_test, market_returns, bid_ask, bid_ask_previous)
    currency.run(predicted_directions, t_cost=False)
    currency.plot_return("lstm_sentiments_hourly_TA", dash=True)
    stats = currency.output_summary_stats()
    print(stats)
    currency = currencyReturn(y_test, market_returns, bid_ask, bid_ask_previous)
    currency.run(predicted_directions, t_cost=True)
    currency.plot_return("lstm_sentiments_hourly_TA_cost", dash=True)
    stats = currency.output_summary_stats()
    print(stats)

    end = time.time()
    print(end - start)


if __name__ == '__main__':
    main()