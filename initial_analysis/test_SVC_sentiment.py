from initial_analysis.SVC import SVC
from utils.GridSearchCV import GridSearchCV
from utils.currencyReturn import currencyReturn
from sklearn.model_selection import KFold, TimeSeriesSplit
import data.data_sentiments as d
from utils.ShiftingWindowSplit import ShiftingWindowSplit
from utils.ExpandingWindowSplit import ExpandingWindowSplit
from utils.utils import add_lags
from utils.utils import weighted_mean, weighted_std
import time
import pdb
import pandas as pd
import warnings
import numpy as np
import math


def main(): 
    """
    Execute matching action for testing
    """
    warnings.filterwarnings("ignore")
    start = time.time()
    
    ##Daily data
    #file where all the sentiment time series are stored
    print('------------------------------------------------------------------------')
    print('Daily data ')
    print('------------------------------------------------------------------------')
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/daily_sentiment_series.txt"
    freq = 'D'
    print ('preprocessing...\n')
    data = d.data_sentiments(fname, freq)
    size = data.get_length()
    X = data.get_df()
    X = X.drop(['Direction', 'close_bid_ask', 'Return', 'close_bid_ask_previous', 'close'], axis=1)
    Y = data.get_direction_num()
    Y = np.asarray(Y.values)
    Cs = np.logspace(-2, 5, 8)
    gammas = np.logspace(-5, 3, 9)
    lags = [5,10]
    size = X.shape[1]
    end = math.ceil(size/10)*10
    space = {"C": Cs, "gamma": gammas, "lags": lags}
    # Test
    tscv_outer = ExpandingWindowSplit(test_size=50, train_size_init=500)
    acc_per_fold = []
    fold_no = 1
    model = SVC()
    predicted_directions = []
    best_accuracy_per_fold = []
    weight = []
    length_tot = 0
    previous = None
    for train, test in tscv_outer.split(X):
        X_train = X.iloc[train]
        X_test = X.iloc[test]
        start = time.time()
        print("fold no: " + str(fold_no))
        tscv_inner = ExpandingWindowSplit(test_size=50, train_size_init=len(train)-150)
        search = GridSearchCV(model, space, tscv_inner)
        Y_train, Y_test = model.prepare_targets(Y[train], Y[test])
        best_model, best_accuracy, best_params, acc_per_fold_total = search.fit(X_train, Y_train, previous=previous, SVC_sentiment=True)
        previous = acc_per_fold_total
        best_accuracy_per_fold.append(best_accuracy)
        nb_lags_best = best_params["lags"]
        X_test = pd.concat([X_train.iloc[len(X_train)-nb_lags_best:len(X_train)], X_test])
        X_test = add_lags(X_test, nb_lags_best, True)
        X_test = X_test.iloc[nb_lags_best:]
        X_test = np.asarray(X_test)
        pred_directions = best_model.predict(X_test)
        predicted_directions = np.concatenate((predicted_directions, pred_directions), axis=0)
        scores = best_model.evaluate(X_test, Y_test)
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
    length = len(predicted_directions)
    history = len(X)
    y_test = data.get_direction_num().iloc[(history-length):]
    market_returns = data.get_return().iloc[(history-length):]
    bid_ask = data.get_close_bid_ask().iloc[(history-length):]
    bid_ask_previous = data.get_close_bid_ask_previous().iloc[(history-length):]
    currency = currencyReturn(y_test, market_returns, bid_ask, bid_ask_previous)
    currency.run(predicted_directions, t_cost=False)
    currency.plot_return("SVC_sentiment_daily", dash=True)
    stats = currency.output_summary_stats()
    print(stats)

    currency = currencyReturn(y_test, market_returns, bid_ask, bid_ask_previous)
    currency.run(predicted_directions, t_cost=True)
    currency.plot_return("SVC_sentiment_daily_cost", dash=True)
    stats = currency.output_summary_stats()
    print(stats)

    ##Hourly data
    #file where all the sentiment time series are stored
    print('------------------------------------------------------------------------')
    print('Hourly data ')
    print('------------------------------------------------------------------------')
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/hourly_sentiment_series.txt"
    freq = 'H'
    print ('preprocessing...\n')
    data = d.data_sentiments(fname, freq)
    size = data.get_length()
    X = data.get_df()
    X = X.drop(['Direction', 'close_bid_ask', 'Return', 'close_bid_ask_previous', 'close'], axis=1)
    Y = data.get_direction_num()
    Y = np.asarray(Y.values)
    Cs = np.logspace(-2, 5, 8)
    gammas = np.logspace(-5, 3, 9)
    lags = [5,10]
    size = X.shape[1]
    end = math.ceil(size/10)*10
    space = {"C": Cs, "gamma": gammas, "lags": lags}
    # Test
    tscv_outer = ExpandingWindowSplit(test_size=500, train_size_init=5000)
    acc_per_fold = []
    fold_no = 1
    model = SVC()
    predicted_directions = []
    best_accuracy_per_fold = []
    weight = []
    length_tot = 0
    previous = None
    for train, test in tscv_outer.split(X):
        X_train = X.iloc[train]
        X_test = X.iloc[test]
        start = time.time()
        print("fold no: " + str(fold_no))
        tscv_inner = ExpandingWindowSplit(test_size=500, train_size_init=len(train)-1500)
        search = GridSearchCV(model, space, tscv_inner)
        Y_train, Y_test = model.prepare_targets(Y[train], Y[test])
        best_model, best_accuracy, best_params, acc_per_fold_total = search.fit(X_train, Y_train, previous=previous, SVC_sentiment=True)
        previous = acc_per_fold_total
        best_accuracy_per_fold.append(best_accuracy)
        nb_lags_best = best_params["lags"]
        X_test = pd.concat([X_train.iloc[len(X_train)-nb_lags_best:len(X_train)], X_test])
        X_test = add_lags(X_test, nb_lags_best, True)
        X_test = X_test.iloc[nb_lags_best:]
        X_test = np.asarray(X_test)
        pred_directions = best_model.predict(X_test)
        predicted_directions = np.concatenate((predicted_directions, pred_directions), axis=0)
        scores = best_model.evaluate(X_test, Y_test)
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
    length = len(predicted_directions)
    history = len(X)
    y_test = data.get_direction_num().iloc[(history-length):]
    market_returns = data.get_return().iloc[(history-length):]
    bid_ask = data.get_close_bid_ask().iloc[(history-length):]
    bid_ask_previous = data.get_close_bid_ask_previous().iloc[(history-length):]
    currency = currencyReturn(y_test, market_returns, bid_ask, bid_ask_previous)
    currency.run(predicted_directions, t_cost=False)
    currency.plot_return("SVC_sentiment_hourly", dash=True)
    stats = currency.output_summary_stats()
    print(stats)

    currency = currencyReturn(y_test, market_returns, bid_ask, bid_ask_previous)
    currency.run(predicted_directions, t_cost=True)
    currency.plot_return("SVC_sentiment_hourly_cost", dash=True)
    stats = currency.output_summary_stats()
    print(stats)

    end = time.time()
    print(end - start)
     
if __name__ == '__main__':
    main()



