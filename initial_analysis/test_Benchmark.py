from utils.ShiftingWindowSplit import ShiftingWindowSplit
from utils.ExpandingWindowSplit import ExpandingWindowSplit
from data.data_currency import data_currency
from utils.currencyReturn import currencyReturn
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

    # Daily
    print('------------------------------------------------------------------------')
    print('Daily ')
    print('------------------------------------------------------------------------')
    file = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_D.txt"
    freq = 'D'
    data_curr = data_currency(file, 'D') 
    size = data_curr.get_length()
    X = data_curr.get_df()
    X = np.asarray(X)
    Y = data_curr.get_direction_num()
    Y = np.asarray(Y.values)
    # Test
    tscv_outer = ExpandingWindowSplit(test_size=50, train_size_init=500)
    acc_test = []
    fold_no = 1
    acc_valid = []
    iteration = 0
    length = 0
    for train, test in tscv_outer.split(X):
        print("fold no: " + str(fold_no))
        X_train = X[train]
        Y_train = Y[train]
        length = length + len(Y[test])
        tscv_inner = ExpandingWindowSplit(test_size=50, train_size_init=len(train)-150)
        target = Y[test]
        acc_test.append(len(target[target==0])/len(target)*100)
        i = 0
        for train_valid, valid in tscv_inner.split(X_train):
            i += 1
            if i==3 and iteration!=0:
                target_valid = Y_train[valid]
                acc_valid.append(len(target_valid[target_valid==0])/len(target_valid)*100)
            elif iteration==0:
                target_valid = Y_train[valid]
                acc_valid.append(len(target_valid[target_valid==0])/len(target_valid)*100)
            else:
                continue
        fold_no = fold_no + 1
        iteration = iteration + 1
        acc_test.append(len(target[target==0])/len(target)*100)
    print('------------------------------------------------------------------------')
    print('Average scores for all folds on validation sets:')
    print(f'> Accuracy: {np.mean(acc_valid)} (+- {np.std(acc_valid)})')
    print('------------------------------------------------------------------------')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_test)} (+- {np.std(acc_test)})')
    print('------------------------------------------------------------------------')

    # Back test
    history = len(X)
    print(history)
    y_test = data_curr.get_direction_num()
    market_returns = data_curr.get_return()
    fold_no = 1
    total_return_per_fold=[]
    for train, test in tscv_outer.split(X):
        print("fold no: " + str(fold_no))
        returns_fold = np.abs(market_returns.iloc[test])
        print(returns_fold.mean()[0])
        print(np.abs(market_returns.iloc[test].mean()[0]))
        print(np.abs(market_returns.iloc[test].std()[0]))
        fold_no = fold_no + 1
    y_test = data_curr.get_direction_num().iloc[(history-length):]
    market_returns = data_curr.get_return().iloc[(history-length):]
    predicted_directions = [0] * length
    predicted_directions = np.asarray(predicted_directions)
    bid_ask = data_curr.get_close_bid_ask().iloc[(history-length):]
    bid_ask_previous = data_curr.get_close_bid_ask_previous().iloc[(history-length):]
    currency = currencyReturn(y_test, market_returns, bid_ask, bid_ask_previous)
    currency.run(predicted_directions, t_cost=False)
    currency.plot_return("Majority_daily", dash=True)
    stats = currency.output_summary_stats()
    print(stats)

    # Hourly
    print('------------------------------------------------------------------------')
    print('Hourly ')
    print('------------------------------------------------------------------------')
    file = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_H.txt"
    freq = 'H'
    data_curr = data_currency(file, 'H') 
    size = data_curr.get_length()
    X = data_curr.get_df()
    X = np.asarray(X)
    Y = data_curr.get_direction_num()
    Y = np.asarray(Y.values)
    # Test
    tscv_outer = ExpandingWindowSplit(test_size=500, train_size_init=5000)
    acc_test = []
    fold_no = 1
    acc_valid = []
    iteration = 0
    length = 0
    for train, test in tscv_outer.split(X):
        print("fold no: " + str(fold_no))
        X_train = X[train]
        Y_train = Y[train]
        length = length + len(Y[test])
        tscv_inner = ExpandingWindowSplit(test_size=500, train_size_init=len(train)-1500)
        target = Y[test]
        acc_test.append(len(target[target==0])/len(target)*100)
        i = 0
        for train_valid, valid in tscv_inner.split(X_train):
            i += 1
            if (i==3 and iteration!=0) or iteration==0:
                target_valid = Y_train[valid]
                acc_valid.append(len(target_valid[target_valid==1])/len(target_valid)*100)
            elif iteration==0:
                target_valid = Y_train[valid]
                acc_valid.append(len(target_valid[target_valid==1])/len(target_valid)*100)
            else:
                continue
        fold_no = fold_no + 1
        iteration = iteration + 1
        acc_test.append(len(target[target==0])/len(target)*100)
    print('------------------------------------------------------------------------')
    print('Average scores for all folds on validation sets:')
    print(f'> Accuracy: {np.mean(acc_valid)} (+- {np.std(acc_valid)})')
    print('------------------------------------------------------------------------')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_test)} (+- {np.std(acc_test)})')
    print('------------------------------------------------------------------------')

    # Back test
    history = len(X)
    print(history)
    y_test = data_curr.get_direction_num()
    market_returns = data_curr.get_return()
    fold_no = 1
    total_return_per_fold=[]
    for train, test in tscv_outer.split(X):
        print("fold no: " + str(fold_no))
        returns_fold = np.abs(market_returns.iloc[test])
        print(returns_fold.mean()[0])
        print(np.abs(market_returns.iloc[test].mean()[0]))
        print(np.abs(market_returns.iloc[test].std()[0]))
        fold_no = fold_no + 1
    y_test = data_curr.get_direction_num().iloc[(history-length):]
    market_returns = data_curr.get_return().iloc[(history-length):]
    predicted_directions = [0] * length
    predicted_directions = np.asarray(predicted_directions)
    bid_ask = data_curr.get_close_bid_ask().iloc[(history-length):]
    bid_ask_previous = data_curr.get_close_bid_ask_previous().iloc[(history-length):]
    currency = currencyReturn(y_test, market_returns, bid_ask, bid_ask_previous)
    currency.run(predicted_directions, t_cost=False)
    currency.plot_return("Majority_hourly", dash=True)
    stats = currency.output_summary_stats()
    print(stats)

    end = time.time()
    print(end - start)
     
if __name__ == '__main__':
    main()



