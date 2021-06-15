from initial_analysis.MiningSVC import MiningSVC
from utils.GridSearchCV import GridSearchCV
from utils.currencyReturn import currencyReturn
from sklearn.model_selection import KFold, TimeSeriesSplit
from data.data_tweets import data_tweets
from utils.ShiftingWindowSplit import ShiftingWindowSplit
from utils.ExpandingWindowSplit import ExpandingWindowSplit
from utils.utils import weighted_mean, weighted_std
import time
import pdb
import pandas as pd
import warnings
import numpy as np
import ast


def main(): 
     """
     Execute matching action for testing
     """
     warnings.filterwarnings("ignore")
     start = time.time()
    
     # Daily
     print('------------------------------------------------------------------------')
     print('Daily data ')
     print('------------------------------------------------------------------------')
     file = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_D.txt"
     freq = 'D'
     data = data_tweets(file, freq)

     X = data.get_text()
     X = np.asarray(X)
     Y = data.get_direction_num()
     Y = np.asarray(Y.values)
     n_features = [500, 2000, 5000]
     Cs = np.logspace(-2, 5, 8)
     ngram_ranges=[(1, 3)]
     weightings = ["Tf-idf"]
     space = {"weighting": weightings, "n_features": n_features, "ngram_range": ngram_ranges, "C": Cs}
     # Test
     tscv_outer = ExpandingWindowSplit(test_size=50, train_size_init=500)
     acc_per_fold = []
     fold_no = 1
     model = MiningSVC()
     predicted_directions = []
     best_accuracy_per_fold = []
     weight = []
     length_tot = 0
     previous = None
     for train, test in tscv_outer.split(X):
          tscv_inner = ExpandingWindowSplit(test_size=50, train_size_init=len(train)-150)
          search = GridSearchCV(model, space, tscv_inner)
          Y_train, Y_test = model.prepare_targets(Y[train], Y[test])
          best_model, best_accuracy, best_params, acc_per_fold_total = search.fit(X[train], Y_train, previous=previous)
          previous = acc_per_fold_total
          best_accuracy_per_fold.append(best_accuracy)
          pred_directions = best_model.predict(X[test])
          predicted_directions = np.concatenate((predicted_directions, pred_directions), axis=0)
          scores = best_model.evaluate(X[test], Y_test)
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
     currency.plot_return("MiningSVC_daily", dash=True)
     stats = currency.output_summary_stats()
     print(stats)
     
     currency = currencyReturn(y_test, market_returns, bid_ask, bid_ask_previous)
     currency.run(predicted_directions, t_cost=True)
     currency.plot_return("MiningSVC_daily_cost", dash=True)
     stats = currency.output_summary_stats()
     print(stats)

     # Hourly
     print('------------------------------------------------------------------------')
     print('Hourly data ')
     print('------------------------------------------------------------------------')
     file = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_H.txt"
     freq = 'H'
     data = data_tweets(file, freq)

     X = data.get_text()
     X = np.asarray(X)
     Y = data.get_direction_num()
     Y = np.asarray(Y.values)
     n_features = [500, 2000, 5000]
     Cs = np.logspace(-2, 5, 8)
     ngram_ranges=[(1, 3)]
     weightings = ["Tf-idf"]
     space = {"weighting": weightings, "n_features": n_features, "ngram_range": ngram_ranges, "C": Cs}
     # Test
     tscv_outer = ExpandingWindowSplit(test_size=500, train_size_init=5000)
     acc_per_fold = []
     fold_no = 1
     model = MiningSVC()
     predicted_directions = []
     best_accuracy_per_fold = []
     weight = []
     length_tot = 0
     previous=None
     for train, test in tscv_outer.split(X):
          print('fold number: ' + str(fold_no))
          tscv_inner = ExpandingWindowSplit(test_size=500, train_size_init=len(train)-1500)
          search = GridSearchCV(model, space, tscv_inner)
          Y_train, Y_test = model.prepare_targets(Y[train], Y[test])
          best_model, best_accuracy, best_params, acc_per_fold_total = search.fit(X[train], Y_train, previous=previous)
          previous = acc_per_fold_total
          best_accuracy_per_fold.append(best_accuracy)
          pred_directions = best_model.predict(X[test])
          predicted_directions = np.concatenate((predicted_directions, pred_directions), axis=0)
          scores = best_model.evaluate(X[test], Y_test)
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
     print('Average scores for all folds:')
     print(f'> Accuracy: {weighted_mean(weight, acc_per_fold*100)} (+- {weighted_std(weight, acc_per_fold*100)})')
     print('------------------------------------------------------------------------')
     # Back test

     length = len(predicted_directions)
     history = len(X)
     y_test = data.get_direction_num().iloc[(history-length):]
     market_returns = data.get_return().iloc[(history-length):]
     bid_ask = data.get_close_bid_ask().iloc[(history-length):]
     bid_ask_previous = data.get_close_bid_ask().iloc[(history-length):]#data.get_close_bid_ask_previous().iloc[(history-length):]
     currency = currencyReturn(y_test, market_returns, bid_ask, bid_ask_previous)
     currency.run(predicted_directions, t_cost=False)
     currency.plot_return("MiningSVC_hourly", dash=True)
     stats = currency.output_summary_stats()
     print(stats)

     currency = currencyReturn(y_test, market_returns, bid_ask, bid_ask_previous)
     currency.run(predicted_directions, t_cost=True)
     currency.plot_return("MiningSVC_hourly_cost", dash=True)
     stats = currency.output_summary_stats()
     print(stats)

     end = time.time()
     print(end - start)
     
if __name__ == '__main__':
    main()
