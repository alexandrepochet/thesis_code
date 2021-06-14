from initial_analysis.Arima import Arima
from initial_analysis.SVC import SVC
from utils.GridSearchCV import GridSearchCV
from utils.currencyReturn import currencyReturn
from sklearn.model_selection import KFold, TimeSeriesSplit
from data.data_currency import data_currency
from utils.ShiftingWindowSplit import ShiftingWindowSplit
from utils.ExpandingWindowSplit import ExpandingWindowSplit
from utils.utils import weighted_mean, weighted_std
import time
import pdb
import warnings
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt


def main(): 
     """
     Execute matching action for testing
     """
     warnings.filterwarnings("ignore")
     start = time.time()
     
     # Daily
     print('------------------------------------------------------------------------')
     print('Daily data Arima')
     print('------------------------------------------------------------------------')
     file = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/data_Arima_D.txt"
     freq = 'D'
     data = data_currency(file, freq)
     start = 1
     step = 1
     stop = 6
     orders = []
     for s_prime in range (start, stop, step):
          orders.append((0, 0, s_prime))
     trends = ["n","c"]
     lags = [1,2,3,4,5]
     space = {"order": orders, "trend": trends, "lags":lags}
     X = data.get_return()
     exog =  data.df.iloc[:,0:10]
     # Test
     tscv_outer = ExpandingWindowSplit(test_size=50, train_size_init=500)
     acc_per_fold = []
     fold_no = 1
     model = Arima()
     predicted_directions = []
     best_accuracy_per_fold = []
     weight = []
     length_tot = 0
     previous = None
     for train, test in tscv_outer.split(X.values):
          print("fold no: " + str(fold_no))
          tscv_inner = ExpandingWindowSplit(test_size=50, train_size_init=len(train)-150)
          search = GridSearchCV(model, space, tscv_inner)
          best_model, best_accuracy, best_params, acc_per_fold_total = search.fit(X.iloc[train], Exog_train=exog.iloc[train], previous=previous)
          previous = acc_per_fold_total
          best_accuracy_per_fold.append(best_accuracy)
          nb_lags_best = best_params["lags"]
          predicted_returns, pred_directions = best_model.predict(X.iloc[test], exog=exog.iloc[test,0:nb_lags_best])
          predicted_directions = np.concatenate((predicted_directions, pred_directions), axis=0)
          scores = best_model.evaluate(X.iloc[test], exog=exog.iloc[test,0:nb_lags_best])
          acc_per_fold.append(scores)
          length_tot += len(X.iloc[test])
          weight.append(len(X.iloc[test]))
          fold_no = fold_no + 1
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
     market_returns = X.iloc[(history-length):]
     bid_ask = data.get_close_bid_ask().iloc[(history-length):]
     bid_ask_previous = data.get_close_bid_ask_previous().iloc[(history-length):]
     currency = currencyReturn(y_test, market_returns, bid_ask, bid_ask_previous)
     currency.run(predicted_directions, t_cost=False)

     arima_curve = currency.eqCurves['Strategy']

     # Daily
     print('------------------------------------------------------------------------')
     print('Daily data SVC')
     print('------------------------------------------------------------------------')
     file = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/data_SVC_ST_10D.txt"

     freq = 'D'
     data_curr = data_currency(file, 'D') 
     size = data_curr.get_length()
     X = data_curr.get_df()
     X = X.drop(['Direction', 'close_bid_ask', 'Return', 'close_bid_ask_previous', 'close'], axis=1)
     X = np.asarray(X)
     Y = data_curr.get_direction_num()
     Y = np.asarray(Y.values)
     Cs = np.logspace(-2, 5, 8)
     gammas = np.logspace(-5, 3, 9)
     size = X.shape[1]
     end = math.ceil(size/10)*10
     space = {"C": Cs, "gamma": gammas}
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
         start = time.time()
         print("fold no: " + str(fold_no))
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
     y_test = data_curr.get_direction_num().iloc[(history-length):]
     market_returns = data_curr.get_return().iloc[(history-length):]
     bid_ask = data_curr.get_close_bid_ask().iloc[(history-length):]
     bid_ask_previous = data_curr.get_close_bid_ask_previous().iloc[(history-length):]
     currency = currencyReturn(y_test, market_returns, bid_ask, bid_ask_previous)
     currency.run(predicted_directions, t_cost=False)

     SVC_curve = currency.eqCurves['Strategy']
     df = pd.DataFrame(index=SVC_curve.index)
     df['SVC'] = SVC_curve
     df['ARMA'] = arima_curve
     df['Long-only'] = currency.eqCurves['Buy and Hold']
     df['Majority'] = 1-currency.eqCurves['Buy and Hold']+1
     df['SVC'].plot(figsize=(10, 8))
     df['ARMA'].plot()
     df['Long-only'].plot()
     df['Majority'].plot()
     plt.ylabel('Index value')
     plt.xlabel('Date')
     plt.legend()
     location = 'C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/src/figures/'
     plt.savefig(location + 'ARMA_SVC' + '.jpg', pad_inches=1)
     plt.close()
     pdb.set_trace()

     end = time.time()
     print(end - start)
     
if __name__ == '__main__':
    main()

