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
import data.data_sentiments as d
from rnn.lstmConfig import lstmConfig
from utils.utils import reset_keras
from silence_tensorflow import silence_tensorflow


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
                                                                                                              previous_training=previous_training,
                                                                                                             previous=previous, neural=True)
        
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
     predicted_directions = [item[0] for sublist in predicted_directions for item in sublist]
     length = len(predicted_directions)
     predicted_directions= np.array(predicted_directions)
     history = len(X)
     y_test = data.get_direction_num().iloc[(history-length):]
     market_returns = data.get_return().iloc[(history-length):]
     bid_ask = data.get_close_bid_ask().iloc[(history-length):]
     bid_ask_previous = data.get_close_bid_ask_previous().iloc[(history-length):]
     currency = currencyReturn(y_test, market_returns, bid_ask, bid_ask_previous)
     currency.run(predicted_directions, t_cost=False)

     RNN_LOCF_curve = currency.eqCurves['Strategy']
     df = pd.DataFrame(index=RNN_LOCF_curve.index)
     df['RNN_SA_LOCF'] = RNN_LOCF_curve
     df['ARMA'] = arima_curve
     df['Long-only'] = currency.eqCurves['Buy and Hold']
     df['Majority'] = 1-currency.eqCurves['Buy and Hold']+1
     df['RNN_SA_LOCF'].plot(figsize=(10, 8))
     df['ARMA'].plot()
     df['Long-only'].plot()
     #df['Majority'].plot()
     plt.ylabel('Index value')
     plt.xlabel('Date')
     plt.legend()
     location = 'C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/src/figures/'
     plt.savefig(location + 'ARMA_RNN_LOCF' + '.jpg', pad_inches=1)
     plt.close()
     pdb.set_trace()

     end = time.time()
     print(end - start)
     
if __name__ == '__main__':
    main()

