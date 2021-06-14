import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import pdb
import warnings
from utils.currencyReturn import currencyReturn
import data.data_sentiments as d


def main(): 
     """
     Execute matching action for testing
     """
     warnings.filterwarnings("ignore")
     start = time.time()
     location = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/src/output/"

     arima_daily = np.loadtxt(location + str('Arima_daily.txt'))
     arima_hourly = np.loadtxt(location + str('Arima_hourly.txt'))
     svc_daily = np.loadtxt(location + str('SVC_daily.txt'))
     svc_hourly = np.loadtxt(location + str('SVC_hourly.txt'))
     miningsvc_daily = np.loadtxt(location + str('MiningSVC_daily.txt'))
     miningsvc_hourly = np.loadtxt(location + str('MiningSVC_hourly.txt'))
     lstm_LOCF_daily = np.loadtxt(location + str('lstm_daily_LOCF.txt'))
     lstm_NA_daily = np.loadtxt(location + str('lstm_daily_NA.txt'))
     lstm_LOCF_hourly = np.loadtxt(location + str('lstm_hourly_LOCF.txt'))
     lstm_NA_hourly = np.loadtxt(location + str('lstm_hourly_NA.txt'))
     length_daily = len(arima_daily)
     length_hourly = len(arima_hourly)


     #daily
     fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/daily_sentiment_series.txt"
     freq = 'D'
     print ('preprocessing...\n')
     data = d.data_sentiments(fname, freq)
     size = data.get_length()
     X = data.get_df()
     X = np.asarray(X)

     length = len(arima_daily)
     history = len(X)
     y_test = data.get_direction_num().iloc[(history-length):]
     market_returns = data.get_return().iloc[(history-length):]
     bid_ask = data.get_close_bid_ask().iloc[(history-length):]
     bid_ask_previous = data.get_close_bid_ask_previous().iloc[(history-length):]
     currency = currencyReturn(y_test, market_returns, bid_ask, bid_ask_previous)
     returns_arima = pd.DataFrame({'Returns':currency.run(arima_daily, t_cost=False)['Strategy'][1:].values, 'Model':["ARMA" for x in range(length)]})
     returns_svc = pd.DataFrame({'Returns':currency.run(svc_daily, t_cost=False)['Strategy'][1:].values, 'Model':["SVC_TA" for x in range(length)]})
     returns_miningsvc = pd.DataFrame({'Returns':currency.run(miningsvc_daily, t_cost=False)['Strategy'][1:].values, 'Model':["SVM_TM" for x in range(length)]})
     returns_lstm_LOCF = pd.DataFrame({'Returns':currency.run(lstm_LOCF_daily, t_cost=False)['Strategy'][1:].values, 'Model':["RNN_SA_LOCF" for x in range(length)]})
     returns_lstm_NA = pd.DataFrame({'Returns':currency.run(lstm_NA_daily, t_cost=False)['Strategy'][1:].values, 'Model':["RNN_SA_NA" for x in range(length)]})

     df_daily = [returns_arima, returns_svc, returns_miningsvc, returns_lstm_LOCF, returns_lstm_NA]
     df_daily = pd.concat(df_daily)
     ax_daily = sns.boxplot(x="Model", y="Returns", data=df_daily)
     plt.xticks(rotation=45, horizontalalignment='right', fontweight='light', fontsize='x-large')
     plt.tight_layout()
     location = 'C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/src/figures/'
     plt.savefig(location + 'Returns_daily' + '.jpg', pad_inches=1)
        

if __name__ == '__main__':
    main()