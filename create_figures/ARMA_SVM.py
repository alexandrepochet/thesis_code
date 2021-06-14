import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import pdb
import warnings
from utils.currencyReturn import currencyReturn
from data.data_currency import data_currency
import math



def main(): 
     """
     Execute matching action for testing
     """
     warnings.filterwarnings("ignore")
     start = time.time()

     # Hourly
     print('------------------------------------------------------------------------')
     print('Hourly data ')
     print('------------------------------------------------------------------------')
     file = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/data_SVC_ST_10H.txt"
     freq = 'H'
     data_curr = data_currency(file, 'H') 
     size = data_curr.get_length()
     X = data_curr.get_df()
     X = X.drop(['Direction', 'close_bid_ask', 'Return', 'close_bid_ask_previous', 'close'], axis=1)
     X = np.asarray(X)
     Y = data_curr.get_direction_num()
     Y = np.asarray(Y.values)

     location = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/src/output/"

     arima_hourly = np.loadtxt(location + str('Arima_hourly.txt'))
     svc_hourly = np.loadtxt(location + str('SVC_hourly.txt'))
     
     # Back test
     length = len(svc_hourly)
     history = len(X)
     y_test = data_curr.get_direction_num().iloc[(history-length):]
     market_returns = data_curr.get_return().iloc[(history-length):]
     bid_ask = data_curr.get_close_bid_ask().iloc[(history-length):]
     bid_ask_previous = data_curr.get_close_bid_ask_previous().iloc[(history-length):]
     currency = currencyReturn(y_test, market_returns, bid_ask, bid_ask_previous)
     currency.run(svc_hourly, t_cost=False)
     currency.plot_return("SVC_hourly2", dash=True)
     stats = currency.output_summary_stats()
     print(stats)
     pdb.set_trace()     

if __name__ == '__main__':
    main()