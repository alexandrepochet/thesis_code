# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 20:42:50 2019

@author: alexa
"""

import TSA as t
import time
import currency_preprocess as c
import pdb


def main():
    
     """
     
     Execute matching action for testing
     
     """

     start = time.time()
     fname_ask = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/forexDATA/eurusd_ask_full.csv"
     fname_bid = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/forexDATA/eurusd_bid_full.csv"

     data = c.Currency(fname_ask = fname_ask, fname_bid = fname_bid)
     
     # Daily
     print("resampling...\n")
     data.resample('D')
     threshold = 0.0000
     data.define_threshold(threshold)
     # Separate in train and validation set and test set
     size = len(data.df_resampled)
     print(size)

     data.df_resampled = data.df_resampled.iloc[0:int(0.8*size)]
     model = t.TSA(data.df_resampled, threshold)
     print ('training the arx model...\n')
     train =[50, 100, 250, 500, 1000, 2000]
     test =[1/8, 1/4, 1/2, 1]
     results_D_Arx = model.run(train, test, True, False)
     
     # Daily Arima
     print ('training the arima model...\n')
     train =[50, 100, 250, 500, 1000, 2000]
     test =[1/8, 1/4, 1/2, 1]
     results_D_Arima = model.run(train, test, False, False)
     
     # Hourly ARX
     print("resampling...\n")
     data.resample('H')
     threshold = 0.0000
     data.define_threshold(threshold)
     # Separate in train and validation set and test set
     size = len(data.df_resampled)
     data.df_resampled = data.df_resampled.iloc[0:int(0.8*size)]
     model = t.TSA(data.df_resampled, threshold)
     print ('training the arx model...\n')
     train =[250, 1000, 5000, 10000, 20000]
     test =[1/8, 1/4, 1/2, 1]
     results_H_Arx = model.run(train, test, True, False)
     
     # Hourly Arima
     print ('training the arima model...\n')
     train =[250, 1000, 5000, 10000, 20000]
     test =[1/8, 1/4, 1/2, 1]
     results_H_Arima = model.run(train, test, False, False)
     
     #printing
     print ('Results daily ARX...\n')
     print (results_D_Arx)
     print ('Results daily Arima...\n')
     print (results_D_Arima)
     print ('Results hourly ARX...\n')
     print (results_H_Arx)
     print ('Results hourly Arima...\n')
     print (results_H_Arima)     
     
     # test based on best training-testing windows
     data = c.Currency(fname_ask = fname_ask, fname_bid = fname_bid)
     # Daily
     print("resampling...\n")
     data.resample('D')
     threshold = 0.0000
     data.define_threshold(threshold)
     # Separate in train and validation set and test set
     size = len(data.df_resampled)
     print(size)
     data.df_resampled = data.df_resampled.iloc[int(0.8*size - 100):]
     model = t.TSA(data.df_resampled, threshold)
         
     print ('training the arx model...\n')
     train = 100
     test = 25
     results = model.time_series_analysis(train, test, True, False)
     model.plot_return(title = 'Daily_ARX')
     print('results daily ARX ' + str(results))
     
     # Daily
     data = c.Currency(fname_ask = fname_ask, fname_bid = fname_bid)
     print("resampling...\n")
     data.resample('D')
     threshold = 0.0000
     data.define_threshold(threshold)
     # Separate in train and validation set and test set
     size = len(data.df_resampled)
     
     # test based on best training-testing windows
     data.df_resampled = data.df_resampled.iloc[int(0.8*size - 500):]
     model = t.TSA(data.df_resampled, threshold)

     print ('training the Arima model...\n')
     train = 500
     test = 500
     results = model.time_series_analysis(train, test, False, False)
     model.plot_return(title = 'Daily_Arima')
     print('results daily Arima ' + str(results))
     
     # Hourly
     data = c.Currency(fname_ask = fname_ask, fname_bid = fname_bid)
     print("resampling...\n")
     data.resample('H')
     threshold = 0.0000
     data.define_threshold(threshold)
     # Separate in train and validation set and test set
     size = len(data.df_resampled)
     
     # test based on best training-testing windows
     data.df_resampled = data.df_resampled.iloc[int(0.8*size - 20000):]
     model = t.TSA(data.df_resampled, threshold)
     
     print ('training the arx model...\n')
     train = 20000
     test = 20000
     results = model.time_series_analysis(train, test, True, False)
     model.plot_return('Hourly_ARX', True)
     print('results hourly ARX ' + str(results))
     
     print ('training the Arima model...\n')
     train = 20000
     test = 20000
     results = model.time_series_analysis(train, test, False, False)
     model.plot_return('Hourly_Arima', True)
     print('results hourly Arima ' + str(results))
     

     end = time.time()
     print(end - start)
     
     
if __name__ == '__main__':
    main()   