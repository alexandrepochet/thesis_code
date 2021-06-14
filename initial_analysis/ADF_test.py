import pandas as pd
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import TimeSeriesSplit
import data.data_tweets as d
import time
import pdb
import numpy as np


def adf_test(timeseries):
    #Perform Dickey-Fuller test:
    #print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    limit = dftest[4]['1%']
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    if dftest[0]<limit:
        print(True)
    else:
        print(False)
    #print (dfoutput)


def main(): 
    """
    Execute matching action for testing
    """
    start = time.time()
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_sentiment_D.txt"
    freq = 'D'
    print ('preprocessing...\n')
    data = d.data_tweets(fname, freq)
    size = data.get_length() 
    returns_daily = data.get_return()
    tscv_outer = TimeSeriesSplit(n_splits=3)
    tscv_inner = TimeSeriesSplit(n_splits=2)
    for train, test in tscv_outer.split(returns_daily):
        X_train_outer = returns_daily.values.ravel()[train]
        adf_test(X_train_outer)
        for train_inner, val in tscv_inner.split(X_train_outer):
            adf_test(X_train_outer[train_inner])

    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_sentiment_H.txt"
    freq = 'H'
    print ('preprocessing...\n')
    data = d.data_tweets(fname, freq)
    size = data.get_length() 
    returns_daily = data.get_return()
    tscv_outer = TimeSeriesSplit(n_splits=10)
    tscv_inner = TimeSeriesSplit(n_splits=5)
    for train, test in tscv_outer.split(returns_daily):
        X_train_outer = returns_daily.values.ravel()[train]
        adf_test(X_train_outer)
        for train_inner, val in tscv_inner.split(X_train_outer):
            adf_test(X_train_outer[train_inner])

    end = time.time()
    print(end - start)
    
if __name__ == '__main__':
    main()
