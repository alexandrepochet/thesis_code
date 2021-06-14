import pandas as pd
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import TimeSeriesSplit
import data.data_tweets as d
import time
import pdb
import numpy as np


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
    print(size)
    data.df["close_bid_ask_previous"] = data.df["close_bid_ask"].shift(1) 
    data.df["adjusted_return"] = np.abs(data.df['Return']) - (data.df["close_bid_ask"]/2 + data.df["close_bid_ask_previous"]/2)/(data.df['close'].shift(1))
    print(len(data.df[data.df['adjusted_return']<0])/len(data.df))
    pdb.set_trace()
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_sentiment_H.txt"
    freq = 'H'
    print ('preprocessing...\n')
    data = d.data_tweets(fname, freq)
    size = data.get_length()
    print(size)
    data.df["close_bid_ask_previous"] = data.df["close_bid_ask"].shift(1) 
    data.df["adjusted_return"] = np.abs(data.df['Return']) - (data.df["close_bid_ask"]/2 + data.df["close_bid_ask_previous"]/2)/(data.df['close'].shift(1))
    print(len(data.df[data.df['adjusted_return']<0])/len(data.df))
    end = time.time()
    print(end - start)
    
if __name__ == '__main__':
    main()
