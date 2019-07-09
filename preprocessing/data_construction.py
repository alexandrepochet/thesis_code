# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 20:42:50 2019

@author: alexa
"""
import numpy as np
import pandas as pd
import math
import time
import preprocessing.currency_preprocess as c
import preprocessing.tweet_preprocess as t
import pdb


def main():
    
    """
     
    Execute matching action for testing
     
    """
    
    start = time.time() 
    fname_ask = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/forexDATA/eurusd_ask_full.csv"
    fname_bid = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/forexDATA/eurusd_bid_full.csv" 
    currency = c.Currency(fname_ask = fname_ask, fname_bid = fname_bid)
    
    raw_file = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/tweetsRawData/tweets_eurusd.txt"
    tweets = t.Tweets(raw_file = raw_file)
    end = time.time()
    print(end - start)


if __name__ == '__main__':
    main()   