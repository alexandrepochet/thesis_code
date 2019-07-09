# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 14:24:51 2019

@author: alexa
"""

import preprocessing.currency_preprocess as c
import preprocessing.tweet_preprocess as t
import pandas as pd
from os.path import dirname as up
import pdb
import time


def main():
    
     """
     
     Execute matching action for testing
     
     """
     start = time.time()
     fname_curr = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/forexDATA/currency.txt"
     fname_tweets = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/tweetsRawData/tweets.txt"
     freq = 'D'
     threshold = 0.0000
     print ('preprocessing daily data...\n')
     preprocess_sentiment(fname_curr, fname_tweets, freq, threshold, False)
     preprocess(fname_curr, fname_tweets, freq, threshold, False) 
     freq = 'H'
     threshold = 0.0000
     print ('preprocessing hourly data...\n')
     preprocess_sentiment(fname_curr, fname_tweets, freq, threshold, False)
     preprocess(fname_curr, fname_tweets, freq, threshold, False) 
    
     end = time.time()
     print(end - start)


if __name__ == '__main__':
    main()   