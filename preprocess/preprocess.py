# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 14:24:51 2019

@author: alexa
"""

import currency_preprocess as c
import tweet_preprocess as t
import pandas as pd
from os.path import dirname as up
import pdb
import time


def preprocess(fname_curr, fname_tweets, freq, threshold, sent):
        
    """
        
    Preprocess the currency and tweets data and merge them
        
    Args:
        fname_curr: Formatted file containing the currency data
        fname_tweets: Formatted file containing the tweet data
        freq: The resampling frequency (timeframe) 
        threshold: Threshold for the estimation of the long, short or neutral
                   positions
        sent: Boolean indicating if sentence structure of the tweets need to 
              be kept
            
    Returns
    The assembled and preprocessed data set for text analysis
            
    """
        
    data_curr = c.Currency(file = fname_curr)
    data_curr.resample(freq)
    data_curr.define_threshold(threshold)
    data_curr.df_resampled = data_curr.df_resampled.drop(['open', 'high', 'low',
                                                          'volume', 'open_bid_ask'],
                                                         axis = 1) 
    data_tweet = t.Tweets(fname = fname_tweets) 
    print ('Resampling the tweets...\n')   
    data_tweet.resample(freq)
    print ('Cleaning the tweets...\n')
    data_tweet.clean()
    print ('Preprocessing the tweets...\n')
    if sent == True:
        data_tweet.preprocess_sent(POS = True, Stemming = True, Lemmatization = False)
    else:
        data_tweet.preprocess(POS = False, Stemming = True, Lemmatization = False)
    # Join both database        
    print ('Merging the tweets and the currency data...\n')
    data_tweet.df_preprocessed = pd.DataFrame(data_tweet.df_preprocessed).reset_index()
    data_tweet.df_preprocessed.columns = ['Date', 'text', 'count']
    data = data_curr.df_resampled.merge(data_tweet.df_preprocessed, on = 'Date')  
    # Drop rows where there is any na
    data.dropna(inplace = True)
    path = str(up(up(up(__file__)))) + "/preprocessed_data/"
    data.to_csv(str(path) + 'tweets_' + str(freq) + '.txt', header=True, 
                    index=False, sep='\t' , float_format='%.6f') 


def preprocess_sentiment(fname_curr, fname_tweets, freq, threshold, sent):
        
    """
        
    Preprocess the currency and tweets data and merge them
        
    Args:
        fname_curr: Formatted file containing the currency data
        fname_tweets: Formatted file containing the tweet data
        freq: The resampling frequency (timeframe) 
        threshold: Threshold for the estimation of the long, short or neutral
                   positions
        sent: Boolean indicating if sentence structure of the tweets need to 
              be kept
            
    Returns
    The assembled and preprocessed data set for text analysis
            
    """
        
    data_curr = c.Currency(file = fname_curr)
    data_curr.resample(freq)
    data_curr.define_threshold(threshold)
    data_curr.df_resampled = data_curr.df_resampled.drop(['open', 'high', 'low',
                                                          'volume', 'open_bid_ask'],
                                                         axis = 1) 
    data_tweet = t.Tweets(fname = fname_tweets) 
    print ('Cleaning the tweets...\n')
    data_tweet.clean()
    print ('Preprocessing the tweets...\n')
    if sent == True:
        data_tweet.preprocess_sent(POS = True, Stemming = False, Lemmatization = True)
    else:
        data_tweet.preprocess(POS = False, Stemming = False, Lemmatization = True)
    print ('Resampling the tweets...\n')   
    data_tweet.resample_sentiment(freq)
    # Join both database        
    print ('Merging the tweets and the currency data...\n')
    pdb.set_trace()
    data = data_curr.df_resampled.merge(data_tweet.df_preprocessed, on = 'Date')  
    # Drop rows where there is any na
    data.dropna(inplace = True)
    path = str(up(up(up(__file__)))) + "/preprocessed_data/"
    data.to_csv(str(path) + 'tweets_sentiment_' + str(freq) + '.txt', header=True, 
                    index=False, sep='\t' , float_format='%.6f') 


def get_preprocessed_data(fname, freq):
    
    """
        
    Open preprocessed data
        
    Args:
        fname: location and name of the preprocessed data
        freq: The resampling frequency (timeframe) 

    Returns
    The assembled and preprocessed data set for text analysis
            
    """    
    
    if freq == 'D':
        mydateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d') 
    elif freq == 'H':
        mydateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') 
    else:
        print ('wrong frequency')
    
    data = pd.read_csv(fname, index_col=0, date_parser=mydateparser, sep = "\t")
    data.reset_index(level=0, inplace=True)
    
    return data


   