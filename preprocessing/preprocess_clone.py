from os.path import dirname as up
import pdb
import pandas as pd
from preprocessing.currency import currency
from preprocessing.tweet import tweet, tweet_preprocess
import numpy as np
from collections import defaultdict, OrderedDict
from datetime import timedelta


def preprocess(fname_tweets, sent):
    """
    Preprocess the currency and tweets data and merge them

    Args:
        fname_tweets: Formatted file containing the tweet data
        sent: Boolean indicating if sentence structure of the tweets need to
              be kept
    Returns
        The assembled and preprocessed data set for text analysis
    """
    mydateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    
    data_tweet = pd.read_csv(fname_tweets, index_col=0, date_parser=mydateparser, sep="\t")
    print('Cleaning the tweets...\n')
    obj = tweet_preprocess()
    data_tweet = obj.clean(data_tweet)
    print('Preprocessing the tweets...\n')
    if sent is True:
        data_tweet = obj.preprocess_sent(data_tweet, pos=True, stemming=True, lemmatization=False)
    else:
        data_tweet = obj.preprocess(data_tweet, pos=False, stemming=True, lemmatization=False)

    data_tweet = pd.DataFrame(data_tweet).reset_index()
    data_tweet.columns = ['Date', 'text', 'count']
    # Drop rows where there is any na
    data_tweet.dropna(inplace=True)
    path = str(up(up(up(__file__)))) + "/preprocessed_data/"
    data_tweet.to_csv(str(path) + 'tweets_split'  + '.txt', header=True,
                index=False, sep='\t', float_format='%.6f')

def preprocess_sentiment(fname_tweets, sent, filter=False):
    """
    Preprocess the currency and tweets data and merge them

    Args:
        fname_tweets: Formatted file containing the tweet data
        sent: Boolean indicating if sentence structure of the tweets need to
              be kept
    Returns
        The assembled and preprocessed data set for text analysis
    """
    mydateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    data_tweet = pd.read_csv(fname_tweets, index_col=0, date_parser=mydateparser, sep="\t")
    obj = tweet_preprocess()
    print('Cleaning the tweets...\n')
    data_tweet = obj.clean(data_tweet)
    print('Replacing negation...\n')
    data_tweet = obj.replace_neg(data_tweet)
    print('Preprocessing the tweets...\n')
    if sent is True:
        data_tweet = obj.preprocess_sent(data_tweet, pos=True, stemming=False, lemmatization=True)
        if filter is True:
            data_tweet = obj.filter_pos(data_tweet)
    else:
        data_tweet = obj.preprocess(data_tweet, pos=False, stemming=False, lemmatization=True)
    data_tweet = pd.DataFrame(data_tweet).reset_index()
    data_tweet.columns = ['Date', 'text', 'count']
    # Join both database
    # Drop rows where there is any na
    data_tweet.dropna(inplace=True)
    path = str(up(up(up(__file__)))) + "/preprocessed_data/"
    if filter is True:
        data_tweet.to_csv(str(path) + 'tweets_sentiment_pos_filter_split' + '.txt', header=True,
                    index=False, sep='\t', float_format='%.6f')
    else:
        if sent is True:
            data_tweet.to_csv(str(path) + 'tweets_sentiment_pos_split' + '.txt', header=True,
                        index=False, sep='\t', float_format='%.6f')
        else:
            data_tweet.to_csv(str(path) + 'tweets_sentiment_split' + '.txt', header=True,
                        index=False, sep='\t', float_format='%.6f')

def preprocess_vader(fname_tweets):

    mydateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    
    data_tweet = pd.read_csv(fname_tweets, index_col=0, date_parser=mydateparser, sep="\t")
    data_tweet.index.name="Date"
    # Drop rows where there is any na
    data_tweet.dropna(inplace=True)
    path = str(up(up(up(__file__)))) + "/preprocessed_data/"
    data_tweet.to_csv(str(path) + 'tweets_sentiment_vader_split' + '.txt', header=True,
                index=True, sep='\t', float_format='%.6f')

def preprocess_currency(fname_curr, freq, threshold):
    mydateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    data_currency = pd.read_csv(fname_curr, index_col=0, date_parser=mydateparser, sep="\t")
    obj = currency()
    data_currency = obj.resample(data_currency, freq)
    data_currency = obj.define_threshold(data_currency, threshold)
    data_currency = data_currency.drop(['open', 'high', 'low',
                                        'volume', 'open_bid_ask'],
                                        axis=1)
    path = str(up(up(up(__file__)))) + "/preprocessed_data/"
    data_currency.to_csv(str(path) + 'currency_' + str(freq)+ '.txt', header=True,
                index=True, sep='\t', float_format='%.6f')