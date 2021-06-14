from os.path import dirname as up
import pdb
import pandas as pd
from preprocessing.currency import currency
from preprocessing.tweet import tweet, tweet_preprocess
import numpy as np
import ast
from collections import defaultdict, OrderedDict
from datetime import timedelta


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
    mydateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    
    data_currency = pd.read_csv(fname_curr, index_col=0, date_parser=mydateparser, sep="\t")
    obj = currency()
    print('Resampling the currency data frame...\n')
    data_currency = obj.resample(data_currency, freq)
    data_currency = obj.define_threshold(data_currency, threshold)
    data_currency = data_currency.drop(['open', 'high', 'low',
                                        'volume', 'open_bid_ask'],
                                        axis=1)
    
    data_tweet = pd.read_csv(fname_tweets, index_col=0, date_parser=mydateparser, sep="\t")
    obj = tweet()
    print('Resampling the tweets...\n')
    data_tweet = obj.resample(data_tweet, freq, ' ')
    print('Cleaning the tweets...\n')
    obj = tweet_preprocess()
    data_tweet = obj.clean(data_tweet)
    print('Preprocessing the tweets...\n')
    if sent is True:
        data_tweet = obj.preprocess_sent(data_tweet, pos=True, stemming=True, lemmatization=False)
    else:
        data_tweet = obj.preprocess_bis(data_tweet, pos=False, stemming=True, lemmatization=False)
    # Join both database
    print('Merging the tweets and the currency data...\n')
    data_tweet = pd.DataFrame(data_tweet).reset_index()
    data_tweet.columns = ['Date', 'text', 'count']
    data = data_currency.merge(data_tweet, on='Date')
    # Drop rows where there is any na
    data.dropna(inplace=True)
    path = str(up(up(up(__file__)))) + "/preprocessed_data/"
    data.to_csv(str(path) + 'tweets_' + str(freq) + '.txt', header=True,
                index=False, sep='\t', float_format='%.6f')

def preprocess_sentiment(fname_curr, fname_tweets, freq, threshold, sent, filter=False):
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
    mydateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    
    data_currency = pd.read_csv(fname_curr, index_col=0, date_parser=mydateparser, sep="\t")
    obj = currency()
    data_currency = obj.resample(data_currency, freq)
    data_currency = obj.define_threshold(data_currency, threshold)
    data_currency = data_currency.drop(['open', 'high', 'low',
                                        'volume', 'open_bid_ask'],
                                        axis=1)

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
        obj = tweet()
        print('Resampling the tweets...\n')
        data_tweet = obj.resample_sentiment(data_tweet, freq)
    else:
        data_tweet = obj.preprocess(data_tweet, pos=False, stemming=False, lemmatization=True)
        obj = tweet()
        print('Resampling the tweets...\n')
        #data_tweet.resample(freq, ' ')
        data_tweet = obj.resample_sentiment(data_tweet, freq)

    # Join both database
    print('Merging the tweets and the currency data...\n')
    data = data_currency.merge(data_tweet, on='Date')
    # Drop rows where there is any na
    data.dropna(inplace=True)
    path = str(up(up(up(__file__)))) + "/preprocessed_data/"
    if filter is True:
        data.to_csv(str(path) + 'tweets_sentiment_pos_filter_2' + str(freq) + '.txt', header=True,
                    index=False, sep='\t', float_format='%.6f')
    else:
        if sent is True:
            data.to_csv(str(path) + 'tweets_sentiment_pos_2' + str(freq) + '.txt', header=True,
                        index=False, sep='\t', float_format='%.6f')
        else:
            data.to_csv(str(path) + 'tweets_sentiment_2' + str(freq) + '.txt', header=True,
                        index=False, sep='\t', float_format='%.6f')

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
    elif freq == 'H' or freq == 'm' or freq == 's':
        mydateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    else:
        mydateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

    data = pd.read_csv(fname, index_col=0, date_parser=mydateparser, sep="\t")
    return data

def preprocess_vader(fname_curr, fname_tweets, freq, threshold):

    mydateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    
    data_currency = pd.read_csv(fname_curr, index_col=0, date_parser=mydateparser, sep="\t")
    obj = currency()
    data_currency = obj.resample(data_currency, freq)
    data_currency = obj.define_threshold(data_currency, threshold)
    data_currency = data_currency.drop(['open', 'high', 'low',
                                        'volume', 'open_bid_ask'],
                                        axis=1)
    data_tweet = pd.read_csv(fname_tweets, index_col=0, date_parser=mydateparser, sep="\t")
    obj = tweet()
    data_tweet = obj.resample(data_tweet, freq, '. ')
    #if freq =='H':
    #    data_tweet.index = data_tweet.index - timedelta(hours=1)
    #elif freq=="D":
    #    data_tweet.index = data_tweet.index - timedelta(days=1)
    #else:
    #    data_tweet.index = data_tweet.index - timedelta(seconds=60)
    data_tweet.index.name="Date"
    # Join both database
    print('Merging the tweets and the currency data...\n')
    data = data_currency.merge(data_tweet, on='Date')
    # Drop rows where there is any na
    data.dropna(inplace=True)
    data.Return = data.Return.astype(float)
    path = str(up(up(up(__file__)))) + "/preprocessed_data/"
    data.to_csv(str(path) + 'tweets_sentiment_vader2_' + str(freq) + '.txt', header=True,
                index=True, sep='\t', float_format='%.6f')

def preprocess140(file):

    def fun(row, format_):
        return pd.datetime.strptime(row['Date'], format_)

    elements = defaultdict(list)
    with open(file, "r") as f:
        i = 0
        lines = f.read().split("\n")
        print(len(lines))
        for line in lines:
            currentline = line.split('::')
            try:
                elements['sentiment'].append(int(currentline[0]))
                elements['text'].append(str(currentline[5]))
                elements['count'].append(int("1"))
                elements['Date'].append("2000-01-01 00:00:01")
            except:
                print('line ' + str(i) + ' issue \n')
                continue
            i = i + 1
            if i%10000 == 0:
                print(i)
    f.close()
    data = pd.DataFrame({'Date': pd.Index(elements['Date']),
                         'sentiment': pd.Index(elements['sentiment']),
                         'text': pd.Index(elements['text']),
                         'count': pd.Index(elements['count'])
                         })
    data['Date'] = data.apply(lambda x: fun(x, '%Y-%m-%d %H:%M:%S'), axis=1)
    path = 'C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/'
    data.to_csv(str(path) + 'sentiment140_vader.txt', sep='\t', index=False)
 
    #Various preprocesses 
    mydateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    fname_tweets = 'C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_datasentiment140_vader.txt'
    data_tweet_init = pd.read_csv(fname_tweets, index_col=0, date_parser=mydateparser, sep="\t")
    obj = tweet_preprocess()
    
    data_tweet = data_tweet_init.copy()
    print('Cleaning the tweets...\n')
    data_tweet = obj.clean(data_tweet)
    print('Replacing negation...\n')
    data_tweet = obj.replace_neg(data_tweet)
    print('Preprocessing the tweets...\n')
    data_tweet = obj.preprocess(data_tweet, pos=False, stemming=False, lemmatization=True)
    data_tweet.dropna(inplace=True)
    path = str(up(up(up(__file__)))) + "/preprocessed_data/"
    data_tweet.to_csv(str(path) + 'sentiment140' + '.txt', header=True,
                      index=True, sep='\t', float_format='%.6f')

    data_tweet = data_tweet_init.copy()
    print('Cleaning the tweets...\n')
    data_tweet = obj.clean(data_tweet)
    print('Replacing negation...\n')
    data_tweet = obj.replace_neg(data_tweet)
    print('Preprocessing the tweets...\n')
    data_tweet = obj.preprocess_sent(data_tweet, pos=True, stemming=False, lemmatization=True)
    data_tweet.dropna(inplace=True)
    path = str(up(up(up(__file__)))) + "/preprocessed_data/"
    data_tweet.to_csv(str(path) + 'sentiment140_pos' + '.txt', header=True,
                      index=True, sep='\t', float_format='%.6f')

    data_tweet = data_tweet_init.copy()
    print('Cleaning the tweets...\n')
    data_tweet = obj.clean(data_tweet)
    print('Replacing negation...\n')
    data_tweet = obj.replace_neg(data_tweet)
    print('Preprocessing the tweets...\n')
    data_tweet = obj.preprocess_sent(data_tweet, pos=True, stemming=False, lemmatization=True)
    data_tweet = obj.filter_pos(data_tweet)
    data_tweet.dropna(inplace=True)
    path = str(up(up(up(__file__)))) + "/preprocessed_data/"
    data_tweet.to_csv(str(path) + 'sentiment140_pos_filter' + '2.txt', header=True,
                      index=True, sep='\t', float_format='%.6f')