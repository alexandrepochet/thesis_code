import technical_analysis.volatility as volatility
import technical_analysis.trend as trend
import technical_analysis.momentum as momentum
import technical_analysis.others as others
import technical_analysis.volume as v
import math
import sys
import pdb
import string
import re
import json
from datetime import datetime, date, time, timedelta
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import tensorflow as tf
import random as python_random
import ast
import os


def dropna(df):
    """
    Drop rows with "Nans" values
    """
    df = df[df < math.exp(709)] # big number
    df = df[df != 0.0]
    df = df.dropna()
    return df

def ema(series, periods, fillna=False):
    if fillna:
        return series.ewm(span=periods, min_periods=0).mean()
    return series.ewm(span=periods, min_periods=periods).mean()
    
def calculate_lags(df, lags):
    """     
    Calculate the lag variables of the return
        
    Args:
        df: Data
        lags: Number of lags to calculate 
    Returns          
    """    
    
    for i in range(1, lags + 1):
        df['Return_lag_' + str(i)] = df.Return.shift(i)

    df = df.iloc[lags:]

    return df

def add_lags(df, lags, keep=False):
    """     
    Calculate the lag variables of the return
        
    Args:
        df: Data
        lags: Number of lags to calculate 
    Returns          
    """    
    columns = df.columns
    for i in range(0, len(columns)):
        for j in range(1, lags + 1):
            df[str(columns[i]) + '_' + str(j)] = df[str(columns[i])].shift(j)
    if keep is False:
        df = df.iloc[lags:]
    return df

def load_words():
    """
    Return file with English vocabulary

    Args:
    Returns:
        A list with valid English words
    """
    dictionary = 'C:/Users/alexa/OneDrive/Desktop/Thesis/Lexicon/English/words_alpha.txt'
    with open(dictionary) as word_file:
        valid_words = set(word_file.read().split())

    return valid_words

def initialize():
    """
    Initialize various syntactic variables for preprocessing

    Args:
    Returns:
        The syntactic variables for further text processing
    """
    pat1 = r'@[A-Za-z0-9_]+'
    pat2 = r'http[s]?:// [^ ]+'
    pat3 = r'http[s]?://www. [^ ]+'
    pat4 = r'http[s]?://[^ ]+'
    pat4 = r'http[s]?://[^ ]+'
    pat5 = r'http[s]?[^ ]+'
    www_pat = r'www.[^ ]+'
    combined_pat = r'|'.join((pat1, pat2, pat3, pat4, pat5, www_pat))

    dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not",
           "weren't":"were not", "haven't":"have not",
           "hasn't":"has not", "hadn't":"had not", "won't":"will not",
           "wouldn't":"would not", "don't":"do not", "doesn't":"does not",
           "didn't":"did not", "can't":"can not", "couldn't":"could not",
           "shouldn't":"should not", "mightn't":"might not", "cannot":"can not",
           "mustn't":"must not", "needn't":"need not", "b uy": "buy",
           "tech nical":"technical", "fu ndamental": "fundamental",
           "s ell":"sell", "technica l":"technical", "foreca st":"forecast",
           "FXStreetFlashP ublic":"FXStreetFlashPublic", "0 2":"02", "0 4":"04"}
    pattern = re.compile(r'\b(' + '|'.join(dic.keys()) + r')\b')

    punctuation = list(string.punctuation)
    stop_words = stopwords.words('english') + punctuation + ['rt', 'via']
    preposition = ["above", "below", "up", "down", "no", "nor", "not"]
    stop_words = [stop_words.remove(x) for x in preposition]

    emoticons_str = r"""(?: [:=;] # Eyes
                            [D\)\]\(\]/\\OpP] # Mouth
                        )"""
    #[oO\-]? # Nose (optional)
    regex_str = [emoticons_str,
                 r'<[^>]+>', # HTML tags
                 r'(?:@[\w_]+)', # @-mentions
                 #r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
                 r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
                 r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
                 r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
                 r'(?:[\w_]+)', # other words
                 r'(?:\S)' # anything else
                ]

    tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
    emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)

    return combined_pat, pattern, dic, stop_words, tokens_re, emoticon_re

def flatten(A):
    rt = []
    for i in A:
        if isinstance(i,list): rt.extend(flatten(i))
        else: rt.append(i)
    return rt

def func(row, frequency, delay=True):
    """
    Convenience method for apply function. Format a date given a frequency
    parameter

    Args:
        row: The row of the dataframe on which the apply function is executed
        frequency: The formatting frequency. Hourly or daily frequencies are possible
        delay: add 1 period to each date, for the merging with the currency data
    Returns:
        The date formatted at the given frequency
    """
    if frequency == 'H':
        if delay:
            date_ = datetime.combine(date(row.name.year, row.name.month, row.name.day),
                                     time(row.name.hour, 0)) + timedelta(hours=1)
        else:
            date_ = datetime.combine(date(row.name.year, row.name.month, row.name.day),
                                     time(row.name.hour, 0))
        return pd.to_datetime(date_, format='%Y-%m-%d %H:%M:%S')
    elif frequency == 'D':
        if delay:
            date_ = date(row.name.year, row.name.month, row.name.day) + timedelta(days=1)
        else:
            date_ = date(row.name.year, row.name.month, row.name.day)
        return pd.to_datetime(date_, format='%Y-%m-%d')
    elif frequency == 'min':
        if delay:
            date_ = datetime.combine(date(row.name.year, row.name.month, row.name.day),
                                     time(row.name.hour, row.name.minute)) + timedelta(minutes=1)
        else:
            date_ = datetime.combine(date(row.name.year, row.name.month, row.name.day),
                                     time(row.name.hour, row.name.minute))
        return pd.to_datetime(date_, format='%Y-%m-%d %H:%M:%S')
    else:
        print("invalid frequency")
        return 0

def define_date_frequency(data, frequency, delay):
    """
    Resample the data by concatenating the tweets at the given frequency
    """
    if frequency == 'H':
        data.index = data.apply(lambda x: func(x, 'H', delay), axis=1)
    elif frequency == 'D':
        data.index = data.apply(lambda x: func(x, 'D', delay), axis=1)
    elif frequency == 'min':
        data.index = data.apply(lambda x: func(x, 'min', delay), axis=1)
    else:
        print("invalid frequency")
        
    return data

def reset_keras():
    seed_value = 0
    np.random.seed(seed_value)
    python_random.seed(seed_value)
    tf.compat.v1.random.set_random_seed(seed_value)
    config = tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=config)
    tf.keras.backend.clear_session()
    sess.close()
    del sess

def count_features(x):
    list_words = []
    for i in range(0, len(x)-1):
        words_array = ast.literal_eval(x[i])
        for word in words_array:
            list_words.append(word)
    count = len(set(list_words))
    return count

def add_features(df, close, high, low, volume, fillna=False):

    n_range = [5]
    for n in n_range:
        df['momentum_rsi_' + str(n)] = momentum.rsi(df[close], n=n, fillna=fillna)
        df['momentum_stoch_' + str(n)] = momentum.stoch(df[high], df[low], df[close], n=n, fillna=fillna)
        df['trend_adx_' + str(n)] = trend.adx(df[high], df[low], df[close], n=n, fillna=fillna)
        df['trend_adx_pos_' + str(n)] = trend.adx_pos(df[high], df[low], df[close], n=n, fillna=fillna)
        df['trend_adx_neg_' + str(n)] = trend.adx_neg(df[high], df[low], df[close], n=n, fillna=fillna)
        df['volatility_atr_' + str(n)] = volatility.average_true_range(df[high], df[low], df[close], n=n, fillna=fillna)

    values_range = [(5,35,1),(5,13,1),(12,26,1)]
    for values in values_range:
        df['trend_macd_' + str(values[0]) + '_' + str(values[1])] = trend.macd(df[close], n_fast=values[0], n_slow=values[1], fillna=fillna)
        df['trend_macd_signal_' + str(values[0]) + '_' + str(values[1]) + '_' + str(values[2])] = trend.macd_signal(df[close], n_fast=values[0], n_slow=values[1], n_sign=values[2], fillna=fillna)
        df['trend_macd_diff_' + str(values[0]) + '_' + str(values[1]) + '_' + str(values[2])] = trend.macd_diff(df[close], n_fast=values[0], n_slow=values[1], n_sign=values[2], fillna=fillna)

    n_range = [5]
    for n in n_range:
        df['trend_aroon_up_' + str(n)] = trend.aroon_up(df[close], n=n, fillna=fillna)
        df['trend_aroon_down_' + str(n)] = trend.aroon_down(df[close], n=n, fillna=fillna)
        df['trend_aroon_ind_' + str(n)] = df['trend_aroon_up_' + str(n)] - df['trend_aroon_down_' + str(n)]
        df['trend_ema_' + str(n)] = trend.ema_indicator(df[close], n=n, fillna=fillna)

    n_range = [5]
    for n in n_range:
        df['volume_obvm_' + str(n)] = v.on_balance_volume_mean(df[close], df[volume], n=n, fillna=fillna)

    n_range = [5]
    for n in n_range:
        df['volume_cmf_' + str(n)] = v.chaikin_money_flow(df[high], df[low], df[close], df[volume], n=n, fillna=fillna)
        df['volatility_bbh_' + str(n)] = volatility.bollinger_hband(df[close], n=n, ndev=2, fillna=fillna)
        df['volatility_bbl_' + str(n)] = volatility.bollinger_lband(df[close], n=n, ndev=2, fillna=fillna)
        df['volatility_bbm_' + str(n)] = volatility.bollinger_mavg(df[close], n=n, fillna=fillna)
        df['volatility_bbhi_' + str(n)] = volatility.bollinger_hband_indicator(df[close], n=n, ndev=2,fillna=fillna)
        df['volatility_bbli_' + str(n)] = volatility.bollinger_lband_indicator(df[close], n=n, ndev=2,fillna=fillna)
    
    return df

def weighted_mean(weight, sample):
    products = []
    for num1, num2 in zip(sample, weight):
          products.append(num1 * num2)
    return sum(products)

def weighted_std(weight, sample):
    mean = weighted_mean(weight, sample)
    products = []
    for num1, num2 in zip(sample, weight):
          products.append(((num1 - mean)**2) * num2)
    N = len(sample)
    return np.sqrt(N/(N-1)*sum(products))
