import ast
import pdb
import sys
import warnings
import pandas as pd
import numpy as np
import preprocessing.preprocess as p
from copy import deepcopy
from collections import defaultdict, OrderedDict
import technical_analysis.wrapper as wr
import utils.utils as utils
import time


class data_currency():

    def __init__(self, fname, freq='m'):
        self.df_tweets = p.get_preprocessed_data(fname, freq)

    def get_length(self):
        return len(self.df_tweets)

    def get_date(self):
        return self.df_tweets.index

    def get_close(self):
        return pd.DataFrame(self.df_tweets.close, columns = ['close'],index=self.df_tweets.index)

    def get_close_bid_ask(self):
        return pd.DataFrame(self.df_tweets.close_bid_ask, columns = ['close_bid_ask'],index=self.df_tweets.index)

    def get_open_bid_ask(self):
        return pd.DataFrame(self.df_tweets.open_bid_ask, columns = ['open_bid_ask'],index=self.df_tweets.index)

    def get_Return(self):
        return pd.DataFrame(self.df_tweets.Return, columns = ['Return'],index=self.df_tweets.index)

    def get_open(self):
        return pd.DataFrame(self.df_tweets.open, columns = ['open'],index=self.df_tweets.index)

    def get_high(self):
        return pd.DataFrame(self.df_tweets.high, columns = ['high'],index=self.df_tweets.index)

    def get_low(self):
        return pd.DataFrame(self.df_tweets.low, columns = ['low'],index=self.df_tweets.index)

    def get_volume(self):
        return pd.DataFrame(self.df_tweets.volume, columns = ['volume'],index=self.df_tweets.index)

    def get_direction(self):
        return pd.DataFrame(self.df_tweets.Direction, columns = ['Direction'],index=self.df_tweets.index)

    def get_df(self):
        return self.df_tweets

    def resample(self, time_frame):
        """
        Resample the data based on the given time frame

        Args:
            time_frame: The time frame at which to resample
        """
        # Resample given the time frame parameter
        df_resampled = self.df_tweets.resample(time_frame).agg( \
                            OrderedDict([('open', 'first'),
                                         ('high', 'max'),
                                         ('low', 'min'),
                                         ('close', 'last'),
                                         ('volume', 'sum'),
                                         ('open_bid_ask', 'first'),
                                         ('close_bid_ask', 'last'),
                                        ])
                            )
        df_resampled.replace(["NaN", 'NaT'], np.nan, inplace=True)
        df_resampled.dropna(how='any', inplace=True)
        df_resampled['Return'] = np.log(df_resampled.close).diff()
        df_resampled.dropna(how='any', inplace=True)
        self.df_tweets = df_resampled

    def define_threshold(self, threshold):
        """
        Defines the direction of the market for each date
        based on the given threshold. If the return is higher
        than the threshold, direction is flagged as up, if the
        return is lower than -threshold, direction is flagged
        as down, and flagged as stable if the return lies in
        between

        Args:
            threshold: The threshold
        """
        def func(x, threshold):
            if x > threshold:
                return 'up'
            elif x < threshold:
                return 'down'
            else:
                return 'stable'

        self.df_tweets['Direction'] = self.df_tweets.apply(lambda x: func(x['Return'], threshold), axis=1)
    
    def slice(self, begin, end):
        new_obj = deepcopy(self)
        try:
            new_obj.df_tweets = new_obj.df_tweets.iloc[begin:min(end, self.get_length())]
        except:
            print('begin and end values are not consistent with the size of the object')
            return

        return new_obj

    def add_indicators(self):
        df = wr.add_all_ta_features(self.get_df(), 'open', 'high', 'low', 'close', 
                                    'volume', fillna=False)
        df.dropna(inplace = True)
        self.df_tweets = df

    def add_lags(self, lags):
        df = utils.calculate_lags(self.get_df(), lags)
        df.dropna(inplace = True)
        self.df_tweets = df