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
from data.data import data
import time


class data_currency(data):

    def __init__(self, fname=None, freq=None):
        super().__init__(fname, freq)

    def get_open_bid_ask(self):
        try:
            return pd.DataFrame(self.df.open_bid_ask, columns = ['open_bid_ask'],index=self.df.index)
        except:
            print("issue")
            return None

    def get_open(self):
        try:
            return pd.DataFrame(self.df.open, columns = ['open'],index=self.df.index)
        except:
            print("issue")
            return None

    def get_high(self):
        try:
            return pd.DataFrame(self.df.high, columns = ['high'],index=self.df.index)
        except:
            print("issue")
            return None

    def get_low(self):
        try:
            return pd.DataFrame(self.df.low, columns = ['low'],index=self.df.index)
        except:
            print("issue")
            return None

    def get_volume(self):
        try:
            return pd.DataFrame(self.df.volume, columns = ['volume'],index=self.df.index)
        except:
            print("issue")
            return None

    def resample(self, time_frame):
        """
        Resample the data based on the given time frame

        Args:
            time_frame: The time frame at which to resample
        """
        # Resample given the time frame parameter
        try:
            df_resampled = self.df.resample(time_frame).agg( \
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
            self.df = df_resampled
        except:
            print("issue")
            return

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
            elif x < -threshold:
                return 'down'
            else:
                return 'stable'
        try:
            self.df['Direction'] = self.df.apply(lambda x: func(x['Return'], threshold), axis=1)
        except:
            print("issue")
            return

    def add_indicators(self):
        try:
            df = wr.add_all_ta_features(self.get_df(), 'open', 'high', 'low', 'close', 
                                        'volume', fillna=False)
            df.dropna(inplace = True)
            self.df = df
        except:
            print("issue")
            return

    def add_lags(self, lags):
        try:
            df = utils.calculate_lags(self.get_df(), lags)
            df.dropna(inplace = True)
            self.df = df
        except:
            print("issue")
            return