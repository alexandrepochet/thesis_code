import pandas as pd
import preprocessing.preprocess as p
from data.data_currency import data_currency
import pdb


class data_tweets(data_currency):

    def __init__(self, fname=None, freq=None):
        super().__init__(fname, freq)

    def get_count(self):
        try:
            return self.df['count']
        except:
            print("issue")
            return None

    def get_text(self):
        try:
            return self.df.text
        except:
            print("issue")
            return None

    def join(self, other_obj):
        try:
            self.df = self.df.append(other_obj.df)
        except:
            print("issue")
            return

    def set_text(self, text, i):
        try:
            self.df.text.iloc[i] = text
        except:
            print("issue")
            return

    def to_csv(self, file, path, sep, index):
        try:
            self.df.to_csv(str(path) + str(file) + '.txt', sep='\t', index=index)
        except:
            print("issue")
            return

    def add_indicators(self):
        pass

    def add_lags(self, lags):
        pass

    def get_open_bid_ask(self):
        pass

    def get_open(self):
        pass

    def get_high(self):
        pass

    def get_low(self):
        pass

    def get_volume(self):
        pass

    def resample(self, time_frame):
        pass