import ast
import pdb
import sys
import warnings
import preprocessing.preprocess as p


class data():

    def __init__(fname, freq):

        df_tweets = p.get_preprocessed_data(fname, freq)
        self.date = df_tweets.index
        self.close = df_tweets.close.value
        self.close_bid_ask = df_tweets.close_bid_ask.value
        self.Return = df_tweets.Return.value
        self.direction = df_tweets.direction.value
        self.count = df_tweets.count.value
        self.text = df_tweets.text.value

    def get_length(self):
        return len(self.date)

    def get_date(self):
        return self.date

    def get_close(self):
        return self.close

    def get_close_bid_ask(self):
        return self.close_bid_ask

    def get_Return(self):
        return self.Return

    def get_direction(self):
        return self.direction

    def get_count(self):
        return self.count

    def get_test(self):
        return self.text