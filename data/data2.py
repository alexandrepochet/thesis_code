import ast
import pdb
import sys
import warnings
import pandas as pd
import preprocessing.preprocess as p
from abc import ABCMeta, abstractmethod
from copy import deepcopy
import pdb


class data(metaclass=ABCMeta):

    def __init__(self, fname=None, freq=None):
        if fname is None and freq is None:
            self.df = pd.DataFrame() 
        else:
            self.df = p.get_preprocessed_data(fname, freq) 

    def get_length(self):
        try:
            return len(self.df)
        except:
            print("issue")
            return None

    def get_date(self):
        try:
            return self.df.index
        except:
            print("issue")
            return None

    def get_close(self):
        try:
            return pd.DataFrame(self.df.close, columns = ['close'],index=self.df.index)
        except:
            print("issue")
            return None

    def get_close_bid_ask(self):
        try:
            return pd.DataFrame(self.df.close_bid_ask, columns = ['close_bid_ask'],index=self.df.index)
        except:
            print("issue")
            return None

    def get_Return(self):
        try:
            return pd.DataFrame(self.df.Return, columns = ['Return'],index=self.df.index)
        except:
            print("issue")
            return None

    def get_direction(self):
        try:
            return pd.DataFrame(self.df.Direction, columns = ['Direction'],index=self.df.index)
        except:
            print("issue")
            return None

    def get_df(self):
        try:
            return self.df
        except:
            print("issue")
            return None

    def slice(self, begin, end):
        new_obj = deepcopy(self)
        try:
            new_obj.df = new_obj.df.iloc[begin:min(end, self.get_length())]
            return new_obj
        except:
            print('begin and end values are not consistent with the size of the object')
            return