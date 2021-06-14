import ast
import pdb
import sys
import warnings
import pandas as pd
import preprocessing.preprocess as p
from abc import ABCMeta, abstractmethod
from copy import deepcopy
import pdb
import numpy as np

class data(metaclass=ABCMeta):

    def __init__(self, fname=None, freq=None):
        np.random.seed(42)
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

    def get_close_bid_ask_previous(self):
        try:
            return pd.DataFrame(self.df.close_bid_ask_previous, columns = ['close_bid_ask_previous'],index=self.df.index)
        except:
            print("issue")
            return None

    def get_return(self):
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

    def set_df(self, df):
        try:
            self.df = df
        except:
            print("issue")

    def slice(self, begin, end):
        new_obj = deepcopy(self)
        try:
            new_obj.df = new_obj.df.iloc[begin:min(end, self.get_length())]
            return new_obj
        except:
            print('begin and end values are not consistent with the size of the object')
            return

    def get_direction_num(self):
        direction_num = pd.DataFrame(self.df.Direction, columns = ['Direction'],index=self.df.index)
        direction_num.loc[direction_num.Direction=='up', 'Direction']=1
        direction_num.loc[direction_num.Direction=='down', 'Direction']=0
        try:
            direction_num.loc[direction_num.Direction=='stable', 'Direction']=0
        except:
            print("no stable")
        return direction_num

    def shuffle(self):
        self.df = self.df.sample(frac=1)