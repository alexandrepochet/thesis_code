import ast
import pdb
import sys
import warnings
import pandas as pd
import preprocessing.preprocess as p
from data.data import data
from copy import deepcopy
import pdb


class data_sentiments(data):

    def __init__(self, fname=None, freq=None):
        super().__init__(fname, freq)