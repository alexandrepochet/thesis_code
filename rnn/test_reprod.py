seed_value= 0
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)
# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
#tf.random.set_seed(seed_value) # tensorflow 2.x
tf.set_random_seed(seed_value) # tensorflow 1.x
# 5. Configure a new global `tensorflow` session
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


import pandas as pd
import numpy as np
import data.data_sentiments as d
from rnn.lstmConfig import lstmConfig
from utils.GridSearchCV import GridSearchCV
from utils.ShiftingWindowSplit import ShiftingWindowSplit
from utils.ExpandingWindowSplit import ExpandingWindowSplit
from utils.currencyReturn import currencyReturn
from utils.utils import reset_keras
import time
import pdb
import warnings
import math


def main(load=1): 
    """  
    Execute matching action for testing  
    """

    #warnings.filterwarnings("ignore")
    start = time.time()

    ##Daily data
    #file where all the sentiment time series are stored
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/daily_sentiment_series2.txt"
    freq = 'D'
    print ('preprocessing...\n')
    data = d.data_sentiments(fname, freq)
    size = data.get_length()
    X = data.get_df()
    cols = X.columns
    cols = cols[4:]
    for col in cols:
        X[col] = X[col].shift(1)
    X = X.fillna(0)
    X = X.drop(['Direction', 'close_bid_ask', 'Return', 'close_bid_ask_previous', 'close'], axis=1)
    X = np.asarray(X)
    Y = data.get_direction_num()
    Y = np.asarray(Y.values)
    reset_keras()
    model = lstmConfig()
    model.fit(X, Y)


    end = time.time()
    print(end - start)


if __name__ == '__main__':
    main()