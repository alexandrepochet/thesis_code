import pandas as pd
import numpy as np
import copy
import data.data_sentiments as d
from rnn.lstmConfig_custom import lstmConfig
from utils.GridSearchCV_custom import GridSearchCV_custom
from utils.ShiftingWindowSplit import ShiftingWindowSplit
from utils.ExpandingWindowSplit import ExpandingWindowSplit
from utils.currencyReturn import currencyReturn
from utils.utils import reset_keras
from silence_tensorflow import silence_tensorflow
import time
import pdb
import warnings
import math


def main(load=1): 
    """  
    Execute matching action for testing  
    """

    warnings.filterwarnings("ignore")
    silence_tensorflow()
    start = time.time()


    ##Daily data
    #file where all the sentiment time series are stored
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/daily_sentiment_series.txt"
    freq = 'D'
    print ('preprocessing...\n')
    data = d.data_sentiments(fname, freq)
    size = data.get_length()
    X = data.get_df()
    
    location = 'C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/src/output/'
    predicted_directions = np.loadtxt(location + str('RNN_SA_NA_return_daily_condition.txt'))

    # Back test
    #predicted_directions = np.concatenate(predicted_directions).ravel()
    length = len(predicted_directions)
    history = len(X)
    y_test = data.get_direction_num().iloc[(history-length):]
    market_returns = data.get_return().iloc[(history-length):]
    bid_ask = data.get_close_bid_ask().iloc[(history-length):]
    bid_ask_previous = data.get_close_bid_ask_previous().iloc[(history-length):]
    currency = currencyReturn(y_test, market_returns, bid_ask, bid_ask_previous)
    currency.run(predicted_directions, t_cost=False)
    stats = currency.output_summary_stats()
    print(stats)

    currency = currencyReturn(y_test, market_returns, bid_ask, bid_ask_previous)
    currency.run(predicted_directions, t_cost=True)
    stats = currency.output_summary_stats()
    print(stats)

    pdb.set_trace()
    end = time.time()
    print(end - start)


if __name__ == '__main__':
    main()