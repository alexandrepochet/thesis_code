# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 20:42:50 2019

@author: alexa
"""

import numpy as np
import pandas as pd
import math
import time 
import currency_preprocess as c 
import tweet_preprocess as t 
import pdb


def main():

	"""
     
    Execute matching action for testing
     
    """
    start = time.time()
    file = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/forexDATA/currency.txt"
    currency = c.Currency(file = file)
    currency.data_analysis()
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/tweetsRawData/tweets.txt"
    tweets = t.Tweets(fname = fname)
    tweets.data_analysis('M')


if __name__ == '__main__':
	main()   
   
    