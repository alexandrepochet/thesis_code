# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 20:42:50 2019

@author: alexa
"""

import numpy as np
import pandas as pd
import preprocess as p
import math
import time
import wrapper as wr
import matplotlib.pyplot as plt 
import currency_preprocess as c 
import SVM as svm 
import pdb
import utils


def main():
    
    """
     
    Execute matching action for testing
     
    """
   
    start = time.time()
    file = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/forexDATA/currency.txt"
	# Daily frequency     

    data = c.Currency(file = file)
    print("resampling to daily...\n")
    data.resample('D')
    threshold = 0.0000
    data.define_threshold(threshold)
    size = len(data.df_resampled)
    print(size)
    
    df = data.df_resampled.iloc[0:int(0.8*size)]
    df = wr.add_all_ta_features(df, 'open', 'high', 'low', 'close', 
                                'volume', fillna=False)
    df.dropna(inplace = True)
    model = svm.SVM(df, threshold)
    train =[50, 100, 250, 500, 1000, 2000]
    test =[1/8, 1/4, 1/2, 1]
    param_grid = [{'kernel': ['rbf'], 'gamma': [10, 1, 0.1],
                   'C': [ 0.1, 1, 10]}]
    scoring = 'f1_micro'
    nfolds = 3
    kwargs = {"param_grid": param_grid, "scoring": scoring, "nfolds": nfolds}
    print ('training the daily svm model...\n')
    results = model.run(training_windows = train, testing_windows = test,
                        optimization = False)
    results_opti = model.run(training_windows = train, testing_windows = test,
                             optimization = True, kwargs = kwargs)
    print('--------------------- \n')
    print(results)
    print('--------------------- \n')
    print(results_opti)
    print('--------------------- \n')
    print([a_i[0] - b_i[0] for a_i, b_i in zip(results, results_opti)])

	# test based on best training-testing windows
    
    print("Test set.....\n")
    # Separate in train and validation set and test set
    df = data.df_resampled.iloc[int(0.8*size - 2000 - 77):]
    df = wr.add_all_ta_features(df, 'open', 'high', 'low', 'close', 
                               'volume', fillna=False)
    df.dropna(inplace = True)
    model = svm.SVM(df, threshold)
    train = 2000
    test = 250
    param_grid = [{'kernel': ['rbf'], 'gamma': [10, 1, 0.1],
                   'C': [ 0.1, 1, 10]}]
    scoring = 'f1_micro'
    nfolds = 3
    kwargs = {"param_grid": param_grid, "scoring": scoring, "nfolds": nfolds}
    print ('training the daily svm model...\n')
    results = model.run_svm(training_window = train, testing_window = test,
                        optimization = True, kwargs = kwargs)
    model.plot_return(title = 'Daily_SVM')
    print(results)
    
    # Prediction based on lag values of return only
    lags = 50
    df = data.df_resampled.iloc[int(0.8*size - lags - 2000):]
    utils.calculate_lags(df, lags)
    df.dropna(inplace = True)
    model = svm.SVM(df, threshold)
    train = 2000
    test = 1000
    param_grid = [{'kernel': ['rbf'], 'gamma': [100, 10, 1, 0.1, 0.01],
                   'C': [0.01, 0.1, 1, 10, 100]}]
    scoring = 'f1_micro'
    nfolds = 3
    kwargs = {"param_grid": param_grid, "scoring": scoring, "nfolds": nfolds}
    print ('training the daily svm model...\n')
    results = model.run_svm(training_window = train, testing_window = test,
                        optimization = True, kwargs = kwargs)
    model.plot_return(title = 'Daily_SVM_lagged')
    print(results)

    # Hourly frequency
    data = c.Currency(file = file)
    print("resampling to hourly...\n")
    data.resample('H')
    threshold = 0.0000
    data.define_threshold(threshold)
    # Separate in train and validation set and test set
    size = len(data.df_resampled)
    
    df = data.df_resampled.iloc[0:int(0.8*size)]
    df = wr.add_all_ta_features(df, 'open', 'high', 'low', 'close', 
                                'volume', fillna=False)
    df.dropna(inplace = True)
    model = svm.SVM(df, threshold)
    train =[250, 1000, 5000, 10000, 20000]
    test =[1/8, 1/4, 1/2, 1]
    print ('training the hourly svm model...\n')
    results = model.run(training_windows = train, testing_windows = test,
                        		optimization = False)
    print('--------------------- \n')
    print(results)
    print('--------------------- \n')
    
	# test based on best training-testing windows
    
    df = data.df_resampled.iloc[int(0.8*size - 20000 - 77):]
    df = wr.add_all_ta_features(df, 'open', 'high', 'low', 'close', 
                               'volume', fillna=False)
    df.dropna(inplace = True)
    model = svm.SVM(df, threshold)
    train = 1000
    test = 125
    print ('training the hourly svm model...\n')
    results = model.run_svm(training_window = train, testing_window = test,
                            optimization = False)
    model.plot_return(title = 'Hourly_SVM_TA')
    print(results)
    
    # Prediction based on lag values of return only
    lags = 50
    df = data.df_resampled.iloc[int(0.8*size - lags - 20000):]
    utils.calculate_lags(df, lags)
    df.dropna(inplace = True)
    model = svm.SVM(df, threshold)
    train = 1000
    test = 125
    print ('training the hourly svm model...\n')
    results = model.run_svm(training_window = train, testing_window = test,
                        optimization = False)
    model.plot_return(title = 'Hourly_SVM_lagged')
    print(results)
    
    end = time.time()
    print(end - start)
    
    
if __name__ == '__main__':
	main()   