import numpy as np
import pandas as pd
import math
import time
import matplotlib.pyplot as plt 
import data.data_currency as d
import crap.SVM as svm 
import pdb
import warnings


def main(): 
    """  
    Execute matching action for testing  
    """
    nb_processes = 4
    warnings.filterwarnings("ignore")
    start = time.time()
    file = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/forexDATA/currency.txt"
    data = d.data_currency(file) 
	
    
    # Daily frequency
    print("resampling to daily...\n")
    data.resample('D')
    threshold = 0.0000
    data.define_threshold(threshold)
    size = data.get_length()
    print(size)
  
    data_sliced = data.slice(0,int(0.8*size))
    data_sliced.add_indicators()
    model = svm.SVM(data_sliced)
    train =[50, 100]
    test =[1/8, 1/4, 1/2, 1]
    param_grid = [{'kernel': ['rbf'], 'gamma': [10, 1, 0.1],
                   'C': [ 0.1, 1, 10]}]
    scoring = 'f1_micro'
    nfolds = 3
    kwargs = {"param_grid": param_grid, "scoring": scoring, "nfolds": nfolds}
    print ('training the daily svm model...\n')
    results1 = model.run("run_svm", nb_processes, train, test, type_='initial')
    results_opti1 = model.run("run_svm", nb_processes, train, test,
                              True, kwargs, type_='initial')
    train =[250, 500, 1000, 2000]
    test =[1/8, 1/4, 1/2, 1]
    results2 = model.run("run_svm", nb_processes, train, test, False, type_='initial')
    results_opti2 = model.run("run_svm", nb_processes, train, test, True, kwargs, type_='initial')
    results = results1 + results2
    results_opti = results_opti1 + results_opti2
    print('--------------------- \n')
    print(results)
    print('--------------------- \n')
    print(results_opti)
    print('--------------------- \n')
    print([a_i[0] - b_i[0] for a_i, b_i in zip(results, results_opti)])
    
	# test based on best training-testing windows
    
    print("Test set.....\n")
    # Separate in train and validation set and test set
    data_sliced = data.slice(int(0.8*size - 2000 - 77), size)
    data_sliced.add_indicators()
    model = svm.SVM(data_sliced)
    train = 2000
    test = 500
    print ('training the daily svm model...\n')
    results = model.run_svm(training_window = train, testing_window = test,
                        optimization = False)
    model.plot_return(title = 'Daily_SVM')
    print(results)

    # Prediction based on lag values of return only
    lags = 50
    data_sliced = data.slice(int(0.8*size - lags - 2000), size)
    data_sliced.add_lags(lags)
    model = svm.SVM(data_sliced)
    train = 2000
    test = 500
    print ('training the daily svm model...\n')
    results = model.run_svm(training_window = train, testing_window = test,
                            optimization = False)
    model.plot_return(title = 'Daily_SVM_lagged')
    print(results)
  
    # Hourly frequency
    data = d.data_currency(file) 
    data.resample('H')
    threshold = 0.0000
    data.define_threshold(threshold)
    # Separate in train and validation set and test set
    size = data.get_length()
    print(size)
  
    data_sliced = data.slice(0,int(0.8*size))
    data_sliced.add_indicators()
    model = svm.SVM(data_sliced)
    train =[250, 1000]
    test =[1/8, 1/4, 1/2, 1]
    print ('training the hourly svm model...\n')
    results1 = model.run("run_svm", nb_processes, train, test, False, type_='initial')
    train =[5000, 10000, 20000]
    test =[1/8, 1/4, 1/2, 1]
    print ('training the hourly svm model...\n')
    results2 = model.run("run_svm", nb_processes, train, test, False, type_='initial')
    results = results1 + results2
    print('--------------------- \n')
    print(results)
    print('--------------------- \n')
    
	# test based on best training-testing windows
    data_sliced = data.slice(int(0.8*size - 1000 - 77), size)
    data_sliced.add_indicators()
    model = svm.SVM(data_sliced)
    train = 1000
    test = 125
    print ('training the hourly svm model...\n')
    results = model.run_svm(training_window = train, testing_window = test,
                            optimization = False)
    model.plot_return(title = 'Hourly_SVM_TA')
    print(results)
    
    # Prediction based on lag values of return only
    lags = 50
    data_sliced = data.slice(int(0.8*size - lags - 1000), size)
    data_sliced.add_lags(lags)
    model = svm.SVM(data_sliced)
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