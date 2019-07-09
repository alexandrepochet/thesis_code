# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 19:46:49 2019

@author: alexa
"""

import pandas as pd
import seaborn; seaborn.set()
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import preprocessing as pre
import currency_preprocess as c
import math
import warnings
import numpy as np
import pdb
import sys
from multiprocessing import Pool
import time
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
np.random.seed(42)


class SVM(object):
    
    """
    
    Support vector machine object. Fit a support vector machine model to
    predict data series prices. The model can choose the optimal parameters
    for the prediction, based on a parameters grid, or fit a model based on 
    given parameters. The model is being updated with most recent observations
    based on a shifting window. The model then compares the predicted 
    performance, based on long, short or neutral positions generated 
    by the value of the prediction with a buy and hold strategy.
    
    Attributes:
        df: The time series of returns 
        threshold: Threshold for the estimation of the long, short or neutral
        positions
        
    """
    
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore",category=DeprecationWarning)
      warnings.filterwarnings(action="ignore",category=FutureWarning)
        
    def __init__(self, df, threshold):
        
        self.df = df
        self.threshold = threshold
        self.accuracy = []
        self.eqCurves = []
        self.signal = []
        self.null_strategy_signal = []


    def run_svm(self, training_window, testing_window, optimization = False,  
                 C = 1, kernel = 'rbf', gamma = 1, degree = 1, kwargs = []):
        
        """
        
        Fit the model and estimate the returns of the prediction
        and the return of the buy and hold strategy
        
        Args:
            training_window: The number of training observations
            testing_window: The number of prediction before the model
                            is updated with new observation
            optimization: boolean indicator for optimization based on a 
                          parameters grid
            C: Penalty parameter C of the error term
            kernel: Specifies the kernel type to be used in the algorithm. 
                    It must be one of linear, poly, rbf, sigmoid, precomputed
                    or a callable. If none is given, rbf will be used
            gamma: Kernel coefficient for rbf, poly and sigmoid
            degree: Degree of the polynomial kernel function
            **kwargs: Parameters for the grid search, in case the optimization
                      indicator is True. Must contain param_grid, scoring and 
                      nfolds. param_grid contains a specification for the grid,
                      scoring is the scoring function used for choosing the 
                      optimal set of parameters contained in param_grid and
                      nfolds is the validation scheme
                      
        Returns:
        
        """
        
        # initialization
        X, y, bid_ask, size = self.__initialize(training_window)
        
        n_epochs = math.ceil((size - training_window)/testing_window)
        null_accuracy_tot = 0
         
        for i in range(0, n_epochs):

            # epoch number
            print (str(i) + ' out of ' + str(n_epochs - 1) )

            # data preparation
            X_train, y_train, X_test, y_test = self.__data_preparation(X, y, training_window, testing_window, size, i)

            # majority class strategy 
            null_strategy, null_accuracy = self.__get_null_strategy(X_test, y_test)

            # features preparation 
            model, nb_comp, X_train_scaled, X_test_scaled = self.__features_preparation(X_train, X_test, y_train) 

            # features selection
            svclassifier_best, index_best = self.__best_selection(model, X_train_scaled, y_train, nb_comp, optimization,
                                                                  C, kernel, gamma, degree, probability, kwargs)

            # prediction based on the selection of best features
            X_test_scaled = X_test_scaled[:,index_best]
            y_pred = svclassifier_best.predict(X_test_scaled)

            # accuracy
            self.accuracy.append(accuracy_score(y_test, y_pred))
            null_accuracy_tot += null_accuracy / len(self.signal)*(len(y_test)*1.)

            # calculate the signal of the strategies
            self.__calculate_signal(y_test, testing_window, null_strategy, i)
        
        # calculates the cumulative return of the strategies   
        self.__calculate_returns(data, training_window, bid_ask)
        
        return (np.mean(self.accuracy), null_accuracy_tot, self.eqCurves['Strategy'][len(self.eqCurves['Strategy'])-1], 
                self.eqCurves['Buy and Hold'][len(self.eqCurves['Buy and Hold'])-1],
                self.eqCurves['Majority'][len(self.eqCurves['Majority'])-1], training_window, testing_window)


    def __calculate_signal(self, y_test, y_pred, testing_window, null_strategy, i):

        """
        
        Calculate the signal of the strategies
        
        Args:
            y_test: The test data (direction)
            y_pred: The predicted directions
            testing_window: The size of testing window
            null_strategy: The majority class of the epoch
            i: The number of the epoch
            
        Returns:

        """    
        
        for j in range(0, y_test.size):
            if y_pred[j] == 'up':
                self.signal.iloc[i*testing_window + j + 1] = +1
            elif y_pred[j] == 'down':
                self.signal.iloc[i*testing_window + j + 1] = -1
            else:
                self.signal.iloc[i*testing_window + j + 1] = 0

        if null_strategy == 'up':
            self.null_strategy_signal.iloc[(i*testing_window + 1):(i*testing_window +
                                            y_test.size + 1)] = +1
        elif null_strategy == 'down':
            self.null_strategy_signal.iloc[(i*testing_window + 1):(i*testing_window +
                                           y_test.size + 1)] = -1
        else:
            self.null_strategy_signal.iloc[(i*testing_window + 1):(i*testing_window +
                                           y_test.size + 1)] = +0


    def __best_selection(self, model, X_train_scaled, y_train, nb_comp, optimization,
                         C, kernel, gamma, degree, kwargs):    

        """
        
        Initialisation
        
        Args:
            model: The ExtraTrees model
            X_train_scaled: The training features after normalization and pca
            y_train: The training set (direction)
            nb_comp: The number of components of the pca model
            optimization: boolean indicator for optimization based on a 
                          parameters grid
            C: Penalty parameter C of the error term
            kernel: Specifies the kernel type to be used in the algorithm. 
                    It must be one of linear, poly, rbf, sigmoid, precomputed
                    or a callable. If none is given, rbf will be used
            gamma: Kernel coefficient for rbf, poly and sigmoid
            degree: Degree of the polynomial kernel function
            kwargs: Parameters for the grid search, in case the optimization
                    indicator is True. Must contain param_grid, scoring and 
                    nfolds. param_grid contains a specification for the grid,
                    scoring is the scoring function used for choosing the 
                    optimal set of parameters contained in param_grid and
                    nfolds is the validation scheme
            
        Returns:
            svclassifier_best: The best model
            index_best: The index of the features leading to the best model

        """
        
        best_accuracy = 0
        svclassifier_best = None 
        index_best = 0
        
        for k in range(min(nb_comp, 3), min(nb_comp + 1, 30)):
            index = np.argsort(-model.feature_importances_)[0:k]
            X_train_scaled_temp = X_train_scaled[:,index]
                  
            if optimization is False:
                if kernel == 'poly':
                    svclassifier = svm.SVC(C = C, kernel=kernel, gamma = gamma,
                                           degree = degree, probability = False)  
                else:
                    svclassifier = svm.SVC(C = C, kernel=kernel, gamma = gamma, probability = False)

                svclassifier = svclassifier.fit(X_train_scaled_temp, y_train)
            else:
                param_grid = kwargs['param_grid']
                scoring = kwargs['scoring']
                nfolds = kwargs['nfolds']
                svclassifier = self.__svc_param_selection(X_train_scaled_temp, y_train,
                                                          param_grid, scoring, nfolds)
                  
            y_pred_train = svclassifier.predict(X_train_scaled_temp)          
            accuracy = accuracy_score(y_train, y_pred_train)

            if accuracy > best_accuracy:
                best_accuracy = accuracy 
                svclassifier_best = svclassifier
                index_best = index    

        return svclassifier_best, index_best


    def __initialize(self, training_window):

        """
        
        Initialisation
        
        Args:
            training_window: The training_window
            
        Returns:
            X: The data with the features
            y: The direction of the data
            bid_ask: The bid ask spread
            size: The size of the data

        """

        del self.accuracy[:]
        del self.signal[:]
        del self.null_strategy_signal[:]
        y = self.df['Direction']   
        X = self.df.drop(['Direction', 'open_bid_ask', 'close_bid_ask'], axis = 1)
        X = X.shift(1)
        X = X.iloc[1:]
        y = y.iloc[1:]
        size = y.size
        self.signal = 0*self.df.Return[-(size - training_window + 1):]
        self.signal.index = self.df.index[-(size - training_window + 1):]
        self.null_strategy_signal = 0*self.df.Return[-(size - training_window + 1):] 
        self.null_strategy_signal.index = self.df.index[-(size - training_window + 1):]
        close = self.df['close'][-(size - training_window + 1):]
        close.index = self.df.index[-(size - training_window + 1):]
        bid_ask = self.df['close_bid_ask'][-(size - training_window + 1):]
        bid_ask.index = self.df.index[-(size - training_window + 1):]
        bid_ask = bid_ask.rolling(min_periods=1, window=2).sum()/(4*close)

        return X, y, bid_ask, size


    def __data_preparation(self, X, y, training_window, testing_window, size, i):

        """
        
        Data preparation
        
        Args:
            X: The test data (features)
            y: The direction of the test data
            training_window: The size of the training window
            testing_window: The size of the testing window
            size: The size of the data
            i: The number of the epoch
            
        Returns:
            X_train: The training features
            y_train: The direction of the training data
            X_test: The test features
            y_test: The direction of the test data

        """

        X_train = X[(i*testing_window):min(i*testing_window+
                    training_window, size)].values
        X_test = X[(i*testing_window+training_window):min((i+1)
                    *testing_window+training_window, size)].values
        y_train = y[(i*testing_window):min(i*testing_window+
                    training_window, size)].values
        y_test = y[(i*testing_window+training_window):min((i+1)*
                    testing_window+training_window, size)].values

        # remove class with less than 10 observations
        unique, counts = np.unique(y_train, return_counts=True)
        _class = unique[np.argwhere(counts < 10)]
        index = np.where(y_train==_class)
        y_train = np.delete(y_train, index)
        X_train = np.delete(X_train, index, axis = 0)
        index = np.where(y_test==_class)
        y_test = np.delete(y_test, index)
        X_test = np.delete(X_test, index, axis = 0)

        return X_train, y_train, X_test, y_test


    def __get_null_strategy(self, X_test, y_test):

        """
        
        Determines the null strategy and calculate accuracy
        
        Args:
            X_test: The test data (features)
            y_test: The direction of the test data
            
        Returns:
            null_strategy: The majority class strategy
            null_accuracy: The majority class accuracy

        """

        #majority class strategy 
        prop_up = len(X_test[y_test == 'up']) / (len(X_test)*1.)
        prop_down = len(X_test[y_test == 'down']) / (len(X_test)*1.)
        prop_stable = len(X_test[y_test == 'stable']) / (len(X_test)*1.)
        null_strategy = ''
        null_accuracy = 0
        if prop_up >= prop_down and prop_up >= prop_stable:
            null_accuracy = prop_up
            null_strategy = 'up'
        elif prop_down >= prop_up and prop_down >= prop_stable:
            null_accuracy = prop_down
            null_strategy = 'down'
        else:
            null_accuracy = prop_stable
            null_strategy = 'stable'
        
        return null_strategy, null_accuracy


    def __calculate_returns(self, data, training_window, bid_ask):

        """
        
        Calculate the return of the different strategies
        
        Args:
            data: The time series return
            training_window: The size of the training window
            bid_ask: The bid ask spread
            
        Returns:
            
        """

        returns = pd.DataFrame(index = self.signal.index, 
                               columns=['Buy and Hold', 'Strategy', 'Majority'])
        returns['Buy and Hold'] = data[-(data.size - training_window + 1):]
        cost = self.signal.diff()
        cost.iloc[0] = 0
        cost = np.abs(cost)* bid_ask
        returns['Strategy'] = self.signal*returns['Buy and Hold'] - cost
        returns['Majority'] = self.null_strategy_signal*returns['Buy and Hold']
        returns['Buy and Hold'].iloc[0] = 0
        returns['Strategy'].iloc[0] = 0
        returns['Majority'].iloc[0] = 0
        self.eqCurves = pd.DataFrame(index = self.signal.index, 
                                     columns=['Buy and Hold', 'Strategy', 'Majority'])
        self.eqCurves['Buy and Hold']=returns['Buy and Hold'].cumsum()+1
        self.eqCurves['Strategy'] = returns['Strategy'].cumsum()+1
        self.eqCurves['Majority'] = returns['Majority'].cumsum()+1


    def __features_preparation(self, X_train, X_test, y_train):

        """
        
        Prepares features based on normalisation, pca and scoring of the features
        
        Args:
            X_train: The training features
            X_test: The test features
            y_train: The test data (direction)
            
        Returns:
            model: The Extra Trees Classifier model
            nb_comp: The minimum number of components for which cumulative variance is higher than 95%
            X_train_scaled: The features of the training set, normalized and projected on the pca space
            X_test_scaled: The features of the test set, normalized and projected on the pca space
            
        """

        # standard normalization
        scaler = pre.StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        # pca
        pca_results = PCA().fit(X_train_scaled);
        var = pca_results.explained_variance_ratio_.cumsum()
        nb_comp = [ n for n,a in enumerate(var) if a > 0.95][0]
        pca = PCA(nb_comp + 1)
        pca.fit(X_train_scaled)
        X_train_scaled = pca.transform(X_train_scaled)
        X_test_scaled = scaler.transform(X_test)                        
        X_test_scaled = pca.transform(X_test_scaled)
        # feature extraction
        model = ExtraTreesClassifier(n_estimators = 100, criterion = 'entropy')
        model.fit(X_train_scaled, y_train)

        return model, nb_comp, X_train_scaled, X_test_scaled


    def plot_return(self, title = None, dash = False):
        
        """
        
        Plot the return of the predicted strategy and the buy and hold
        strategy
        
        Args:
            
        Returns:
            
        """
        self.eqCurves['Strategy'].plot(figsize=(10,8))
        self.eqCurves['Buy and Hold'].plot()
        if dash == True:
            self.eqCurves['Majority'].plot(dashes = [2,6])
        else:
            self.eqCurves['Majority'].plot()
        plt.ylabel('Index value')
        plt.xlabel('Date')
        plt.legend()
        plt.savefig('C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/src/preprocess/figures/' + str(title) + '.jpg', pad_inches=1)
        plt.close()



    def __svc_param_selection(self, X, y, param_grid, scoring, nfolds):
        
        """
        
        Return the best svm model based on a parameters grid, a scoring function
        and a validation scheme
        
        Args:
            X: Explanatory variables
            y: Predictive variable
            param_grid: The grid of parameters for the search
            scoring: The scoring function
            nfolds: The validation scheme
            
        Returns:
            The best fitted model
            
        """
        
        grid_search = GridSearchCV(svm.SVC(probability = False), param_grid, cv=nfolds, scoring = scoring)
        grid_search.fit(X, y)
        
        return grid_search  


    def run(self, training_windows, testing_windows, optimization = False, 
            C = 1, kernel = 'rbf', gamma = 1, degree = 1, kwargs = []):

        """
        
        Run parallel 
        
        Args:
            training_window: The number of training observations
            testing_window: The number of prediction before the model is updated 
                            with new observations
            optimization: Boolean indicating if optimization of the parameters should be performed
            C: Penalty parameter of the SVM
            kernel: Kernel of the SVM
            gamma: Parameter of the RBF kernel
            degree: Degree of the polynomial kernel
            kwargs: Range of parameters in case optimization of the parameters is performed
            
        Returns:
            
        """
        
        t = Pool(processes=4)
        rs = t.map(parallel_call, self.__prepare_call("run_svm", training_windows, testing_windows, optimization, 
                                                     C, kernel, gamma, degree, kwargs))
        t.close()
        return rs
    

    def __prepare_call(self, name, training_windows, testing_windows, optimization, 
                     C, kernel, gamma, degree, kwargs): 
        
        """
        
        Prepare relevant arguments for parallel call
        
        Args:
            name: The name of the method to call for parallelization 
            training_window: The number of training observations
            testing_window: The number of prediction before the model is updated 
                            with new observations
            optimization: Boolean indicating if optimization of the parameters should be performed
            C: Penalty parameter of the SVM
            kernel: Kernel of the SVM
            gamma: Parameter of the RBF kernel
            degree: Degree of the polynomial kernel
            kwargs: Range of parameters in case optimization of the parameters is performed

        Returns:
            
        """
        for train in training_windows:
            for testing in testing_windows:
                test = int(train*testing)
                yield [self.__class__.__name__, self.__dict__, name, [train, test, optimization, 
                                                                      C, kernel, gamma, degree, kwargs]]


def parallel_call(params):   
    
    """
        
    A helper for calling 'remote' instances
        
    Args:
        params: list containing the class type, the object, the method to call
                for parallilization and the arguments of the method
            
    Returns:
        method(*args): expand arguments, call our method and return the result
        
    """
        
    cls = getattr(sys.modules[__name__], params[0])  # get our class type
    instance = cls.__new__(cls)  # create a new instance without invoking __init__
    instance.__dict__ = params[1]  # apply the passed state to the new instance
    method = getattr(instance, params[2])  # get the requested method
    args = params[3] if isinstance(params[3], (list, tuple)) else [params[3]]
      
    return method(*args)  # expand arguments, call our method and return the result