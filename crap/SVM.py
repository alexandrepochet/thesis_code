import pandas as pd
import seaborn; seaborn.set()
import matplotlib.pyplot as plt
import utils.utils as utils
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import preprocessing as pre
import math
import warnings
import numpy as np
import pdb
import sys
from multiprocessing import Pool
import time
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from crap.algorithm import algorithm
from utils.parallel import parallel
np.random.seed(42)
 

class SVM(algorithm, parallel):
    """ 
    Support vector machine object. Fit a support vector machine model to
    predict data series prices. The model can choose the optimal parameters
    for the prediction, based on a parameters grid, or fit a model based on 
    given parameters. The model is being updated with most recent observations
    based on a shifting window. The model then compares the predicted 
    performance, based on long, short or neutral positions generated 
    by the value of the prediction with a buy and hold strategy.
    
    Attributes:
        data: A data object containing the data of interest     
    """
    #Defining module name as global variable for access in parent class (parallel)
    mod = sys.modules[__name__]

    def run_svm(self, training_window, testing_window, optimization = False,  
                 kwargs = [], C = 1, kernel = 'rbf', gamma = 1, degree = 1):
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
        X, y, bid_ask, size, Return = self._initialize(training_window, shift = True)
        n_epochs = math.ceil((y.size - training_window)/testing_window)
        test_up = 0
        test_up_length = 0

        for i in range(0, n_epochs):

            # epoch number
            print (str(i) + ' out of ' + str(n_epochs - 1) )

            # data preparation
            X_train, y_train, X_test, y_test = self._data_preparation(X, y, training_window, testing_window, size, i)

            # majority class strategy 
            null_strategy = self._get_null_strategy(X_test, y_test)

            # features preparation 
            model, nb_comp, X_train_scaled, X_test_scaled = self._features_preparation(X_train, X_test, y_train) 

            # features selection
            svclassifier_best, index_best = self._best_selection(model, X_train_scaled, y_train, nb_comp, optimization,
                                                                 C, kernel, gamma, degree, kwargs)

            # prediction based on the selection of best features
            X_test_scaled = X_test_scaled[:,index_best]
            y_pred = svclassifier_best.predict(X_test_scaled)

            # accuracy
            self.accuracy.append(accuracy_score(y_test, y_pred))

            # calculate the signal of the strategies
            self._calculate_signal(y_test, y_pred, testing_window, null_strategy, i)
            test_up += len(y_test[y_test == self.up])
            test_up_length += len(y_test)
        
        # calculates the cumulative return of the strategies   
        self._calculate_returns(y, training_window, bid_ask, Return)
        accuracy_BH = test_up/test_up_length
    
        return (np.mean(self.accuracy), accuracy_BH, self.null_accuracy, self.eqCurves['Strategy'][len(self.eqCurves['Strategy'])-1], 
                self.eqCurves['Buy and Hold'][len(self.eqCurves['Buy and Hold'])-1],
                self.eqCurves['Majority'][len(self.eqCurves['Majority'])-1], training_window, testing_window)

    def _features_preparation(self, X_train, X_test, y_train):
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

    def _best_selection(self, model, X_train_scaled, y_train, nb_comp, optimization,
                         C, kernel, gamma, degree, kwargs):    
        """       
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
                svclassifier = self._svc_param_selection(X_train_scaled_temp, y_train,
                                                         param_grid, scoring, nfolds)
                  
            y_pred_train = svclassifier.predict(X_train_scaled_temp)          
            accuracy = accuracy_score(y_train, y_pred_train)

            if accuracy > best_accuracy:
                best_accuracy = accuracy 
                svclassifier_best = svclassifier
                index_best = index    

        return svclassifier_best, index_best

    def _svc_param_selection(self, X, y, param_grid, scoring, nfolds): 
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
