# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 10:40:54 2019

@author: alexa
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 20:06:08 2019

@author: alexa
"""

import pandas as pd
import numpy as np
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from arch.univariate import ARX
import matplotlib.pyplot as plt
from arch.univariate import GARCH
from sklearn.metrics import classification_report, accuracy_score
import pypr
import warnings
import math 
from statistics import mean
import sys
from multiprocessing import Pool
import time
import pdb


def fxn():
    warnings.warn("", RuntimeWarning)

    
class TSA(object):
    
    """
    
    Time series analysis object. Fit an autoregressive model to predict
    data series prices. The model chooses the optimal lag and then predict
    price evolution for one observation going forward, the model being updated
    with most recent observation at every step. The parameters of the model are
    re-estimated based on a shifting window. The model then compares the 
    predicted performance, based on long, short or neutral positions generated 
    by the value of the prediction with a buy and hold strategy
    
    Attributes:
        df: The time series of returns 
        threshold: Threshold for the estimation of the long, short or neutral
        positions
        
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fxn()
    
    
    def __init__(self, df, threshold):
        
        self.df = df
        self.threshold = threshold
        self.accuracy = []
        self.null_accuracy = 0
        self.report = []
        self.eqCurves = []
        self.signal = []
        self.null_strategy_signal = []

        
    def time_series_analysis(self, training_window, testing_window, confidence,
                             plot):
        
        """
        
        Fit the model and estimate the returns of the prediction 
        and the return of the buy and hold strategy
        
        Args:
            training_window: The number of training observations
            testing_window: The number of prediction before the model
                            is updated with new observation
            confidence: Confidence interval for the strategy
            plot: Boolean indicating if plotting the autocorrelation of residuals
                  of the arch model
                            
        Returns:
            
        """
        warnings.filterwarnings("ignore")

        data, bid_ask = self.__initialize(training_window)

        if plot == True:
            self.__tsplot(data)

        n_epochs = math.ceil((data.size - training_window)/testing_window)

        for i in range(0, n_epochs):
            
            print ('epochs ' + str(i) + ' out of ' + str(n_epochs))
            
            training = 10000*data[(i*testing_window):min(i*testing_window+training_window, data.size)].values
            test_obs = 10000*self.df.Return[(i*testing_window+training_window):min((i+1)*testing_window+training_window, data.size)].values
            test = self.df.Direction[(i*testing_window+training_window):min((i+1)*testing_window+training_window, data.size)].values
            
            null_strategy = self.__get_null_strategy(test, test_obs)
            bic, order, mdl = self.__get_best_model(training)
            p_ = order[0]
            q_ = order[2]
            
            if confidence == True:
                model_fit, predicted = self.__Arx_Garch(training, p_, q_, test, test_obs, testing_window, i)
            else:
                model_fit, predicted = self.__Arima(mdl, training, p_, q_, test, test_obs, testing_window, i)
            
            res_ = model_fit.resid
            
            if null_strategy == 'up':
                self.null_strategy_signal.iloc[(i*testing_window + 1):(i*testing_window +
                                               len(test_obs) + 1)] = +1
            elif null_strategy == 'down':
                self.null_strategy_signal.iloc[(i*testing_window + 1):(i*testing_window +
                                               len(test_obs) + 1)] = -1
            else:
                self.null_strategy_signal.iloc[(i*testing_window + 1):(i*testing_window +
                                               len(test_obs) + 1)] = +0
            
            if plot == True:
                self.__tsplot(res_, lags = 30)
                res_squared = [x**2 for x in res]
                smt.graphics.plot_acf(res_squared)
                smt.graphics.plot_pacf(res_squared)
                h, pV, Q, cV = pypr.stattest.ljungbox.lbqtest(res_, range(1, 30), alpha=0.1)
                #print 'lag   p-value'
                for a in range(len(h)):
                    print ("%-2d %10.3f %s" % (a+1, pV[a], str(h[a])))
            
            self.accuracy.append(accuracy_score(test, predicted))
            self.report.append(classification_report(test, predicted))
        
        self.__calculate_returns(data, training_window, bid_ask)
        
        return (mean(self.accuracy), self.null_accuracy, self.eqCurves['Strategy'][len(self.eqCurves['Strategy'])-1],
                self.eqCurves['Buy and Hold'][len(self.eqCurves['Buy and Hold'])-1],
                self.eqCurves['Majority'][len(self.eqCurves['Majority'])-1], training_window, testing_window)
        

    def __initialize(self, training_window):

        """
        
        Initialisation
        
        Args:
            training_window: The training_window
            
        Returns:
            data: The return
            bid_ask: The bid ask spread

        """

        del self.accuracy[:]
        del self.report[:]
        del self.signal[:]
        del self.null_strategy_signal[:]
        self.null_accuracy = 0

        data = self.df['Return']    
        self.signal = 0*data[-(data.size - training_window + 1):]
        self.null_strategy_signal = 0*data[-(data.size - training_window + 1):]
        close = self.df['close'][-(data.size - training_window + 1):]
        bid_ask = self.df['close_bid_ask'][-(data.size - training_window + 1):]
        bid_ask = bid_ask.rolling(min_periods=1, window=2).sum()/(4*close)

        return data, bid_ask


    def __get_null_strategy(self, test, test_obs):

        """
        
        Determines the null strategy and calculate accuracy
        
        Args:
            test: The test data representing the direction (ground truth)
            test_obs: The return of the test data
            
        Returns:
            null_strategy: The majority class strategy

        """

        prop_up = len(test_obs[test == 'up']) / (len(test_obs)*1.)
        prop_down = len(test_obs[test == 'down']) / (len(test_obs)*1.)
        prop_stable = len(test_obs[test == 'stable']) / (len(test_obs)*1.)
        null_strategy = ''
        if prop_up >= prop_down and prop_up >= prop_stable:
            self.null_accuracy += prop_up / len(self.signal)*(len(test_obs)*1.)
            null_strategy = 'up'
        elif prop_down >= prop_up and prop_down >= prop_stable:
            self.null_accuracy += prop_down / len(self.signal)*(len(test_obs)*1.)
            null_strategy = 'down'
        else:
            self.null_accuracy += prop_stable / len(self.signal)*(len(test_obs)*1.)
            null_strategy = 'stable'
        
        return null_strategy


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


    def __get_best_model(self, TS):
        
        """
        
        Choose the optimal parameter of the autoregressive model, 
        based on bic criteria
        
        Args:
            TS: The time series 
            
        Returns:
            The optimal parameters
            
        """
    
        best_bic = np.inf 
        best_order = None
        best_mdl = None
    
        pq_rng = range(1, 5) # [1,2,3,4]
        d_rng = range(1) # [0], no differencing applied on log return
        for i in pq_rng:
            for d in d_rng:
                for j in pq_rng:
                    try:
                        tmp_mdl = smt.ARIMA(TS, order=(i,0,j)).fit(method='mle', trend='nc', disp = 0)
                        tmp_bic = tmp_mdl.bic
                        if tmp_bic < best_bic:
                            best_bic = tmp_bic
                            best_order = (i, d, j)
                            best_mdl = tmp_mdl
                    except: continue  
                     
        return best_bic, best_order, best_mdl
    
    
    def plot_return(self, title = None, dash = False):
        
        """
        
        Plot the return of the predicted strategy and the buy and hold
        strategy
        
        Args: 
            title: Title of the figure to save
            
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
        plt.savefig('C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/src/figures/' + str(title) + '.jpg', bbox_inches='tight', pad_inches=1)
        plt.close()
    
    
    def __tsplot(self, y, lags=None, figsize=(10, 8), style='bmh'):
    
        """
        
        Run autocorrelation, partial ann QQ plots for autocorrelation
        analysis
        
        Args:
            y: The time series or the residuals 
            lags: The number of lags to consider in the plots
            figsize: The size of the figures
            style: The style
            
        Returns:
            
        """
        
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
            
        with plt.style.context(style):    
            plt.figure(figsize=figsize)
            layout = (3, 2)
            ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
            acf_ax = plt.subplot2grid(layout, (1, 0))
            pacf_ax = plt.subplot2grid(layout, (1, 1))
            qq_ax = plt.subplot2grid(layout, (2, 0))
            pp_ax = plt.subplot2grid(layout, (2, 1))
            
            y.plot(ax=ts_ax)
            ts_ax.set_title('Time Series Analysis Plots')
            smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
            smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
            sm.qqplot(y, line='s', ax=qq_ax)
            qq_ax.set_title('QQ Plot')        
            scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)
    
            plt.tight_layout()

        return


    def __Arima(self, mdl, training, p_, q_, test, test_obs, testing_window, i):
        
        """
        
        Estimates the direction of the prediction for the Arima model mld 
        
        Args:
            mdl: The Arima model
            training: The training set
            p_: The optimal order (number of time lags) of the autoregressive model
            q_: The optimal order of the moving-average model
            test: The test data representing the direction (ground truth)
            test_obs: The return of the test data
            testing_window: The number of prediction before the model
                            is updated with new observations
            i: Integer representing the number of the epoch
                            
        Returns:
            model_fit: mld
            predicted: The predicted directions of the test set

        """

        model_fit = mdl
        res = model_fit.resid
        coef = model_fit.params
        mean_params = coef[0:p_]
        error_params = coef[p_ :(p_ + q_)]
        yhat = []
        res = model_fit.resid
        predicted  = []
        for t in range(len(test)):
            pred = 0
            length = len(training)
            lag = [training[i] for i in range(length-p_,length)]
            lag_res = [res[i] for i in range(length-q_,length)]
            for d in range(p_):
                pred += mean_params[d] * lag[p_-d-1]    
            for d in range(q_):
                pred += error_params[d] * lag_res[q_-d-1]
            pred = pred/10000
            obs = test_obs[t]
            yhat.append(pred)
            residual = obs-pred
            res = np.append(res, residual)
            training = np.append(training, obs)
            if pred > self.threshold :
                self.signal.iloc[i*testing_window + t + 1] = +1
                predicted.append('up')
            elif pred < self.threshold:
                self.signal.iloc[i*testing_window + t + 1] = -1
                predicted.append('down')
            else:
                self.signal.iloc[i*testing_window + t + 1] = 0
                predicted.append('stable')

        return model_fit, predicted


    def __Arx_Garch(self, training, p_, q_, test, test_obs, testing_window, i):
        
        """
        
        Fits and estimates the direction of the prediction for the Arx-Garch model 
        
        Args:
            training: The training set
            p_: The optimal order (number of time lags) of the autoregressive model
            q_: The optimal order of the moving-average model
            test: The test data representing the direction (ground truth)
            test_obs: The return of the test data
            testing_window: The number of prediction before the model
                            is updated with new observations
            i: Integer representing the number of the epoch
                            
        Returns:
            model_fit: mld
            predicted: The predicted directions of the test set

        """

        # Using Normal distribution
        model = ARX(training, lags = list(range(1, p_+1)))
        model.volatility = GARCH(p = p_, q = q_)
        model_fit = model.fit(update_freq=5, disp='off')
        coef = model_fit.params
        conditional_variance = coef[p_ + 1]
        mean_params = coef[1:(p_ + 1)]
        garch_params = coef[(p_ + 2):(2*p_ + q_ + 3)]
        yhat = []
        sigma_squared = model_fit.conditional_volatility
        res = model_fit.resid
        predicted  = []

        for t in range(len(test)):
            pred = coef[0]
            length = len(training)
            lag = [training[i] for i in range(length-p_,length)]
            lag_res = [res[i] for i in range(length-p_,length)]
            lag_sigma_squared = [sigma_squared[i] for i in range(length-q_,length)]
            
            for d in range(p_):
                pred += mean_params[d] * lag[p_-d-1]
                conditional_variance += garch_params[d] * lag_res[p_-d-1]
            for d in range(q_):
                conditional_variance += garch_params[p_ + d] * lag_sigma_squared[q_-d-1]
            
            lower_conf = (pred/10000)# - 0*conditional_variance)/10000
            upper_conf = (pred/10000)# + 0*conditional_variance)/10000
            obs = test_obs[t]
            yhat.append(pred)
            residual = obs-pred
            res = np.append(res, residual)
            sigma_squared = np.append(sigma_squared, conditional_variance)
            training = np.append(training, obs)
            if lower_conf > self.threshold :
                self.signal.iloc[i*testing_window + t + 1] = +1
                predicted.append('up')
            elif upper_conf < self.threshold:
                self.signal.iloc[i*testing_window + t + 1] = -1
                predicted.append('down')
            else:
                self.signal.iloc[i*testing_window + t + 1] = 0
                predicted.append('stable')   

        return model_fit, predicted     


    def run(self, training_windows, testing_windows, confidence, plot):
        
        """
        
        Run autocorrelation, partial ann QQ plots for autocorrelation
        analysis
        
        Args:
            y: The time series or the residuals 
            lags: The number of lags to consider in the plots
            figsize: The size of the figures
            style: The style
            
        Returns:
            
        """
        
        t = Pool(processes=4)
        rs = t.map(parallel_call, self.__prepare_call("time_series_analysis", 
                                                      training_windows, testing_windows, 
                                                      confidence, plot))
        t.close()

        return rs
    

    def __prepare_call(self, name, training_windows, testing_windows, confidence, plot):  # creates a 'remote call' package for each argument
        
        """
        
        Prepare relevant arguments for parallel call
        
        Args:
            name: The name of the method to call for parallelization 
            training_window: The number of training observations
            testing_window: The number of prediction before the model is updated 
                            with new observation
            confidence: Confidence interval for the strategy
            plot: Boolean indicating if plotting the autocorrelation of residuals
                  of the arch model
            
        Returns:
            
        """
        for train in training_windows:
            for testing in testing_windows:
                test = int(train*testing)
                yield [self.__class__.__name__, self.__dict__, name, [train,  test, 
                                                                      confidence, plot]]


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