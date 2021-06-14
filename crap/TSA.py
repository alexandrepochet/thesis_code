import pandas as pd
import numpy as np
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
import utils.utils as utils
from arch.univariate import ARX, GARCH
import matplotlib.pyplot as plt 
from sklearn.metrics import classification_report, accuracy_score
from initial_analysis.algorithm import algorithm
from utils.parallel import parallel
import pypr
import warnings
import math 
from statistics import mean
import sys
from multiprocessing import Pool
import time
import pdb
 
 
class TSA(algorithm, parallel): 
    """  
    Time series analysis object. Fit an autoregressive model to predict
    data series prices. The model chooses the optimal lag and then predict
    price evolution for one observation going forward, the model being updated
    with most recent observation at every step. The parameters of the model are
    re-estimated based on a shifting window. The model then compares the 
    predicted performance, based on long, short or neutral positions generated 
    by the value of the prediction with a buy and hold strategy
    
    Attributes:
        data: A data object with relevant data 
        threshold: Threshold for the estimation of the long, short or neutral
        positions
        
    """ 
    warnings.filterwarnings("ignore")
    # Defining module name as global variable for access in parent class (parallel)
    mod = sys.modules[__name__]

    def __init__(self, data, threshold): 
        super().__init__(data)
        self.threshold = threshold
        self.report = [] 
       
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
        Returns: A lot        
        """ 
        data, bid_ask = self._initialize(training_window)

        if plot == True:
            self._tsplot(data)

        n_epochs = math.ceil((data.size - training_window)/testing_window)
        test_up = 0
        test_up_length = 0
        for i in range(0, n_epochs):
            
            print ('epochs ' + str(i) + ' out of ' + str(n_epochs))
            
            training = 10000*data.iloc[(i*testing_window):min(i*testing_window+training_window, data.size)].values
            test_obs = 10000*data.iloc[(i*testing_window+training_window):min((i+1)*testing_window+training_window, data.size)].values
            test = self.data.get_direction().iloc[(i*testing_window+training_window):min((i+1)*testing_window+training_window, data.size)].values
            
            null_strategy = self._get_null_strategy(test_obs, test)
            bic, order, mdl = self._get_best_model(training)
            p_ = order[0]
            q_ = order[2]
            
            if confidence == True:
                model_fit, predicted = self._arx_garch(training, p_, q_, test, test_obs, testing_window, i)
            else:
                model_fit, predicted = self._arima(mdl, training, p_, q_, test, test_obs, testing_window, i)
            
            res_ = model_fit.resid
            
            if null_strategy == self.up:
                self.null_strategy_signal.iloc[(i*testing_window + 1):(i*testing_window +
                                               len(test_obs) + 1)] = +1
            elif null_strategy == self.down:
                self.null_strategy_signal.iloc[(i*testing_window + 1):(i*testing_window +
                                               len(test_obs) + 1)] = -1 
            else:
                self.null_strategy_signal.iloc[(i*testing_window + 1):(i*testing_window +
                                               len(test_obs) + 1)] = +0
            
            if plot is True:
                self._tsplot(res_, lags = 30)
                res_squared = [x**2 for x in res]
                smt.graphics.plot_acf(res_squared)
                smt.graphics.plot_pacf(res_squared)
                h, pV, Q, cV = pypr.stattest.ljungbox.lbqtest(res_, range(1, 30), alpha=0.1)
                #print 'lag   p-value'
                for a in range(len(h)):
                    print ("%-2d %10.3f %s" % (a+1, pV[a], str(h[a])))
            
            self.accuracy.append(accuracy_score(test, predicted))
            self.report.append(classification_report(test, predicted))
            test_up += len(test[test == self.up])
            test_up_length += len(test)
        
        self._calculate_returns(data, training_window, bid_ask)
        accuracy_BH = test_up/test_up_length
        return (mean(self.accuracy), accuracy_BH, self.null_accuracy,
                self.eqCurves['Strategy'][len(self.eqCurves['Strategy'])-1],
                self.eqCurves['Buy and Hold'][len(self.eqCurves['Buy and Hold'])-1],
                self.eqCurves['Majority'][len(self.eqCurves['Majority'])-1], training_window, testing_window)
 
    def _get_best_model(self, TS):
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

    def _tsplot(self, y, lags=None, figsize=(10, 8), style='bmh'):
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

    def _arima(self, mdl, training, p_, q_, test, test_obs, testing_window, i):
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
                predicted.append(self.up)
            elif pred < -self.threshold:
                self.signal.iloc[i*testing_window + t + 1] = -1
                predicted.append(self.down)
            else:
                self.signal.iloc[i*testing_window + t + 1] = 0
                predicted.append(self.stable)
        return model_fit, predicted

    def _arx_garch(self, training, p_, q_, test, test_obs, testing_window, i):
        
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
            
            #lower_conf = (pred/10000)# - 0*conditional_variance)/10000
            #upper_conf = (pred/10000)# + 0*conditional_variance)/10000
            pred = pred/10000
            obs = test_obs[t]
            yhat.append(pred)
            residual = obs-pred
            res = np.append(res, residual)
            sigma_squared = np.append(sigma_squared, conditional_variance)
            training = np.append(training, obs)
            if pred > self.threshold :
                self.signal.iloc[i*testing_window + t + 1] = +1
                predicted.append(self.up)
            elif pred < -self.threshold:
                self.signal.iloc[i*testing_window + t + 1] = -1
                predicted.append(self.down)
            else:
                self.signal.iloc[i*testing_window + t + 1] = 0
                predicted.append(self.stable)   
        return model_fit, predicted

    def _initialize(self, training_window):
        """  
        Initialisation
        
        Args:
            training_window: The training_window   
        Returns:
            data: The return
            bid_ask: The bid ask spread
        """
        del self.accuracy[:]
        self.null_accuracy = 0
        data = self.data.get_Return()  
        self.signal = 0*data.iloc[-(data.size - training_window + 1):]
        self.null_strategy_signal = 0*data.iloc[-(data.size - training_window + 1):]
        close = self.data.get_close().iloc[-(data.size - training_window + 1):].values
        bid_ask = self.data.get_close_bid_ask().iloc[-(data.size - training_window + 1):]
        bid_ask = bid_ask.rolling(min_periods=1, window=2).sum()/(4*close)

        return data, bid_ask

    def _data_preparation(self, X, y, training_window, testing_window, size, i):
        pass

    def _calculate_signal(self, y_test, y_pred, testing_window, null_strategy, i):
        pass
