from sklearn.metrics import classification_report, accuracy_score
from utils.algorithm import algorithm
from itertools import chain
import statsmodels.api as sm
import pandas as pd
import numpy as np
import pdb


class Arima(algorithm): 
    
    def __init__(self):
        super().__init__() 
        self.X_train = None
        self.exog_train = None
        self.params = None
        self.order = None

    def fit(self, X_train, exog=None, Y_train=None, X_val=None, Y_val=None, X_test=None, Y_test=None, order=(2,0,1), trend=None, lags=None):
        try:
            mod = sm.tsa.SARIMAX(X_train, exog=exog, order=order, trend=trend)
            fit_res = mod.fit(disp=False)
            self.params = fit_res.params
            self.X_train = X_train
            self.exog_train = exog
            self.order = order
            self.trend = trend
            return None
        except: 
            return "issue"

    def evaluate(self, X_test, exog=None, Y_test=None):
        y_hat, dir_hat = self.predict(X_test, exog=exog)
        if y_hat is None:
            score = 0
        else:
            score = accuracy_score(self.convert(X_test.values), dir_hat)
        return score

    def predict(self, X_test, exog=None):
        y_hat = None
        dir_hat = None
        X = pd.concat([self.X_train, X_test])
        if exog is not None:
            exog_test = pd.concat([self.exog_train, exog])
        try:
            mod = sm.tsa.SARIMAX(X, exog=exog_test, order=self.order, trend=self.trend)
            res = mod.filter(self.params)
            y_hat = res.get_prediction().predicted_mean.loc[X_test.index]
            dir_hat = self.convert(y_hat.values)
        except: 
            return
        return y_hat, dir_hat

    def get_X_train(self):
        return self.X_train

    def get_params(self):
        return self.params

    def get_order(self):
        return self.order

    def get_trend(self):
        return self.trend


