from sklearn.metrics import classification_report, accuracy_score
from utils.algorithm import algorithm
from sklearn import svm
from sklearn import preprocessing as pre
from sklearn.ensemble import BaggingClassifier
from utils.utils import add_lags
from copy import deepcopy
import pandas as pd
import numpy as np
import pdb


class SVC(algorithm): 
    
    def __init__(self):
        super().__init__() 
        self.model = None
        self.scaler = None

    def fit(self, X_train, Y_train,  X_val=None, Y_val=None, X_test=None, Y_test=None, C=1, gamma=1, lags=None):
        try:
            X = deepcopy(X_train)
            if lags is not None:
                X = add_lags(X, lags, True)
                X = X.fillna(X.mean())
                X = np.asarray(X)
            Y = Y_train
            self.init_scale(X)
            X_normalized = self.scale(X)
            svc_model = svm.SVC(C=C, kernel='rbf', gamma=gamma, probability=False, max_iter=100000, tol=0.001)
            self.model = BaggingClassifier(base_estimator=svc_model, n_estimators=1, bootstrap=False, n_jobs=-1)
            self.model.fit(X_normalized, Y)
            return None
        except: 
            return "issue"

    def predict(self, X_test, normalize=True):
        x_test = X_test
        if normalize is True:
            x_test_normalized = self.scale(x_test)
        else:
            x_test_normalized = x_test
        y_hat = self.model.predict(x_test_normalized)
        return y_hat

    def evaluate(self, X_test, Y_test, normalize=True):
        y_hat = self.predict(X_test, normalize)
        if y_hat is None:
            score = 0
        else:
            score = accuracy_score(Y_test, y_hat)
        return score

    def init_scale(self, X):
        self.scaler = pre.StandardScaler().fit(X)

    def scale(self, X):
        # standard normalization
        return self.scaler.transform(X)                        

 