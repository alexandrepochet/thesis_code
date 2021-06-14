import numpy as np
from itertools import product, chain
from utils.utils import reset_keras
import pandas as pd
import copy
import pdb


class GridSearchCV2():

    def __init__(self, model, space, cv_model, scoring="accuracy", refit=True):
        self.model = model
        self.space = space
        self.cv_model = cv_model
        if scoring != "accuracy":
            print("Other scorings than accuracy are not supported \n")
        self.scoring = "accuracy"
        self.refit = refit

    def fit(self, X_train, Y_train=None, X_test=None, Y_test=None, verbose=False, neural=False):
        if len(list(self.space.values()))==1:
            flatten = chain.from_iterable
            params_list = list(flatten(list(self._product(list(self.space.values())))))
        else:
            params_list = list(self._product(list(self.space.values())))
        list_keys = list(self.space.keys())
        best_accuracy = 0
        best_params = None
        for params in params_list:
            acc_per_fold = []
            fold_no = 1
            if len(list_keys)==1:
                params_dic = {list_keys[0]: params}
            else:
                params_dic = {list_keys[i]: params[i] for i in range(0, len(list_keys) ) }
            print(params)
            i = 0
            for train, test in self.cv_model.split(X_train):
                reset_keras()
                i = i + 1
                if isinstance(X_train, pd.DataFrame):
                    x_train = X_train.iloc[train]
                    x_test = X_train.iloc[test]
                else:
                    x_train = X_train[train]
                    x_test = X_train[test]
                if Y_train is None:
                    y_train = None
                    y_test = None
                else:
                    if isinstance(Y_train, pd.DataFrame):
                        y_train = Y_train.iloc[train]
                        y_test = Y_train.iloc[test]
                    else:
                        y_train = Y_train[train]
                        y_test = Y_train[test]
                if neural is False:
                    message = self.model.fit(x_train, y_train, **params_dic)
                    if message is None:
                        score = self.model.evaluate(x_test, y_test)
                    else:
                        score = 0
                else:
                    history, message = self.model.fit(x_train, y_train, x_test, y_test,**params_dic)
                    if message is None:
                        #score = history.history['val_acc']
                        score = history.history['val_accuracy'] 
                    else:
                        epochs = params_dic["epochs"]
                        score = [0] * epochs
                acc_per_fold.append(score)
                fold_no = fold_no + 1
            if neural is True:
                try:
                    window = 5
                    index = np.argmax(self.moving_average(acc_per_fold[0], window))
                except:
                    index = np.argmax(acc_per_fold[0])
                acc_per_fold[0] = acc_per_fold[0][index]
            if verbose is True:
                print(acc_per_fold)
                print('------------------------------------------------------------------------')
                print('Average scores for all folds:')
                print(f'> Accuracy: {np.mean(acc_per_fold*100)} (+- {np.std(acc_per_fold*100)})')
                print('------------------------------------------------------------------------')
            if np.mean(acc_per_fold) > best_accuracy:
                best_accuracy = np.mean(acc_per_fold)
                best_params = params_dic
                if neural is True:
                    best_params["epochs"] = index + 1
        if self.refit is True:
            print(best_params)
            self.model.fit(X_train, Y_train, X_test=X_test, Y_test=Y_test, **best_params)
            return self.model, best_accuracy, best_params
        else:
            return best_accuracy, best_params

    def _product(self, list_params):
        result = [[]]
        for pool in list_params:
            result = [x+[y] for x in result for y in pool]
        for prod in result:
            yield list(prod)

    def moving_average(self, vec, window):
        ma = np.zeros((len(vec)))
        for i in range(int((window-1)/2),len(vec)-int((window-1)/2)):
            for j in range(0, window):
                ma[i] += vec[int(i+j-(window-1)/2)]/window
        return ma

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model

    def get_space(self):
        return self.space

    def set_space(self, spaec):
        self.space = space

    def get_cv_model(self):
        return self.cv_model

    def set_cv_model(self, cv_model):
        self.cv_model = cv_model

    def get_refit(self):
        return self.refit

    def set_refit(self, refit):
        self.refit = refit