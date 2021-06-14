import numpy as np
from itertools import product, chain
from utils.utils import reset_keras
from utils.callback import callback
from utils.utils import add_lags
import pandas as pd
import copy
import pdb


class GridSearchCV_custom():

    def __init__(self, model, space, cv_model, scoring="accuracy", refit=True):
        self.model = model
        self.space = space
        self.cv_model = cv_model
        if scoring != "accuracy":
            print("Other scorings than accuracy are not supported \n")
        self.scoring = "accuracy"
        self.refit = refit

    def fit(self, X_train, Y_train=None, Exog_train=None, X_test=None, Y_test=None, verbose=False, previous_training=None, 
            previous=None, previous_accuracy_training=None, previous_accuracy=None, neural=False, SVC_sentiment=False, history_index=None):
        if len(list(self.space.values()))==1:
            flatten = chain.from_iterable
            params_list = list(flatten(list(self._product(list(self.space.values())))))
        else:
            params_list = list(self._product(list(self.space.values())))
        list_keys = list(self.space.keys())
        best_loss = np.inf
        best_accuracy = 0
        best_params = None
        param_iteration = 0
        smoothed_acc_training_bool = False
        if neural is True:
            epochs = self.space['epochs'][0]
            acc_per_fold_total = np.zeros((len(params_list),3,epochs))
            acc_per_fold_training_total = np.zeros((len(params_list),3,epochs))
            loss_per_fold_total = np.zeros((len(params_list),3,epochs))
            loss_per_fold_training_total = np.zeros((len(params_list),3,epochs))
        else:
            acc_per_fold_total = np.zeros((len(params_list),3))
            loss_per_fold_total = np.zeros((len(params_list),3))
        for params in params_list:
            acc_per_fold = []
            acc_per_fold_training = []
            loss_per_fold = []
            loss_per_fold_training = []
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
                if (i==3) or previous is None:
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
                    if Exog_train is not None:
                        lags = params_dic["lags"]
                        exog_train = Exog_train.iloc[train,0:lags]
                        exog_test = Exog_train.iloc[test,0:lags]
                    if neural is False:
                        if Exog_train is None:
                            message = self.model.fit(x_train, y_train, **params_dic)
                            if message is None:
                                if SVC_sentiment is True:
                                    lags = params_dic["lags"]
                                    x_test = pd.concat([x_train.iloc[len(x_train)-lags:], x_test])
                                    x_test = add_lags(x_test, lags, True)
                                    x_test = x_test.iloc[lags:]
                                    x_test = np.asarray(x_test)
                                score = self.model.evaluate(x_test, y_test)
                            else:
                                score = 0
                        else:
                            message = self.model.fit(x_train, exog=exog_train, **params_dic)
                            if message is None:
                                score = self.model.evaluate(x_test, exog=exog_test)
                    else:
                        if history_index is not None:
                            self.model.set_indices(X_train.index[train], history_index)
                        history, message = self.model.fit(x_train, y_train, x_test, y_test,**params_dic)
                        if message is None:
                            score = history.history['val_loss']
                            score_training = history.history['loss'] 
                            score_accuracy = history.history['val_custom_accuracy']
                            score_accuracy_training = history.history['custom_accuracy'] 
                        else:
                            epochs = params_dic["epochs"]
                            score = [np.inf] * epochs
                            score_training = [np.inf] * epochs
                            score_accuracy = [0] * epochs
                            score_accuracy_training = [0] * epochs
                else:
                    if i!=3:
                        score = previous[param_iteration,i]
                        score_accuracy = previous_accuracy[param_iteration,i]
                        if neural is True:
                            score_training = previous_training[param_iteration,i]
                            score_accuracy_training = previous_accuracy_training[param_iteration,i]
                    else:
                        continue
                loss_per_fold.append(score)
                acc_per_fold.append(score_accuracy)
                if neural is True:
                    acc_per_fold_training.append(score_accuracy_training)
                    loss_per_fold_training.append(score_training)
                fold_no = fold_no + 1
            if neural is True:
                window = 11
                somme_val = np.add(np.add(loss_per_fold[0],loss_per_fold[1]),loss_per_fold[2])/3
                somme_acc_val = np.add(np.add(acc_per_fold[0],acc_per_fold[1]),acc_per_fold[2])/3
                somme_acc_training = np.add(np.add(acc_per_fold_training[0],acc_per_fold_training[1]),acc_per_fold_training[2])/3
                #callback_model = callback()
                #epoch = callback_model.EarlyStopping(somme_acc_training, somme_acc_val)
                vec_temp = somme_val#[0:epoch]
                index = np.argmax(self.moving_average(vec_temp, window))
                smoothed_acc_training = self.moving_average(somme_acc_training, window)
                acc_per_fold_temp = copy.deepcopy(acc_per_fold)
                loss_per_fold_temp = copy.deepcopy(loss_per_fold)
                for i in range (0,3):
                    acc_per_fold[i] = acc_per_fold[i][index]
                    loss_per_fold[i] = loss_per_fold[i][index]
            if verbose is True:
                print(acc_per_fold)
                print('------------------------------------------------------------------------')
                print('Average scores for all folds:')
                print(f'> Accuracy: {np.mean(acc_per_fold*100)} (+- {np.std(acc_per_fold*100)})')
                print('------------------------------------------------------------------------')
            if neural is True:
                if np.mean(loss_per_fold) < best_loss:# and smoothed_acc_training[index]>0.55:
                    smoothed_acc_training_bool = True
                    best_loss = np.mean(loss_per_fold)
                    best_accuracy = np.mean(acc_per_fold)
                    best_params = params_dic
                    if neural is True:
                        best_params["epochs"] = index + 1
                #else:
                #    if np.mean(loss_per_fold) < best_loss and smoothed_acc_training_bool is False:
                #        best_loss = np.mean(loss_per_fold)
                #        best_accuracy = np.mean(acc_per_fold)
                #        best_params = params_dic
                #        if neural is True:
                #            best_params["epochs"] = index + 1
            else:
                if np.mean(loss_per_fold) < best_loss:
                    best_loss = np.mean(loss_per_fold)
                    best_accuracy = np.mean(acc_per_fold)
                    best_params = params_dic
                    if neural is True:
                        best_params["epochs"] = index + 1
            if neural is True:
                for i in range (0,3):
                    acc_per_fold_total[param_iteration,i,:] = acc_per_fold_temp[i]
                    acc_per_fold_training_total[param_iteration,i,:] = acc_per_fold_training[i]
                    loss_per_fold_total[param_iteration,i,:] = loss_per_fold_temp[i]
                    loss_per_fold_training_total[param_iteration,i,:] = loss_per_fold_training[i]
            else:
                acc_per_fold_total[param_iteration,:] = acc_per_fold
                loss_per_fold_total[param_iteration,:] = loss_per_fold
            param_iteration = param_iteration + 1
        if self.refit is True:
            if Exog_train is not None:
                lags = best_params["lags"]
                self.model.fit(X_train, exog=Exog_train.iloc[:,0:lags], X_test=X_test, Y_test=Y_test, **best_params)
            else:
                reset_keras()
                if history_index is not None:
                    self.model.set_indices(X_train.index, history_index)
                if neural is True:
                    self.model.fit(X_train, Y_train, X_test=X_test, Y_test=Y_test, **best_params)
                else:
                    self.model.fit(X_train, Y_train, X_test=X_test, Y_test=Y_test, **best_params)
            if neural is True:
                return self.model, best_loss, best_accuracy, best_params, loss_per_fold_training_total, loss_per_fold_total, acc_per_fold_training_total, acc_per_fold_total
            else:
                return self.model, best_loss, best_accuracy, best_params, loss_per_fold_total, acc_per_fold_total
        else:
            return best_loss, best_accuracy, best_params, loss_per_fold_total, acc_per_fold_total

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