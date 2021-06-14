import warnings
from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb


class algorithm(metaclass=ABCMeta):
    """
    Abstract class which supports the machine learning methods implemented 
    for currency prediction
    """
    up = 'up'
    down = 'down'
    stable = 'stable'
    location = 'C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/src/figures/'
    warnings.filterwarnings("ignore")

    def __init__(self, data):
        self.data = data
        self.accuracy = []
        self.null_accuracy = 0
        self.eqCurves = []
        self.signal = []
        self.null_strategy_signal = []

    def _initialize(self, training_window, shift=False):
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
        self.null_accuracy = 0
        Return = 0
        close = 0
        bid_ask = 0
        if shift is True:
            df = self.data.get_df()
            Return = self.data.get_Return()
            y_var = df['Direction']
            x_var = df.shift(1)
            x_var = x_var[1:]
            y_var = y_var[1:]
            close = pd.DataFrame(x_var.close, columns = ['close'])
            close_bid_ask = pd.DataFrame(x_var.close_bid_ask, columns = ['close_bid_ask'])
            x_var = x_var.drop(['Direction', 'open_bid_ask', 'close_bid_ask'], axis=1)
        else:
            df = self.data.get_df()
            y_var = df.Direction
            close = pd.DataFrame(df.close, columns = ['close'])
            Return = pd.DataFrame(df.Return, columns = ['Return'])
            close_bid_ask = pd.DataFrame(df.close_bid_ask, columns = ['close_bid_ask'])
            x_var = df.drop(['Direction', 'Return', 'count', 'close', 'close_bid_ask'], axis=1)
        size = y_var.size
        self.signal = 0*Return.iloc[-(size - training_window + 1):]
        self.null_strategy_signal = 0*Return.iloc[-(size - training_window + 1):]
        close = close.iloc[-(size - training_window + 1):].values
        bid_ask = close_bid_ask.iloc[-(size - training_window + 1):]
        bid_ask = bid_ask.rolling(min_periods=1, window=2).sum()/(4*close)

        return x_var, y_var, bid_ask, size, Return

    def _calculate_returns(self, data, training_window, bid_ask, Return=None):
        """
        Calculate the return of the different strategies

        Args:
            data: The time series return
            training_window: The size of the training window
            bid_ask: The bid ask spread
        Returns:
        """
        returns = pd.DataFrame(index=self.signal.index,
                               columns=['Buy and Hold', 'Strategy', 'Majority'])
        if Return is not None:
            returns['Buy and Hold'] = Return[-(data.size - training_window + 1):].values
        else:
            returns['Buy and Hold'] = self.data.get_Return().iloc[-(data.size - training_window + 1):].values
        cost = self.signal.diff()
        cost.iloc[0] = 0
        cost = np.abs(cost)* bid_ask.values
        returns['Strategy'] = self.signal.Return*returns['Buy and Hold'] - cost.Return
        returns['Majority'] = self.null_strategy_signal.Return*returns['Buy and Hold']
        returns['Buy and Hold'].iloc[0] = 0
        returns['Strategy'].iloc[0] = 0
        returns['Majority'].iloc[0] = 0
        self.eqCurves = pd.DataFrame(index=self.signal.index,
                                     columns=['Buy and Hold', 'Strategy', 'Majority'])
        self.eqCurves['Buy and Hold'] = returns['Buy and Hold'].cumsum()+1
        self.eqCurves['Strategy'] = returns['Strategy'].cumsum()+1
        self.eqCurves['Majority'] = returns['Majority'].cumsum()+1

    def plot_return(self, title, dash=False):
        """
        Plot the return of the predicted strategy and the buy and hold
        strategy

        Args:
            title: Title of the figure to save
            dash: Boolean indicator for the plot
        Returns:
        """
        self.eqCurves['Strategy'].plot(figsize=(10, 8))
        self.eqCurves['Buy and Hold'].plot()
        if dash is True:
            self.eqCurves['Majority'].plot(dashes=[2, 6])
        else:
            self.eqCurves['Majority'].plot()
        plt.ylabel('Index value')
        plt.xlabel('Date')
        plt.legend()
        plt.savefig(self.location + str(title) + '.jpg', pad_inches=1)
        plt.close()

    def _data_preparation(self, x_var, y_var, training_window, testing_window, size, i):
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
        x_train = x_var[(i*testing_window):min(i*testing_window+ \
                         training_window, size)].values
        x_test = x_var[(i*testing_window+training_window):min((i+1) \
                        *testing_window+training_window, size)].values
        y_train = y_var[(i*testing_window):min(i*testing_window+ \
                         training_window, size)].values
        y_test = y_var[(i*testing_window+training_window):min((i+1)* \
                        testing_window+training_window, size)].values
        # remove class with less than 10 observations
        unique, counts = np.unique(y_train, return_counts=True)
        _class = unique[np.argwhere(counts < 10)]
        index = np.where(y_train == _class)
        y_train = np.delete(y_train, index)
        x_train = np.delete(x_train, index, axis=0)
        index = np.where(y_test == _class)
        y_test = np.delete(y_test, index)
        x_test = np.delete(x_test, index, axis=0)

        return x_train, y_train, x_test, y_test

    def _get_null_strategy(self, x_test, y_test):
        """
        Determines the null strategy and calculate accuracy

        Args:
            x_test: The test data (features)
            y_test: The direction of the test data
        Returns:
            null_strategy: The majority class strategy
            null_accuracy: The majority class accuracy
        """
        #majority class strategy
        prop_up = len(x_test[y_test == self.up]) / (len(x_test)*1.)
        prop_down = len(x_test[y_test == self.down]) / (len(x_test)*1.)
        prop_stable = len(x_test[y_test == self.stable]) / (len(x_test)*1.)
        print('up ' + str(prop_up))
        print('down ' + str(prop_down))
        print('stable ' + str(prop_stable))
        null_strategy = ''
        if prop_up >= prop_down and prop_up >= prop_stable:
            self.null_accuracy += prop_up / len(self.signal)*(len(x_test)*1.)
            null_strategy = self.up
        elif prop_down >= prop_up and prop_down >= prop_stable:
            self.null_accuracy += prop_down / len(self.signal)*(len(x_test)*1.)
            null_strategy = self.down
        else:
            self.null_accuracy += prop_stable / len(self.signal)*(len(x_test)*1.)
            null_strategy = self.stable

        return null_strategy

    def _calculate_signal(self, y_test, y_pred, testing_window, null_strategy, i):
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
            if y_pred[j] == self.up:
                self.signal.iloc[i*testing_window + j + 1] = +1
            elif y_pred[j] == self.down:
                self.signal.iloc[i*testing_window + j + 1] = -1
            else:
                self.signal.iloc[i*testing_window + j + 1] = 0
        if null_strategy == self.up:
            self.null_strategy_signal.iloc[(i*testing_window + 1):(i*testing_window + \
                                            y_test.size + 1)] = +1
        elif null_strategy == self.down:
            self.null_strategy_signal.iloc[(i*testing_window + 1):(i*testing_window + \
                                            y_test.size + 1)] = -1
        else:
            self.null_strategy_signal.iloc[(i*testing_window + 1):(i*testing_window + \
                                            y_test.size + 1)] = +0
