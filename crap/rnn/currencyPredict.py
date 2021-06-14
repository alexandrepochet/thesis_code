from sklearn.metrics import accuracy_score
from utils.performance import max_drawdown, sharpe_ratio, std, mean
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import pdb

class currencyPredict():

    up = 1
    down = 0
    stable = 2
    location = 'C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/src/figures/'
    warnings.filterwarnings("ignore")

    def __init__(self, model, y_test, market_returns, bid_ask, bid_ask_previous):
        self.model = model
        self.y_test = y_test
        self.market_returns = market_returns
        self.bid_ask = pd.DataFrame(bid_ask.values, index=market_returns.index, columns = ["bid_ask"])
        self.bid_ask_previous = pd.DataFrame(bid_ask_previous.values, index=market_returns.index, columns = ["bid_ask_previous"])
        self.eqCurves = []
        self.strategy = pd.DataFrame(index=market_returns.index, columns = ["signal"])
        self.cost = pd.DataFrame(0, index=market_returns.index, columns="cost")

    def predict(self, x_test, t_cost=True):
        y_pred = self.model.model.predict_classes(x_test)
        null_strategy = self.get_null_strategy()
        self._calculate_returns(y_pred, null_strategy, t_cost)
        return y_pred

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model

    def get_y_test(self):
        return self.y_test

    def get_market_returns(self):
        return self.market_returns

    def get_bid_ask(self):
        return self.bid_ask

    def get_cost(self):
        return self.cost

    def get_accuracy(self, y_pred):
        return accuracy_score(self.y_test, y_pred)

    def get_eqCurves(self):
        return self.eqCurves

    def get_strategy(self):
        return self.strategy

    def set_y_test(self, y_test):
        self.y_test = y_test

    def set_market_returns(self, market_returns):
        self.market_returns = market_returns

    def set_bid_ask(self, bid_ask):
        self.bid_ask = bid_ask

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


    def get_null_accuracy(self):
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
        length = len(self.y_test)
        prop_up = len(self.y_test[self.y_test == self.up]) / (length*1.)
        prop_down = len(self.y_test[y_test == self.down]) / (length*1.)
        prop_stable = len(y_test[y_test == self.stable]) / (length*1.)
        print('up ' + str(prop_up))
        print('down ' + str(prop_down))
        print('stable ' + str(prop_stable))
        null_accuracy = 0
        if prop_up >= prop_down and prop_up >= prop_stable:
            null_accuracy += prop_up / len(self.strategy)*(length*1.)
        elif prop_down >= prop_up and prop_down >= prop_stable:
            null_accuracy += prop_down / len(self.strategy)*(length*1.)
        else:
            null_accuracy += prop_stable / len(self.strategy)*(length*1.)

        return null_accuracy

    def get_null_strategy(self):
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
        length = len(self.y_test)
        prop_up = len(self.y_test[self.y_test == self.up]) / (length*1.)
        prop_down = len(self.y_test[self.y_test == self.down]) / (length*1.)
        prop_stable = len(self.y_test[self.y_test == self.stable]) / (length*1.)
        print('up ' + str(prop_up))
        print('down ' + str(prop_down))
        print('stable ' + str(prop_stable))
        if prop_up >= prop_down and prop_up >= prop_stable:
            return self.up
        elif prop_down >= prop_up and prop_down >= prop_stable:
            return self.down
        else:
            return self.stable

    def _calculate_signal(self, y_pred, null_strategy):
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
        length = y_pred.size
        for j in range(0, length):
            if y_pred[j][0] == self.up:
                self.strategy.iloc[j] = +1
            elif y_pred[j][0] == self.down:
                self.strategy.iloc[j] = -1
            else:
                self.strategy.iloc[j] = 0
        null_strategy_vec = pd.DataFrame(index=self.market_returns.index, columns = ["signal"])
        if null_strategy == self.up:
            null_strategy_vec.iloc[0:(length)] = +1
        elif null_strategy == self.down:
            null_strategy_vec.iloc[0:(length)] = -1
        else:
            null_strategy_vec.iloc[0:(length)] = +0

        return null_strategy_vec

    def _calculate_returns(self, y_pred, null_strategy, t_cost=True):
        """
        Calculate the return of the different strategies

        Args:
            data: The time series return
            training_window: The size of the training window
            bid_ask: The bid ask spread
        Returns:
        """
        null_strategy_vec = self._calculate_signal(y_pred, null_strategy)
        returns = pd.DataFrame(index=self.strategy.index,
                               columns=['Buy and Hold', 'Strategy', 'Majority'])
        returns['Buy and Hold'] = self.market_returns.values
        if t_cost is True:
            length = y_pred.size
            cost.iloc[0] = [self.bid_ask_previous.iloc[0]]/2
            cost.iloc[length-1] = [self.bid_ask_previous.iloc[length-1]]/2
            for j in range(0, length-1):
                if self.strategy.iloc[j]!= self.strategy.iloc[j+1]:
                    cost.iloc[j] = self.bid_ask.iloc[j]
            returns['Strategy'] = self.strategy.signal*returns['Buy and Hold'] - cost['cost']
        else:
            returns['Strategy'] = self.strategy.signal*returns['Buy and Hold']
        returns['Majority'] = null_strategy_vec.signal*returns['Buy and Hold']
        delta = returns.index[2]-returns.index[1]
        index = returns.index[0] - delta
        index = pd.DatetimeIndex([index])
        init =  pd.DataFrame([[0,0,0]], index=index, columns = ['Buy and Hold', 'Strategy', 'Majority'])
        returns = pd.concat([init, returns], ignore_index=False)
        self.eqCurves = pd.DataFrame(index=self.strategy.index,
                                     columns=['Buy and Hold', 'Strategy', 'Majority', 'Returns'])
        self.eqCurves['Buy and Hold'] = returns['Buy and Hold'].cumsum()+1
        self.eqCurves['Strategy'] = returns['Strategy'].cumsum()+1
        self.eqCurves['Majority'] = returns['Majority'].cumsum()+1
        self.eqCurves['Returns'] = returns['Strategy']


    def output_summary_stats(self):
        """
        Creates a list of summary statistics for the portfolio such
        as Sharpe Ratio and drawdown information.
        """
        total_return = self.eqCurves['Strategy'][-1]
        returns =  self.eqCurves['Returns']
        pnl = self.eqCurves['Strategy']

        sharpe_ratio = sharpe_ratio(returns)
        max_dd, dd_duration = drawdowns(pnl)
        mean_return = mean(returns)
        standard_dev = std(returns)

        stats = [("Average Return", "%0.2f%%" % ((mean_return) * 100.0)), 
                 ("Standard Deviation", "%0.2f%%" % ((standard_dev) * 100.0)), 
                 ("Total Return", "%0.2f%%" % ((total_return - 1.0) * 100.0)),
                 ("Sharpe Ratio", "%0.2f" % sharpe_ratio),
                 ("Max Drawdown", "%0.2f%%" % (max_dd * 100.0))]
        return stats