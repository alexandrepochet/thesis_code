from sklearn.metrics import accuracy_score
from utils.performance import max_drawdown, sharpe_ratio, std, mean
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import pdb

class currencyReturn():

    up = 1
    down = 0
    stable = 2
    location = 'C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/src/figures/'
    warnings.filterwarnings("ignore")

    def __init__(self, y_test, market_returns, bid_ask, bid_ask_previous):
        self.y_test = y_test
        self.market_returns = market_returns
        self.bid_ask = pd.DataFrame(bid_ask.values, index=market_returns.index, columns = ["bid_ask"])
        self.bid_ask_previous = pd.DataFrame(bid_ask_previous.values, index=market_returns.index, columns = ["bid_ask_previous"])
        self.eqCurves = []
        self.strategy = pd.DataFrame(index=market_returns.index, columns = ["signal"])
        self.cost = pd.DataFrame(0, index=market_returns.index, columns=["cost"])

    def run(self, y_pred, t_cost=True):
        null_strategy = self.get_null_strategy()
        returns = self._calculate_returns(y_pred, null_strategy, t_cost)
        return returns

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
        self.eqCurves['SVM_TA'].plot(figsize=(10, 8))
        self.eqCurves['Buy and Hold'].plot()
        self.eqCurves['Majority'].plot()
        #if dash is True:
            #self.eqCurves['Majority'].plot(dashes=[2, 6])
        #else:
            #self.eqCurves['Majority'].plot()
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
        #print('up ' + str(prop_up))
        #print('down ' + str(prop_down))
        #print('stable ' + str(prop_stable))
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
        #print('up ' + str(prop_up))
        #print('down ' + str(prop_down))
        #print('stable ' + str(prop_stable))
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
            if y_pred[j] == self.up:
                self.strategy.iloc[j] = +1
            elif y_pred[j] == self.down:
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
                               columns=['Buy and Hold', 'SVM_TA', 'Majority'])
        returns['Buy and Hold'] = self.market_returns.values
        if t_cost is True:
            length = y_pred.size
            self.cost.iloc[0] = self.bid_ask_previous.iloc[0].values/2
            self.cost.iloc[length-1] = self.bid_ask.iloc[length-1].values/2
            for j in range(0, length-1):
                if self.strategy.iloc[j].values != self.strategy.iloc[j+1].values:
                    self.cost.iloc[j] += self.bid_ask.iloc[j].values
            returns['SVM_TA'] = self.strategy.signal*returns['Buy and Hold'] - self.cost['cost']
        else:
            returns['SVM_TA'] = self.strategy.signal*returns['Buy and Hold']
        delta = returns.index[2]-returns.index[1]
        index = returns.index[0] - delta
        index = pd.DatetimeIndex([index])
        init =  pd.DataFrame([[0,0,0]], index=index, columns = ['Buy and Hold', 'SVM_TA', 'Majority'])
        returns = pd.concat([init, returns], ignore_index=False)
        self.eqCurves = pd.DataFrame(index=self.strategy.index,
                                     columns=['Buy and Hold', 'SVM_TA', 'Majority', 'Returns', 'Returns_bm'])
        self.eqCurves['Buy and Hold'] = returns['Buy and Hold'].cumsum()+1
        self.eqCurves['SVM_TA'] = returns['SVM_TA'].cumsum()+1
        self.eqCurves['Majority'] = 1-self.eqCurves['Buy and Hold']+1
        self.eqCurves['Returns'] = returns['SVM_TA']
        self.eqCurves['Returns_bm'] = returns['Buy and Hold']

        return returns


    def output_summary_stats(self, bm=False):
        """
        Creates a list of summary statistics for the portfolio such
        as Sharpe Ratio and drawdown information.
        """
        total_return = self.eqCurves['SVM_TA'][-1]
        returns =  self.eqCurves['Returns']
        pnl = self.eqCurves['SVM_TA']

        sharpe_ratio_val = sharpe_ratio(returns)
        max_dd = max_drawdown(pnl)
        mean_return = mean(returns)
        standard_dev = std(returns)

        stats = [("Average Return", "%0.2f%%" % ((mean_return) * 100.0)), 
                 ("Standard Deviation", "%0.2f%%" % ((standard_dev) * 100.0)), 
                 ("Total Return", "%0.2f%%" % ((total_return - 1.0) * 100.0)),
                 ("Sharpe Ratio", "%0.2f" % sharpe_ratio_val),
                 ("Max Drawdown", "%0.2f%%" % (max_dd * 100.0))]
        if bm is True:
            total_return_bm = self.eqCurves['Buy and Hold'][-1]
            returns_bm =  self.eqCurves['Returns_bm']
            pnl_bm = self.eqCurves['Buy and Hold']

            sharpe_ratio_val_bm = sharpe_ratio(returns_bm)
            max_dd_bm = max_drawdown(pnl_bm)
            mean_return_bm = mean(returns_bm)
            standard_dev_bm = std(returns_bm)

            stats_bm = [("Average Return", "%0.2f%%" % ((mean_return_bm) * 100.0)), 
                        ("Standard Deviation", "%0.2f%%" % ((standard_dev_bm) * 100.0)), 
                        ("Total Return", "%0.2f%%" % ((total_return_bm - 1.0) * 100.0)),
                        ("Sharpe Ratio", "%0.2f" % sharpe_ratio_val_bm),
                        ("Max Drawdown", "%0.2f%%" % (max_dd_bm * 100.0))]
            return stats, stats_bm
        else:
            return stats