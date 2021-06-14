from abc import ABCMeta, abstractmethod
from sklearn import preprocessing
from statsmodels.tsa.stattools import grangercausalitytests
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.utils import define_date_frequency
import ast
import pdb


class sentiment(metaclass=ABCMeta):
    """
    Abstract class which supports the sentiment methods implemented 
    for currency prediction based on Twitter data
    """
    LAGS = 5

    def __init__(self):
        self.data = None
        self.index = dict()
        self.trailing_correlation = dict()

    @abstractmethod
    def calculate_sentiments(self):
        pass

    def get_data(self):
        return self.data

    def get_index(self, sentiment):
        return self.index[sentiment]

    def get_correlation(self, sentiment):
        return self._correl()[sentiment]

    def get_trailing_correlation(self, sentiment):
        return self.trailing_correl[sentiment]

    def get_sentiments(self):
        return self.sentiments

    def _calculate_close_diff(self):
        difference  = self.data.get_close() - self.data.get_close()/np.exp(self.data.get_Return().values) 
        return difference

    def _correl(self):
        """
        Calculates correlation of the sentiment scores with the return
        """
        correlation = dict()
        for sentiment in self.sentiments:
            index = self.index[sentiment]
            correlation[sentiment] = np.corrcoef(self.data.get_return().Return, \
                                                 index.value.astype(float))[0, 1]
        return correlation


    def trailing_correl(self, window):
        """
        Calculates trailing correlation of the sentiment scores with the return
        Args:
            window: The window for which the correlation will be calculated
        Returns:
        """
        length = self.data.get_length()
        nb_elem = (length - window)
        for sentiment in self.sentiments:
            self.trailing_correlation[sentiment] = pd.DataFrame(index=self.data.get_date()[(window):],\
                                                                       columns=['value'])
        Return = self.data.get_return()
        for i in range(nb_elem):
            Return_ = Return.iloc[i:min(length, window + i)]
            for sentiment in self.sentiments:
                sentiment_score = self.index[sentiment]
                sentiment_score = sentiment_score.value[(i): min(length, window + i)]
                self.trailing_correlation[sentiment].iloc[i] = np.corrcoef(Return_.Return, sentiment_score.astype(float))[0, 1]

    def standardize(self):
        for key in self.index:
            self.index[key].value = preprocessing.scale(self.index[key].value)

    def aggregate(self, freq):
        for key in self.index:
            self.index[key] = define_date_frequency(self.index[key], freq, False)
            self.index[key].index.name="index"
            self.index[key] = self.index[key].groupby('index')['value'].agg(['sum','count'])
            self.index[key]['value'] = self.index[key]['sum']/self.index[key]['count'] 

    def granger_causality(self, Return=None):
        print('Number of lags: ' + str(self.LAGS) + '\n')
        for key in self.index:
            if Return is None:
                Return_intern = self.data.get_return()
                Return_intern = Return_intern.shift(1)
                Return_intern = Return_intern.iloc[1:]
                data_return = pd.concat([Return_intern, self.index[key][1:]], axis=1)
            else:
                data_return  = self.index[key]
            print('-------------------------------------------------------------\n')
            print('sentiment: ' + str(key) + '\n')
            grangercausalitytests(data_return[['Return', 'value']], maxlag=self.LAGS, verbose=True, addconst=True)
            print('\n')
            
    def plot_sentiment(self, title):
        result = self.index
        for sentiment in self.sentiments:
            fig, ax = plt.subplots()
            ax.plot(result[sentiment].index, result[sentiment].value, label=sentiment)
            plt.tight_layout()
            ax.set_ylabel('Sentiment score')
            ax.tick_params('y')
            plt.legend()
            plt.xticks(rotation=45)
            ax.grid(which='both', axis='x')
            plt.savefig(self.location + str(title) + str('_') + str(sentiment) + '.jpg', bbox_inches='tight')
            plt.close()

    def plot_trailing_corr(self, title):
        for sentiment in self.sentiments:
            fig, ax = plt.subplots()
            ax.plot(self.trailing_correlation[sentiment].index, self.trailing_correlation[sentiment].value, label=sentiment)
            plt.tight_layout()
            ax.set_ylabel('Trailing correlation')
            ax.tick_params('y')
            plt.legend()
            plt.xticks(rotation=45)
            ax.grid(which='both', axis='x')
            plt.savefig(self.location + str(title) + str('_')  + str(sentiment) + '.jpg', bbox_inches='tight')
            plt.close()

