import warnings
from collections import defaultdict
from collections import Counter
from sentiment.sentiment import sentiment
from copy import deepcopy
from utils.parallel import parallel
from sklearn import preprocessing
from statsmodels.tsa.stattools import grangercausalitytests
from utils.utils import define_date_frequency
import math
import ast
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import pdb
import sys


class association(sentiment, parallel):
    """
    association object. Perform sentiment analysis of the tweets passed as
    arguments. Sentiment results are based on two metrics, the pointwise
    mutual information association or the word proportions. The two metrics
    can be calculated based on dictionaries of sentiment words, like the Bing, 
    OpinionFinder or the financial Loughran-McDonald dictionaries. Two sentiments are
    available in Bing, i.e. positive or negative. In Loughran-McDonald,
    words are classified based on for eight different sentiments. The
    sentiment categories are: negative, positive, uncertainty, litigious,
    modal, constraining. The results are available in the form of a time
    series of sentiment scores than can be plotted together with the evolution
    of the EUR-USD exchange rate.
    """
    #Defining module name as global variable for access in parent class (parallel)
    mod = sys.modules[__name__]
    location = 'C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/src/figures/'

    def __init__(self, dictionary, association=True):
        super().__init__()
        self.dictionary = dictionary
        self.association = dict()
        self.orientation = None
        self.sentiments = []
        self.boolean_association = association
        for key in self.dictionary:
            self.sentiments += [key]
    
    def set_association(self, bool):
        self.boolean_association = bool 

    def get_boolean_association(self, bool):
        return self.boolean_association
        
    def get_dictionary(self):
        return self.dictionary

    def get_association(self, sentiment):
        return self.association[sentiment]

    def get_index(self, sentiment):
        return self.index[sentiment]

    def get_orientation(self):
        return self.orientation

    def get_correlation(self, type_, sentiment):
        return self._correl(type_)[sentiment]

    def update_dictionary(self, dictionary):
        self.dictionary = dictionary
        self.sentiments = []
        for key in self.dictionary:
            self.sentiments += list(key)

    def calculate_sentiments(self, data, parallel=False):
        """
        Calculate sentiments scores based on the chosen dictionary
        """
        self.data = data
        for sentiment in self.sentiments:
            if self.boolean_association is True:
                self.association[sentiment] = pd.DataFrame(index=self.data.get_date(), columns=['value'])
            self.index[sentiment] = pd.DataFrame(index=self.data.get_date(), columns=['value'])
        length = self.data.get_length()

        for i in tqdm(range(0, length), desc="progress"):
            terms = ast.literal_eval(self.data.get_text().iloc[i])
            n_tweets = self.data.get_count().iloc[i]
            if self.boolean_association is True:
                pmi, p_t, semantic_index = self._semantic(terms, n_tweets,
                                                            self.dictionary)
                term_assoc = defaultdict(lambda: defaultdict(int))
            else:
                semantic_index = self._semantic_index(terms, self.dictionary)
            for sentiment, n in semantic_index.items():
                if self.boolean_association is True:
                    for term, n in p_t.items():
                        term_assoc[sentiment][term] = sum(pmi[term][key] for key, value \
                                                        in self.dictionary \
                                                        [sentiment].items())
                    self.association[sentiment].iloc[i] = sum(term_assoc[sentiment].values())/ \
                                                            len(term_assoc[sentiment])
                self.index[sentiment].iloc[i] = semantic_index[sentiment]

        for sentiment in self.sentiments:
            if self.boolean_association is True:
                self.association[sentiment] = self.association[sentiment].fillna(0)
            self.index[sentiment] = self.index[sentiment].fillna(0)

        if parallel:
            if self.boolean_association is True:
                return (self.association, self.index)
            else:
                return self.index

    def _semantic(self, terms, n_tweets, dictionary):
        """
        Calculate the PMI sentiment score based on the words included in the
        given dictionary

        Args:
            terms: The tweets sample
            n_tweets: The number of tweets contained in the sample
            dictionary: The sentiment dictionary
        Returns:
            The score
        """
        count_terms = Counter()
        com = defaultdict(lambda: defaultdict(int))
        semantic_index = defaultdict(lambda: 0)
        length = 0
        if len(terms) > 0:
            obj = terms[0]
            if isinstance(obj, list):
            # Build co-occurrence matrix
                for count_elem, elem in enumerate(terms):
                    count_terms.update(elem)
                    elem_length = len(elem)
                    for i, elem2 in enumerate(elem):
                        length += 1
                        for sentiment, terms_ in dictionary.items():
                            if elem2 in dictionary[sentiment]:
                                semantic_index[sentiment] += 1
                        if self.boolean_association is True:     
                            for j in range(i+1, elem_length):
                                w_1, w_2 = sorted([elem[i], elem[j]])
                                if w_1 != w_2:
                                    com[w_1][w_2] += 1
            else:
                count_terms.update(terms)
                elem_length = len(terms)
                for i, elem in enumerate(terms):
                    length += 1
                    for sentiment, terms_ in dictionary.items():
                        if elem in dictionary[sentiment]:
                            semantic_index[sentiment] += 1
                    if self.boolean_association is True:  
                        for j in range(i+1, elem_length):
                            w_1, w_2 = sorted([terms[i], terms[j]])
                            if w_1 != w_2:
                                com[w_1][w_2] += 1

        if length > 0:
            semantic_index = {k: v / length for k, v in semantic_index.items()}
        # n_docs is the total n. of tweets
        if self.boolean_association is True:
            p_t = {}
            p_t_com = defaultdict(lambda: defaultdict(int))

            for term, n in count_terms.items():
                p_t[term] = n / n_tweets
                for t_2 in com[term]:
                    p_t_com[term][t_2] = com[term][t_2] / n_tweets

            pmi = defaultdict(lambda: defaultdict(int))
            for t_1 in p_t:
                for t_2 in com[t_1]:
                    denom = p_t[t_1] * p_t[t_2]
                    pmi[t_1][t_2] = math.log2(p_t_com[t_1][t_2] / denom)
            return pmi, p_t, semantic_index
        else:
            return semantic_index

    def _semantic_index(self, terms, dictionary):
        semantic_index = defaultdict(lambda: 0)
        length = 0
        if len(terms) > 0:
            obj = terms[0]
            if isinstance(obj, list):
                for elem in terms:
                    for term in elem:
                        length += 1
                        for sentiment, terms_ in dictionary.items():
                            if term in dictionary[sentiment]:
                                semantic_index[sentiment] += 1
            else:
                for term in terms:
                    length += 1
                    for sentiment, terms_ in dictionary.items():
                        if term in dictionary[sentiment]:
                            semantic_index[sentiment] += 1

        if length > 0:
            semantic_index = {k: v / length for k, v in semantic_index.items()}
        
        return semantic_index

    def calculate_orientation(self):
        """
        Calculates semantic orientation
        """
        if self.boolean_association is True:
            self.orientation = pd.DataFrame(index=self.data.get_date(), columns=['value'])
            self.orientation = self.association['positive'] - self.association['negative']
        else:
            print ('association was not calculated \n')

    def _correl(self, type_):
        """
        Calculates correlation of the sentiment scores with the return
        """
        if type_ == 'index':
            return super()._correl()
        else:
            if self.boolean_association is True:
                correlation = dict()
                Return = self.data.get_return()
                for sentiment in self.sentiments:
                    association = self.association[sentiment]
                    correlation[sentiment] = np.corrcoef(Return.Return.astype(float), \
                                                         association.value.astype(float))[0, 1]
            else:
                print ('association was not calculated \n')
                correlation = 0
            return correlation

    def trailing_correl(self, window, type_):
        """
        Calculates trailing correlation of the sentiment scores with the return

        Args:
            window: The window for which the correlation will be calculated
        Returns:
        """
        if type_ == 'index':
            super().trailing_correl(window)
        else:
            if self.boolean_association is True:
                length = self.data.get_length()
                nb_elem = (length - window)
                for key in self.association:
                    self.trailing_correlation[key] = pd.DataFrame(index=self.data.get_date()[(window):],\
                                                                         columns=['value'])
                Return = self.data.get_return()
                for i in range(nb_elem):
                    Return_ = Return.iloc[i:min(length, window + i)]
                    for key in self.association:
                        sentiment_score = self.association[key]
                        sentiment_score = sentiment_score.shift(1)
                        sentiment_score = sentiment_score[1:]
                        sentiment_score = sentiment_score.value[(i ): min(length, window + i)]
                        self.trailing_correlation[key].iloc[i] = np.corrcoef(Return_.Return, sentiment_score.astype(float))[0, 1]
            else:
                print ('association was not calculated \n')

    def standardize(self, type_):
        if type_ == 'index':
            super().standardize()
        else:
            if self.boolean_association is True:
                for key in self.association:
                    self.association[key].value = preprocessing.scale(self.association[key].value)
            else:
                print ('association was not calculated \n')

    def granger_causality(self, type_, Return=None):
        if type_ == 'index':
            super().granger_causality(Return=Return)
        else:
            if self.boolean_association is True:
                print('Number of lags: ' + str(self.LAGS) + '\n')
                for key in self.association:
                    if Return is None:
                        Return_intern = self.data.get_return()
                        Return_intern = Return_intern.shift(1)
                        Return_intern = Return_intern.iloc[1:]
                        data_return = pd.concat([Return_intern, self.association[key][1:]], axis=1)
                    else:
                        data_return  = self.association[key]
                    print('-------------------------------------------------------------\n')
                    print('sentiment: ' + str(key) + '\n')
                    grangercausalitytests(data_return[['Return', 'value']], maxlag=self.LAGS, verbose=True, addconst=True)
                    print('\n')
            else:
                print ('association was not calculated \n')

    def aggregate(self, freq, type_):
        if type_ == 'index':
            super().aggregate(freq=freq)
        else:
            for key in self.association:
                self.association[key] = define_date_frequency(self.association[key], freq, False)
                self.association[key].index.name="index"
                self.association[key] = self.association[key].groupby('index')['value'].agg(['sum','count'])
                self.association[key]['value'] = self.association[key]['sum']/self.association[key]['count'] 

 
    def plot_sentiment(self, type_, title1, title2):
        if self.boolean_association is True:
            fig, ax1 = plt.subplots()
            ax1.plot(self.orientation.index, self.orientation.value, 'g-', label='orientation')
            plt.tight_layout()
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Semantic orientation')
            ax1.tick_params('y')
            plt.xticks(rotation=45)
            ax1.grid(which='both', axis='x')
            plt.savefig(self.location + str(title1) + '.jpg', bbox_inches='tight')
            plt.close()
        else:
            print ('association was not calculated \n')

        if type_ == 'association':
            if self.boolean_association is True:
                result = self.association
            else:
                return
        elif type_ == 'index':
            result = self.index
        else:
            print('wrong type')
            return

        for sentiment in self.sentiments:
            fig, ax = plt.subplots()
            ax.plot(result[sentiment].index, result[sentiment].value, label=sentiment)
            plt.tight_layout()
            ax.set_ylabel('Sentiment score')
            ax.tick_params('y')
            plt.legend()
            plt.xticks(rotation=45)
            ax.grid(which='both', axis='x')
            plt.savefig(self.location + str(title2) + str('_') + str(sentiment) + '.jpg', bbox_inches='tight')
            plt.close()