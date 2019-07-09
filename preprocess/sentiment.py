# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 14:42:32 2019

@author: alexa
"""
import pandas as pd
import numpy as np
import preprocess as p
from collections import defaultdict
from collections import Counter
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import pdb
import time
import ast


class Sentiment(object):
    
    """
    
    Sentiment object. Perform sentiment analysis of the tweets passed as 
    arguments. Sentiment results are based on two metrics, the pointwise
    mutual information association or the word proportions. The two metrics
    can be calculated based on two dictionaries of sentiment words, the Bing 
    or the financial Loughran-McDonald dictionaries. Two sentiments are a-
    -vailable in Bing, i.e. positive or negative. In Loughran-McDonald, 
    words are classified based on for eight different sentiments. The 
    sentiment categories are: negative, positive, uncertainty, litigious,
    modal, constraining. The results are available in the form of a time
    series of sentiment scores than can be plotted together with the evolution
    of the EUR-USD exchange rate.
    
    Attributes:
        fname_curr: The preprocessed currency data
        fname_tweets: The tweets data
        freq: The frequency of the model (minute, hour or day)
        threshold: Threshold for the estimation of the long, short or neutral
                   positions
        dictionary: Dictionary to use for the sentiment scores
        
    """
    sentiments = ['positive', 'negative', 'litigious', 'constraining','uncertainty',
                  'strong_modal', 'moderate_modal', 'weak_modal']
    
    def __init__(self, df):
        
        self.df = df
        self.df.set_index('Date')
        self.eqCurves = []
        self.dictionary = None
        self.association = dict()
        self.index = dict()
        self.correlation = dict()
        self.orientation = pd.DataFrame(index = self.df.Date, columns=['value'])
        for sentiment in self.sentiments:
            self.association[sentiment] = pd.DataFrame(index = self.df.Date, columns=['value'])
            self.index[sentiment] = pd.DataFrame(index = self.df.Date, columns=['value'])
            self.correlation[sentiment] = pd.DataFrame(index = self.df.Date, columns=['value'])
    

    def calculate_sentiments(self, dictionary):
        
        """
        
        Calculate sentiments scores based on the chosen dictionary
        
        Args:
                            
        Returns:
            
        """
        
        master_dictionary = MasterDictionary(dictionary)
        self.dictionary = dictionary
        length = len(self.df)
        for i in tqdm(range(0, length), desc = "progress"):
            terms = ast.literal_eval(self.df.text.iloc[i])
            n_tweets = self.df['count'].iloc[i]
            pmi, p_t, semantic_index = self.__semantic(terms, n_tweets, master_dictionary.dictionary) 
            term_assoc = defaultdict(lambda : defaultdict(int))
            for sentiment, n in semantic_index.items():
                for term, n in p_t.items():
                    term_assoc[sentiment][term] = sum(pmi[term][key] for key, value in master_dictionary.dictionary[sentiment].items())
                self.association[sentiment].iloc[i] = sum(term_assoc[sentiment].values())/len(term_assoc[sentiment]) 
                self.index[sentiment].iloc[i] = semantic_index[sentiment] 
                self.association[sentiment] = self.association[sentiment].fillna(0)
                self.index[sentiment] = self.index[sentiment].fillna(0)


    def __semantic(self, terms, n_tweets, dictionary):
        
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
        com = defaultdict(lambda : defaultdict(int))   
        semantic_index = defaultdict(lambda: 0)
        length = 0
        # Build co-occurrence matrix
        for k in range(len(terms)):
            count_terms.update(terms[k])
            for i in range(len(terms[k])-1):       
                length += 1
                for sentiment, terms_ in dictionary.items():
                    if terms[k][i] in dictionary[sentiment]:
                        semantic_index[sentiment] += 1
                for j in range(i+1, len(terms[k])):
                    w1, w2 = sorted([terms[k][i], terms[k][j]])                
                    if w1 != w2:
                        com[w1][w2] += 1
            length += 1
            for sentiment, terms_ in dictionary.items():
                if terms[k][len(terms[k]) - 1] in dictionary[sentiment]:
                    semantic_index[sentiment] += 1

        semantic_index = {k: v / length for k, v in semantic_index.items()} 
        # n_docs is the total n. of tweets
        p_t = {}
        p_t_com = defaultdict(lambda : defaultdict(int))
     
        for term, n in count_terms.items():
            p_t[term] = n / n_tweets
            for t2 in com[term]:
                p_t_com[term][t2] = com[term][t2] / n_tweets
      
        pmi = defaultdict(lambda : defaultdict(int))
        for t1 in p_t:
            for t2 in com[t1]:
                denom = p_t[t1] * p_t[t2]
                pmi[t1][t2] = math.log2(p_t_com[t1][t2] / denom)
        
        return pmi, p_t, semantic_index
    

    def calculate_orientation(self):

        """
        
        Calculates semantic orientation
        
        Args:
            
        Returns:
            
        """
        for i in range(len(self.orientation)):
            self.orientation.iloc[i] = self.association['positive'].iloc[i] - self.association['negative'].iloc[i]


    def correl(self):

        """
        
        Calculates correlation of the sentiment scores with the return
        
        Args:
            
        Returns:
            
        """

        if self.dictionary == 'Bing':
            self.correlation['positive'] = np.corrcoef(self.df.Return, self.association['positive'].value)
            self.correlation['negative'] = np.corrcoef(self.df.Return, self.association['negative'].value)
        else:  
            for sentiment in self.sentiments:
               self.correlation[sentiment] = np.corrcoef(self.df.Return, self.association[sentiment].value)           


    def plot_(self, type_):
        
        """
        
        Plot the cumulative return of a buy and hold and the scores of the
        different sentiments
        
        Args:
            type_: The sentiment scores to plot, i.e. index or association
            
        Returns:
            
        """
        fig, ax1 = plt.subplots()
        ax1.plot(self.orientation.index, self.orientation.value, 'b-', label = 'orientation')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Semantic ortientation', color='b')
        ax1.tick_params('y', colors='b')
        ax2 = ax1.twinx()
        if type_ == 'association':
            result = self.association
        elif type_ == 'index':
            result = self.index
        else:
            print ('wrong type')
            return
        if self.dictionary == 'Bing':
            ax2.plot(result['positive'].index, result['positive'].value, label = 'positive')
            ax2.plot(result['negative'].index, result['negative'].value, label = 'negative')
        else:
            for sentiment in self.sentiments:
                ax2.plot(result[sentiment].index, result[sentiment].value, label = sentiment)
        ax2.set_ylabel('Sentiment score', color='r')
        ax2.tick_params('y', colors='r')

        fig.tight_layout()
        plt.legend()
        plt.show()
        
        
class MasterDictionary(object):
    
    """
    
    MasterDictionary object. Contains the words dictionaries relevant for
    each sentiments. For Bing, two sentiment dictionaries are available, i.e.
    positive and negative. For Loughran-McDonald, eight sentiment dictionaries
    are available
    
    Attributes:
        dictionary: Dictionary to use for the sentiment scores
        
    """

    sentiments = ['positive', 'negative', 'litigious', 'constraining','uncertainty',
                           'strong_modal', 'moderate_modal', 'weak_modal']
    
    def __init__(self, dictionary):
        
        self.dictionary = defaultdict(lambda : defaultdict(int))
        if dictionary == 'Bing':
            self.__Bing()
        elif dictionary == 'Loughran-McDonald':
            self.__Loughran_McDonald()
        else:
            print ('invalid dictionary')
            return
            
        
    def __Bing(self):
        
        """
        
        Load sentiment dictionaries for Bing
        
        Args:
                            
        Returns:
            
        """
        
        file = open('C:/Users/alexa/OneDrive/Desktop/Thesis/Lexicon/Bing/negative-words.txt', 'r')
        negativeList = file.readlines()
        negativeList = [x.replace('\n', '') for x in negativeList]
        self.dictionary['negative'] = dict.fromkeys(negativeList, 1)
        
        file = open('C:/Users/alexa/OneDrive/Desktop/Thesis/Lexicon/Bing/positive-words.txt', 'r')
        positiveList = file.readlines()
        positiveList = [x.replace('\n', '') for x in positiveList]
        self.dictionary['positive'] = dict.fromkeys(positiveList, 1) 
        
                
    def __Loughran_McDonald(self): 
        
        """
        
        Load sentiment dictionaries for Loughran-McDonald
        
        Args:
                            
        Returns:
            
        """
        
        path = 'C:/Users/alexa/OneDrive/Desktop/Thesis/Lexicon/Loughran-McDonald/'     
        for sentiment in self.sentiments:
            file = open(str(path + sentiment + '.txt'), 'r')
            List = file.readlines()
            List = [x.replace('\n', '').lower() for x in List]
            self.dictionary[sentiment] = dict.fromkeys(List, 1)


     
