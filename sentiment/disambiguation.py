# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 17:55:23 2019

@author: alexa
"""
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
import preprocess as pr
import pandas as pd
import matplotlib.pyplot as plt


class Disambiguation(object):
    
    """
    
    Word disambiguation sentiment object. Perform sentiment analysis based 
    on word disambiguation for the tweets passed as arguments. The algorithm
    perform word sense disambiguation, i.e. derived the most probable sense
    of each word based on the presence of the other words in the sentence, and
    then derive a sentiment score for each tweet. Word sense disambiguation is 
    performed with Wordnet and sentiment analysis with SentiwordNet. Results
    are available in the form of a time series of sentiment scores than can
    be plotted together with the evolution of the EUR-USD exchange rate.
    
    Attributes:
        fname_curr: The preprocessed currency data
        fname_tweets: The tweets data
        freq: The frequency of the model (minute, hour or day)
        threshold: Threshold for the estimation of the long, short or neutral
                   positions
        
    """
    
    def __init__(self, fname_curr, fname_tweets, freq, threshold):
        
        self.freq = freq
        self.threshold = threshold
        self.eqCurves = []
        self.positive = pd.DataFrame(index = self.df.index, columns=['value'])
        self.negative = pd.DataFrame(index = self.df.index, columns=['value'])
        self.objective = pd.DataFrame(index = self.df.index, columns=['value'])
        self.df = pr.preprocess__(fname_curr, fname_tweets, freq, threshold, True) 
    
    
    def calculate_sentiments(self):
        
        """
        
        Calculate sentiments scores based on word sense disambiguation
        
        Args:
                            
        Returns:
            
        """
        
        length = len(self.df)
        for i in range(0, length):
            text = self.df.text.iloc[i]
            pos, neg, obj = self.__WSD_classify__(text)
            self.positive.iloc[i] = pos
            self.negative.iloc[i] = neg
            self.objective.iloc[i] = obj
            
        returns = pd.DataFrame(index = self.df.index, 
                               columns=['Buy and Hold'])
        returns['Buy and Hold'] = self.df.Return
        returns['Buy and Hold'].iloc[0] = 0
        returns['Strategy'].iloc[0] = 0
        
        self.eqCurves = pd.DataFrame(index = self.df.index, 
                           columns=['Buy and Hold'])
        self.eqCurves['Buy and Hold']=returns['Buy and Hold'].cumsum()+1
            
            
    def __WSD_classify__(self, text):
        
        """
        
        Calculate sentiment based on word sense disambiguation
        
        Args:
            text: The tweet sample
                            
        Returns:
            The positive, negative and objective scores
            
        """
    
        words = self.__WSD_process__(text)
        pos = neg = obj = 0
        for word in words:
            pos += word.pos_score
            neg += word.neg_score
            obj += word.obj_score
        
        pos = pos/len(words)
        neg = neg/len(words)
        obj = obj/len(words)
        
        return pos, neg, obj
    
    
    def __WSD_process__(self, wordsList):
        
        """
        
        Perform word sense disambiguation
        
        Args:
            wordsList: The tweet sample
                            
        Returns:
            The disambiguated sense of the words of the tweets
            
        """
        
        text = []
        for sentence in wordsList:
            for word, pos in sentence:
                synset = self.__disambiguateWordSenses__(sentence, (word, pos))
                myword = Word(word, pos, synset)
                text.append(myword)
        
        return text
    
    
    def __disambiguateWordSenses__(self, sentence, word):
        
        """
        
        Perform word sense disambiguation
        
        Args:
            wordsList: The tweet sample
                            
        Returns:
            The disambiguated sense of the words of the given sentence
            
        """
    
        token = word[0]
        POS = self.__wordnet_pos_code__(word[1])
        wordsynsets = wn.synsets(token, POS)
        bestScore = 0.0
        result = None
        for synset in wordsynsets:
            for w, p in sentence:
                score = 0.0
                for wsynset in wn.synsets(w, self.__wordnet_pos_code__(p)):
                    sim = wn.path_similarity(wsynset, synset)
                    if sim is None:
                        continue
                    else:
                        score += sim
                if score > bestScore:
                    bestScore = score
                    result = synset
                    
        return result
    
    
    def __wordnet_pos_code__(self, tag):
        
        """
        
        Perform mapping of POS tagging between nltk and Wordnet
        
        Args:
            tag: The POS tag to map
                            
        Returns:
            The mapped POS tag
            
        """

        if tag.startswith('NN'):
            return wn.NOUN
        elif tag.startswith('VB'):
            return wn.VERB
        elif tag.startswith('JJ'):
            return wn.ADJ
        elif tag.startswith('RB'):
            return wn.ADV
        else:
            return '' 
        
        
    def plot_(self):
        
        """
        
        Plot the cumulative return of a buy and hold and the scores of the
        different sentiments
        
        Args:
            
        Returns:
            
        """
        
        self.eqCurves['Buy and Hold'].plot(figsize=(10,8))
        self.positive.plot()
        self.negative.plot()
        self.objective.plot()
        plt.legend()
        plt.show()


class Word(object):
    
    """
    
    Word object. Stores information about the sentiment and the synset of
    a word. Convenience class for performing word sense disambiguation
    
    Attributes:
        word: The word
        POS: The POS tag of the word
        synset: The synset of the word
        
    """
    
    def __init__(self, word, POS, synset):
        try:
            self.word = word
            self.pos_score = swn.senti_synset(synset.name()).pos_score()
            self.neg_score = swn.senti_synset(synset.name()).neg_score()
            self.obj_score = swn.senti_synset(synset.name()).obj_score()
            self.synset = synset
            self.POS = POS
        except:
            self.word = word
            self.pos_score = 0
            self.neg_score = 0
            self.obj_score = 1
            self.synset = synset
            self.POS = POS

    def __str__(self):
        """Prints just the Pos/Neg scores for now."""
        s = ""
        s += self.synset.name + "\t"
        s += "PosScore: %s\t" % self.pos_score
        s += "NegScore: %s" % self.neg_score
        return s
    
    
def main():
    
     """
     
     Execute matching action for testing
     
     """
 
     
if __name__ == '__main__':
    main()   
    