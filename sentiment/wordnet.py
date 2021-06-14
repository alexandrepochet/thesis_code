from sentiment.sentiment import sentiment 
from sentiment.Word import Word
from utils.parallel import parallel
from nltk.corpus import wordnet as wn
from tqdm import tqdm
import ast
import sys
import pandas as pd
import numpy as np
#import networkx as nx
import pdb


class wordnet(sentiment, parallel):

    #Defining module name as global variable for access in parent class (parallel)
    mod = sys.modules[__name__]
    sentiments = ['positive', 'negative', 'objective']
    location = 'C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/src/figures/'

    def __init__(self):
        super().__init__()

    def calculate_sentiments(self, data, parallel=False):
        """
        Calculate sentiments scores based on word sense disambiguation
        """
        self.data = data
        for sentiment in self.sentiments:
            self.index[sentiment] = pd.DataFrame(index=self.data.get_date(), columns=['value'])
        length = self.data.get_length()

        for i in tqdm(range(0, length), desc="progress"):
            text = self.data.get_text().iloc[i]
            pos, neg, obj = self.__WSD_classify(text)
            self.index['positive'].iloc[i] = pos
            self.index['negative'].iloc[i] = neg
            self.index['objective'].iloc[i] = obj

        if parallel:
            return self.index
                     
    def __WSD_classify(self, text):
        """
        Calculate sentiment based on word sense disambiguation
        
        Args:
            text: The tweet sample                   
        Returns:
            The positive, negative and objective scores  
        """
        words = self.__WSD_process(text)
        pos = neg = obj = 0
        for word in words:
            pos += word.get_pos_score()
            neg += word.get_neg_score()
            obj += word.get_obj_score()
        if len(words) != 0:
            pos = pos/len(words)
            neg = neg/len(words)
            obj = obj/len(words)
        
        return pos, neg, obj
    
    def __WSD_process(self, wordsList):
        """
        Perform word sense disambiguation
        
        Args:
            wordsList: The tweet sample                  
        Returns:
            The disambiguated sense of the words of the tweets   
        """
        text = []
        wordsList = ast.literal_eval(wordsList)
        i = 0
        for word in wordsList:
            synset = wn.synset(word)
            myword = Word(synset)
            text.append(myword)
            i = i + 1
        return text

    @staticmethod
    def closure_graph(synset, fn):
        seen = set()
        graph = nx.DiGraph()
        def recurse(s):
            if not s in seen:
                seen.add(s)
                graph.add_node(s.name())
                for s1 in fn(s):
                    graph.add_node(s1.name())
                    graph.add_edge(s.name(), s1.name())
                    recurse(s1)

        recurse(synset)
        return graph
    