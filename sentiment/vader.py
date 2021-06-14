from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from sentiment.sentiment import sentiment 
from utils.parallel import parallel
import warnings
from tqdm import tqdm
import ast
import sys
import pandas as pd
import numpy as np
import pdb


class vader(sentiment, parallel):

    #Defining module name as global variable for access in parent class (parallel)
    mod = sys.modules[__name__]
    sentiments = ['positive', 'negative', 'objective', 'compound']
    location = 'C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/src/figures/'

    def __init__(self):
        super().__init__()
        self.sid_obj = SentimentIntensityAnalyzer() 

    def calculate_sentiments(self, data, parallel=False):
        """
        Calculate sentiments scores based on Vader
        """
        self.data = data
        for sentiment in self.sentiments:
            self.index[sentiment] = pd.DataFrame(index=self.data.get_date(), columns=['value'])
        length = data.get_length()
        for i in tqdm(range(0, length), desc="progress"):
            text = data.get_text().iloc[i]
            try:
                tokenized = sent_tokenize(text)
                n = 0
                positive = 0
                negative = 0
                objective = 0
                compound = 0
                for sentence in tokenized:
                    sentiment_dict = self.sid_obj.polarity_scores(sentence)
                    pos = sentiment_dict['pos']
                    neg = sentiment_dict['neg']
                    obj = sentiment_dict['neu']
                    comp = sentiment_dict['compound']
                    if np.isnan(pos) or np.isnan(neg) or np.isnan(obj) or np.isnan(comp):
                        pass 
                    else:
                        positive += pos
                        negative += neg
                        objective += obj
                        compound += comp
                    n += 1
                self.index['positive'].iloc[i] = positive / n
                self.index['negative'].iloc[i] = negative / n
                self.index['objective'].iloc[i] = objective / n
                self.index['compound'].iloc[i] = compound / n
            except:
                self.index['positive'].iloc[i] = 0
                self.index['negative'].iloc[i] = 0
                self.index['objective'].iloc[i] = 0
                self.index['compound'].iloc[i] = 0
        if parallel:
            return self.index