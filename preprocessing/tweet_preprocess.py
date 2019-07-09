# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 14:22:19 2019

@author: alexa
Load, preprocess and represents the Tweets data on various charts
"""

import re
from os.path import dirname as up
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import sent_tokenize 
from nltk import pos_tag
from nltk.stem import PorterStemmer
import string
from nltk.stem import WordNetLemmatizer
import json
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.dates as mdates
from datetime import datetime, date, time, timedelta
from pprint import pprint
import numpy as np
import pdb


def load_words():
    
    """
        
    Return file with English vocabulary 
        
    Args:
            
    Returns:
        A list with valid English words
            
    """

    with open('C:/Users/alexa/OneDrive/Desktop/Thesis/Lexicon/English/words_alpha.txt') as word_file:
        valid_words = set(word_file.read().split())

    return valid_words

    
def initialize(): 
    
    """
        
    Initialize various syntactic variables for preprocessing 
        
    Args:
            
    Returns:
        The syntactic variables for further text processing
            
    """
    
    pat1 = r'@[A-Za-z0-9_]+'
    pat2 = r'http[s]?:// [^ ]+'
    pat3 = r'http[s]?://www. [^ ]+'
    pat4 = r'http[s]?://[^ ]+'
    pat4 = r'http[s]?://[^ ]+'
    pat5 = r'http[s]?[^ ]+'
    www_pat = r'www.[^ ]+'
    combined_pat = r'|'.join((pat1, pat2, pat3, pat4, pat5, www_pat))

    dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
           "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
           "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
           "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
           "mustn't":"must not","needn't":"need not", "b uy": "buy", "tech nical":"technical",
           "fu ndamental": "fundamental", "s ell":"sell", "technica l":"technical", "foreca st":"forecast",
           "FXStreetFlashP ublic":"FXStreetFlashPublic", "0 2":"02", "0 4":"04"}
    pattern = re.compile(r'\b(' + '|'.join(dic.keys()) + r')\b')
        
    punctuation = list(string.punctuation)
    stop_words = stopwords.words('english') + punctuation + ['rt', 'via'] 
    r = ["above", "below", "up", "down", "no", "nor", "not"]
    stop_words = [stop_words.remove(x) for x in r]
     
    emoticons_str = r"""
            (?:
                 [:=;] # Eyes
                 [oO\-]? # Nose (optional)
                 [D\)\]\(\]/\\OpP] # Mouth
             )"""
 
    regex_str = [emoticons_str,
                r'<[^>]+>', # HTML tags
                r'(?:@[\w_]+)', # @-mentions
                r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
                r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
                r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
                r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
                r'(?:[\w_]+)', # other words
                r'(?:\S)' # anything else
                ]
            
    tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
    emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
    
    return combined_pat, pattern, dic, stop_words, tokens_re, emoticon_re 


def func(row, frequency, delay = True):  
        
    """
        
    Convenience method for apply function. Format a date given a frequency
    parameter
        
    Args:
        row: The row of the dataframe on which the apply function is executed
        frequency: The formatting frequency. Hourly or daily frequencies are possible
        delay: add 1 period to each date, for the merging with the currency data 

    Returns:
        The date formatted at the given frequency
            
    """

    if frequency == 'H':
        if delay:
            date_ = datetime.combine(date(row.name.year, row.name.month, row.name.day),
                                    time(row.name.hour, 0)) + timedelta(hours = 1) 
        else:
            date_ = datetime.combine(date(row.name.year, row.name.month, row.name.day),
                                    time(row.name.hour, 0))
        return pd.to_datetime(date_, format='%Y-%m-%d %H:%M:%S')
    elif frequency == 'D':
        if delay:  
            date_ = date(row.name.year, row.name.month, row.name.day) + timedelta(days=1)
        else:
            date_ = date(row.name.year, row.name.month, row.name.day)
        return pd.to_datetime(date_,format='%Y-%m-%d')
    elif frequency == 'min':  
        if delay:
            date_ = datetime.combine(date(row.name.year, row.name.month, row.name.day),
                                    time(row.name.hour, row.name.minute)) + timedelta(minutes = 1)
        else:
            date_ = datetime.combine(date(row.name.year, row.name.month, row.name.day),
                                    time(row.name.hour, row.name.minute))
        return pd.to_datetime(date_, format='%Y-%m-%d %H:%M:%S') 
    else:
        print ("invalid frequency")
    

class Tweets(object):
    
    """
    
    Represents the tweets data. Preprocess, plot and resample the tweets data.
    The tweets are stored in formatted files. Alternatively, unformatted 
    file can be passed as arguments.
    
    Attributes:
        fname: Formatted file containing the tweets data 
        raw_file: Unformatted file containing the tweets data
        
    """
    
    vocabulary = dict.fromkeys(load_words(), 1)
    porter = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    combined_pat, pattern, dic, stop_words, tokens_re, emoticon_re = initialize()

    def __init__(self, fname = None, raw_file = None):
        
        if raw_file != None:
            self.df = self.__read_file(raw_file)
        else:
            if fname == None:
                print("file missing")
            else:
                self.df = self.__open(fname)
        self.df.text = [str(t) for t in self.df.text]
        self.df_preprocessed = self.df.copy()
    
    
    @staticmethod
    def tokenize(s, TweetsTokenize = False):
        
        """
        
        Tokenize a string sequence. 
        
        Args:
            s: Lower case indicator 
            TweetsTokenize: Tweet tokenizer indicator
            
        Returns:
            The tokenized sequence
            
        """

        if TweetsTokenize == True:
            tknzr = TweetTokenizer()
            return tknzr.tokenize(s)
        else:
            return Tweets.tokens_re.findall(s)
                
    
    def preprocess(self, lowercase = True, POS = True, Stemming = False, 
                   Lemmatization = True):
        
        """
        
        Preprocess and tokenize the tweets. 
        
        Args:
            lowercase: Lower case indicator 
            POS: Part-of-Speech tagging indicator
            Stemming: Stemming indicator
            Lemmatization: Lemmatization indicator
            
        Returns:
            The preprocessed and tokenized tweets
            
        """
              
        def fun(row, lowercase, POS, Stemming, Lemmatization):
            
            def is_in_voc(word):
                try:
                    if Tweets.vocabulary[word] == 1:
                        return 1
                except:
                    return 0

            tokens = Tweets.tokenize(row.text)
            tagged = pos_tag(tokens) 
            if lowercase:
                tagged = [(token, POS) if Tweets.emoticon_re.search(token) else
                (token.lower(), POS) for token, POS in tagged]
            words = [(w,e) for w,e in tagged if not w in Tweets.stop_words 
                     and not w.startswith(('@', 'http', 'https:// ',
                                           'http:// ', 'https://www. ', 
                                           'http://www. ', 'rt', '*http'))
                     and len(w) > 1]
            words = [(word, POS) for word, POS in words if is_in_voc(word) == 1 
                         and word.isalpha() and not word.isdigit()]   
            if Stemming == True:
                words = [(Tweets.porter.stem(word), POS) for word, POS in words]
            if Lemmatization == True:
                words = [(Tweets.lemmatizer.lemmatize(word), POS) for word, POS
                         in words]     
            if POS == False:
                words = [word for word, POS in words]
         
            return words

        self.df_preprocessed.text = self.df_preprocessed.apply(
                                        lambda x:fun(x, lowercase, POS, 
                                        Stemming, Lemmatization),axis=1)
        
        
    def preprocess_sent(self, lowercase = True, POS = True, Stemming = False, 
                   Lemmatization = True):
        
        """
        
        Preprocess and tokenize the tweets, but keep sentence structure. 
        
        Args:
            lowercase: Lower case indicator 
            POS: Part-of-Speech tagging indicator
            Stemming: Stemming indicator
            Lemmatization: Lemmatization indicator
            
        Returns:
            The preprocessed and tokenized tweets
            
        """
              
        def fun(row, lowercase, POS, Stemming, Lemmatization):
            
            def is_in_voc(word):
                try:
                    if Tweets.vocabulary[word] == 1:
                        return 1
                except:
                    return 0
                
            wordsList = []
            tokenized = sent_tokenize(row.text) 
            for i in tokenized: 
                tokens = Tweets.tokenize(i)
                tagged = pos_tag(tokens) 
                if lowercase:
                    tagged = [(token, POS) if Tweets.emoticon_re.search(token) else
                    (token.lower(), POS) for token, POS in tagged]
                words = [(w,e) for w,e in tagged if not w in Tweets.stop_words 
                         and not w.startswith(('@', 'http', 'https:// ',
                                               'http:// ', 'https://www. ', 
                                               'http://www. ', 'rt', '*http'))
                         and len(w) > 1]
                words = [(word, POS) for word, POS in words if is_in_voc(word) == 1 
                         and word.isalpha() and not word.isdigit()]   
                if Stemming == True:
                    words = [(Tweets.porter.stem(word), POS) for word, POS in words]
                if Lemmatization == True:
                    words = [(Tweets.lemmatizer.lemmatize(word), POS) for word, POS
                             in words]     
                if POS == False:
                    words = [word for word, POS in words]
                wordsList.append(words)
            return wordsList

        self.df_preprocessed.text = self.df_preprocessed.apply(
                                        lambda x:fun(x, lowercase, POS, 
                                        Stemming, Lemmatization),axis=1)

    def clean(self):
        
        """
        
        Clean the tweets
        
        """
    
        def fun(row):
            row.text = row.text.replace("*", "")
            wordsList = []
            chars = ["/", '-', '%', '.com', '&', '=']
            tokens = row.text.split()     
            index = 0
            indicator = False
            while index <= (len(tokens)-1):
                if  (any(e in tokens[index] for e in chars) and indicator == False):
                    wordsList.append(tokens[index])
                    indicator = True
                    index = index + 1        
                elif (any(e in tokens[index] for e in chars) and indicator == True and
                      wordsList[-1].startswith(('http', 'https:// ',
                              'http:// ', 'https://www. ', 
                              'http://www. ', '*http', 'www.', 'pic'))):
                    word = wordsList[-1] + tokens[index]
                    del wordsList[-1]
                    wordsList.append(word)
                    indicator = True
                    index = index + 1
                else:
                    wordsList.append(tokens[index])
                    if tokens[index].startswith(('http', 'https:// ',
                      'http:// ', 'https://www. ', 
                      'http://www. ', '*http', 'www.')):
                        indicator = True
                    else:
                        indicator = False
                    index = index + 1
            text = " ".join(wordsList)
            stripped = re.sub(Tweets.combined_pat, '', text)
            handled = Tweets.pattern.sub(lambda x: Tweets.dic[x.group()], stripped)
            handled = handled.replace("#", "")
           
            return handled
        
        self.df_preprocessed.text = self.df_preprocessed.apply(lambda x: fun(x), axis=1)        
    
    
    def __open(self, fname):
        
        """
        
        Read the formatted tweets file and store the data in a dataframe.
        
        Args:
            fname: The file with the tweets data
            
        Returns:
            The tweets data stored in a dataframe
            
        """
        
        mydateparser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
        df = pd.read_csv(fname, index_col=0, date_parser=mydateparser, sep = "\t")
        
        return df
    
        
    def __read_file(self, fname):
        
        """
        
        Read raw tweets file and store the data in a dataframe. Dataframe
        is saved in the same location as the raw file
        
        Args:
            fname: The raw file with the tweets
            
        Returns:
            The tweets data stored in a dataframe
            
        """
        
        def func(row, format_):
            return pd.datetime.strptime(row['date'], format_)
    
        elements_keys = ['text', 'date', 'user']
        elements = defaultdict(list)
        with open(fname, 'r') as file:
            i = 0
            lines = file.read().split("\n")
            for line in lines:
                try:
                    Tweets = json.loads(line)
                    for key in elements_keys:
                        elements[key].append(str(Tweets[key]))
                except:
                    continue
                i = i + 1
                if i%10000==0:
                    print (i)

        file.close()
        df_tweets=pd.DataFrame({'date': pd.Index(elements['date']),
                                'text': pd.Index(elements['text']),
                                'user': pd.Index(elements['user'])})
        df_tweets['date'] = df_tweets.apply(lambda x: func(x, '%Y-%m-%d %H:%M:%S'), axis=1)
        df_tweets.index.names = ['date'] 
        df_tweets = df_tweets.sort_values('date')
        path = str(up(up(up(__file__)))) + "/tweetsRawData/"
        df_tweets.to_csv(str(path) + 'Tweets.txt', sep='\t', index = False)
        
        return df_tweets
    
    
    def wordcloud_draw(self, color = 'black'):
        
        """
        
        Represent the most frequent words on a cloud
        
        Args:
            color: Background color of the cloud
            
        """
    
        words = ' '.join(self.df_preprocessed.text)
        cleaned_word = " ".join([word for word in words.split()
                                if 'http' not in word
                                    and not word.startswith('@')
                                    and not word.startswith('#')
                                    and word != 'RT'])
        wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color=color,
                          width=1600, height=800,max_font_size=200
                         ).generate(cleaned_word)
        plt.figure(1,figsize=(13, 13))
        plt.imshow(wordcloud)
        plt.axis('off')


    def resample(self, frequency):
        
        """
        
        Resample the data by concatenating the tweets at the given frequency
        
        Args:
            frequency: The resampling frequency
            
        """    
        
        if frequency == 'H':
            self.df_preprocessed.index = self.df_preprocessed.apply(lambda x: func(x, 'H', True), axis=1)
        elif frequency == 'D':
            self.df_preprocessed.index = self.df_preprocessed.apply(lambda x: func(x, 'D', True), axis=1)   
        elif frequency == 'min':
            self.df_preprocessed.index = self.df_preprocessed.apply(lambda x: func(x, 'min', True), axis=1)
        else:
            print ("invalid frequency")
            
        self.df_preprocessed = self.df_preprocessed.groupby(
                                self.df_preprocessed.index)['text'].agg({
                                        'text' : lambda col: ' '.join(col), 
                                        'count': 'count'})
    

    def resample_sentiment(self, frequency):
        
        """
        
        Resample the data by concatenating the tweets at the given frequency, but keeping the distinction for each different tweet
        
        Args:
            frequency: The resampling frequency
            
        """    
        if frequency == 'H':
            self.df_preprocessed.index = self.df_preprocessed.apply(lambda x: func(x, 'H', False), axis=1)
        elif frequency == 'D':
            self.df_preprocessed.index = self.df_preprocessed.apply(lambda x: func(x, 'D', False), axis=1)   
        elif frequency == 'min':
            self.df_preprocessed.index = self.df_preprocessed.apply(lambda x: func(x, 'min', False), axis=1)
        else:
            print ("invalid frequency")
         
        grouped = self.df_preprocessed.groupby(self.df_preprocessed.index)
        data = defaultdict(list)
        for name,group in grouped:
            data['Date'].append(name)
            data['count'].append(len(group))
            temp = []
            for i in range(len(group)):
                if len(group['text'][i]) != 0:
                    temp.append(group['text'][i])
            data['text'].append(temp)

        data = pd.DataFrame(data)
        pdb.set_trace()
        self.df_preprocessed = data
        

    def data_analysis(self, freq):
        
        """
        
        Data analysis of the tweets. Plot various charts and
        
        Args:
            freq: The frequency at which the tweets are counted
            
        """        
        
        def resample(df, frequency):
            if frequency == 'H':
                df['date_'] = df.apply(lambda x: func(x, 'H', False), axis=1)
            elif frequency == 'D':
                df['date_'] = df.apply(lambda x: func(x, 'D', False), axis=1)   
            elif frequency == 'M':
                df['date_'] = df.apply(lambda x: func(x, 'M', False), axis=1) 
            else:
                print ("invalid frequency")
        
            return df

        years = mdates.YearLocator()  # every year
        yearsFmt = mdates.DateFormatter('%Y-%m-%d')
        
        # Number of Tweets per day/hour
        self.df_preprocessed = resample(self.df_preprocessed, freq)    
        count = self.df_preprocessed.groupby(['date_']).size()
        # Plot 
        fig, ax = plt.subplots()
        ax.plot(count.index, count)
        # format the ticks
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(yearsFmt)
        plt.xticks(rotation=45)
        ax.grid(which='both', axis='x')
        plt.ylabel('Number of Tweets')
        plt.title('Evolution of the number of Tweets per month')  
        plt.savefig('./figures/n_tweets.jpg', bbox_inches='tight', pad_inches=1)
        self.df_preprocessed.drop('date_', inplace=True, axis=1)
        
        self.df_preprocessed.text = [str(t) for t in self.df_preprocessed.text]
        self.df_preprocessed['pre_clean_len'] = [len(t) for t in self.df_preprocessed.text]
        data_dict = {
            'text':{
                'type':self.df_preprocessed.text.dtype,
                'description':'tweet text'
            },
            'pre_clean_len':{
                'type':self.df_preprocessed.pre_clean_len.dtype,
                'description':'Length of the tweet before cleaning'
            },
            'dataset_shape':self.df_preprocessed.shape
            }
    
        pprint(data_dict)   
        
        fig, ax = plt.subplots(figsize=(5, 5))
        plt.boxplot(self.df_preprocessed.pre_clean_len)
        plt.ylabel('Number of characters')
        plt.title('Distribution of the number of characters of each tweet') 
        ax.set_xticklabels([])
        plt.savefig('./figures/length_tweets.jpg', bbox_inches='tight', pad_inches=1)
        print(self.df_preprocessed[self.df_preprocessed.pre_clean_len > 200].text.head(10))
     
        self.wordcloud_draw()
        plt.savefig('./figures/cloud.jpg', bbox_inches='tight', pad_inches=1)    
        cvec = CountVectorizer()
        cvec.fit(self.df_preprocessed.text)
        print(len(cvec.get_feature_names()))
        document_matrix = cvec.transform(self.df_preprocessed.text)
    
        batches = np.linspace(0,self.df_preprocessed.text.size,1000).astype(int)
        i=0
        tf = []
        while i < len(batches)-1:
            batch_result = np.sum(document_matrix[batches[i]:batches[i+1]].toarray(),axis=0)
            tf.append(batch_result)
            if (i % 10 == 0) | (i == len(batches)-2):
                print (batches[i+1]),"entries term frequency calculated"
            i += 1    
        tot = np.sum(tf,axis=0)
        term_freq_df = pd.DataFrame([tot],columns=cvec.get_feature_names()).transpose()
        term_freq_df.columns = ['total']
        # Zip's law
        y = np.arange(500)
        plt.figure(figsize=(10,8))
        s = 1
        expected_zipf = [term_freq_df.sort_values(by='total', ascending=False)['total'][0]/(i+1)**s for i in y]
        plt.bar(y, term_freq_df.sort_values(by='total', ascending=False)['total'][:500], align='center', alpha=0.5)
        plt.plot(y, expected_zipf, color='r', linestyle='--',linewidth=2,alpha=0.5)
        plt.ylabel('Frequency')
        plt.title('Top 500 tokens in tweets')
        plt.gca().legend(("Zipf's Law", "Frequency"))
        plt.savefig('./figures/zipf_law.jpg', bbox_inches='tight', pad_inches=1)
        term_freq_df.sort_values(by='total', ascending=False).iloc[:10]
        y = np.arange(50)
        plt.figure(figsize=(12,10))
        plt.bar(y, term_freq_df.sort_values(by='total', ascending=False)['total'][:50], align='center', alpha=0.5)
        plt.xticks(y, term_freq_df.sort_values(by='total', ascending=False)['total'][:50].index,rotation='vertical')
        plt.ylabel('Frequency')
        plt.xlabel('Top 50 tokens')
        plt.title('Top 50 tokens in sample tweets')
        plt.savefig('./figures/top_tweet.jpg', bbox_inches='tight', pad_inches=1)
        