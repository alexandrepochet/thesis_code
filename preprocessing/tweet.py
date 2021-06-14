import pdb
import string
import re
import json
from collections import defaultdict
from os.path import dirname as up
from pprint import pprint
from datetime import datetime, date, time, timedelta
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer, sent_tokenize
from nltk import pos_tag
from nltk.stem import PorterStemmer, WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib import pyplot as plt, dates as mdates
from preprocessing.replacer import AntonymReplacer
from utils.utils import load_words, initialize, flatten, func


class tweet_format_raw():

    def format_file(self, fname):
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
                    tweets = json.loads(line)
                    for key in elements_keys:
                        elements[key].append(str(tweets[key]))
                except:
                    continue
                i = i + 1
                if i%10000 == 0:
                    print(i)

        file.close()
        df_tweets = pd.DataFrame({'date': pd.Index(elements['date']),
                                  'text': pd.Index(elements['text']),
                                  'user': pd.Index(elements['user'])})
        df_tweets['date'] = df_tweets.apply(lambda x: func(x, '%Y-%m-%d %H:%M:%S'), axis=1)
        df_tweets.index.names = ['index']
        df_tweets = df_tweets.sort_values('date')
        path = str(up(up(up(__file__)))) + "/tweetsRawData/"
        df_tweets.to_csv(str(path) + 'tweets.txt', sep='\t', index=False)

        return df_tweets


class tweet_preprocess():
    """
    Object representing the tweets data. Preprocess the tweets data.
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

    def is_in_voc(self, word):
        """
        Check if a word is in the English dictionary

        Args:
            word: The word to check
        Returns:
            1 if present, 0 otherwise
        """
        try:
            if self.vocabulary[word] == 1:
                return 1
        except:
            return 0

    @staticmethod
    def tokenize(seq, tweets_tokenize=False):
        """
        Tokenize a string sequence.

        Args:
            s: Lower case indicator
            TweetsTokenize: Tweet tokenizer indicator
        Returns:
            The tokenized sequence
        """
        if tweets_tokenize is True:
            tknzr = TweetTokenizer()
            return tknzr.tokenize(seq)
        else:
            return tweet_preprocess.tokens_re.findall(seq)

    @staticmethod
    def ruler(first, second, third):
        if first == 'JJ':
            if second == 'NN' or second == 'NNS':
                return 1
            elif second == 'JJ':
                if third != 'NN' and third != 'NNS':
                    return 1
                else:
                    return 0
            else:
                return 0
        elif first == 'RB' or first == 'RBR' or first == 'RBS' :
            if second == 'JJ':
                if third != 'NN' and third != 'NNS':
                    return 1
                else:
                    return 0
            elif second == 'VB' or second == 'VBD' or second == 'VBN' or second == 'VBG':
                return 1
            else:
                return 0
        elif first == 'NN' or first == 'NNS':
            if second == 'JJ':
                if third != 'NN' and third != 'NNS':
                    return 1
                else:
                    return 0
            else:
                return 0
        else:
            return 0

    def filter_pos(self, data):

        def fun(row):
            lst = row.text
            words = []
            indicator = False
            length = len(lst)
            for ind, item in enumerate(lst):
                if ind == (length - 1):
                    pass
                else:
                    first = item[1]
                    second = lst[ind + 1][1]
                    if ind == (length - 2):
                        third = " "
                    else:    
                        third = lst[ind + 2][1]
                    if self.ruler(first, second, third) == 1:
                        if indicator is False:
                            words.append(item[0])
                            words.append(lst[ind + 1][0])
                            indicator = True
                        else:
                            words.append(lst[ind + 1][0])
                    else:
                        indicator = False
            return words

        data.text = data.apply(lambda x: fun(x), axis=1)
        return data

    def preprocess(self, data, lowercase=True, pos=True, stemming=False,
                   lemmatization=True):
        """
        Preprocess and tokenize the tweets.

        Args:
            lowercase: Lower case indicator
            pos: Part-of-Speech tagging indicator
            stemming: Stemming indicator
            lemmatization: Lemmatization indicator
        Returns:
            The preprocessed and tokenized tweets
        """
        def fun(row, lowercase, pos, stemming, lemmatization):

            words = self._treat(row.text, lowercase, pos, stemming, lemmatization)
            return words

        data.text = data.apply(lambda x: fun(x, lowercase, pos, stemming, lemmatization), axis=1)
        return data

    def preprocess_bis(self, data, lowercase=True, pos=True, stemming=False,
                        lemmatization=True):
        """
        Preprocess and tokenize the tweets.

        Args:
            lowercase: Lower case indicator
            pos: Part-of-Speech tagging indicator
            stemming: Stemming indicator
            lemmatization: Lemmatization indicator
        Returns:
            The preprocessed and tokenized tweets
        """
        def fun(row, lowercase, pos, stemming, lemmatization):

            words = self._treat_bis(row.text, lowercase, pos, stemming, lemmatization)
            return words

        data.text = data.apply(lambda x: fun(x, lowercase, pos, stemming, lemmatization), axis=1)
        return data

    def replace_neg(self, data):

        rep = AntonymReplacer()

        def fun(row):
            words = rep.negreplace(row.text)
            return words

        data.text = data.apply(lambda x: fun(x), axis=1)
        return data

    def _treat(self, text, lowercase, pos, stemming, lemmatization):
        """
        Preprocess and tokenize the tweets.

        Args:
            text: The test to preprocess and tokenize
            lowercase: Lower case indicator
            pos: Part-of-Speech tagging indicator
            stemming: Stemming indicator
            lemmatization: Lemmatization indicator
        Returns:
            The preprocessed and tokenized tweets
        """
        tokens = tweet_preprocess.tokenize(text)
        tagged = pos_tag(tokens)
        if lowercase:
            tagged = [(token, pos) if tweet_preprocess.emoticon_re.search(token) else
                      (token.lower(), pos) for token, pos in tagged]
            words = [(w, e) for w, e in tagged if not w in tweet_preprocess.stop_words
                     and not w.startswith(('@', 'http', 'https:// ',
                                           'http:// ', 'https://www. ',
                                           'http://www. ', 'rt', '*http'))
                     and len(w) > 1]
            words = [(word, pos) for word, pos in words if self.is_in_voc(word) == 1
                     and word.isalpha() and not word.isdigit()]
            if stemming is True:
                words = [(tweet_preprocess.porter.stem(word), pos) for word, pos in words]
            if lemmatization is True:
                words = [(tweet_preprocess.lemmatizer.lemmatize(word), pos) for word, pos
                         in words]
            if pos is False:
                words = [word for word, pos in words]

            return words  

    def _treat_bis(self, text, lowercase, pos, stemming, lemmatization):
        """
        Preprocess and tokenize the tweets.

        Args:
            text: The test to preprocess and tokenize
            lowercase: Lower case indicator
            pos: Part-of-Speech tagging indicator
            stemming: Stemming indicator
            lemmatization: Lemmatization indicator
        Returns:
            The preprocessed and tokenized tweets
        """
        tokens = tweet_preprocess.tokenize(text)
        for i in range(0,len(tokens)):
            if tokens[i]=="#" and i<(len(tokens)-1):
                tokens[i] = tokens[i] + tokens[i+1]
                tokens[i+1] = ""
        if lowercase:
            tokens = [token if tweet_preprocess.emoticon_re.search(token) else
                      token.lower() for token in tokens]
            words = [w for w in tokens if not w in tweet_preprocess.stop_words
                     and not w.startswith(('@', 'http', 'https:// ',
                                           'http:// ', 'https://www. ',
                                           'http://www. ', 'rt', '*http'))
                     and len(w) > 1]
            for i in range(0, len(words)):
                word = words[i]
                if word.startswith(("#")):
                    res = tweet_preprocess.tokenize(word)
                    if len(res)>1:
                        for j in range(1, len(res)):
                            if self.is_in_voc(res[j])==0:
                                words[i]=""
            words = [word for word in words if word.startswith(("#")) or (self.is_in_voc(word) == 1 
                     and word.isalpha()  and not word.isdigit())]
            if stemming is True:
                words = [tweet_preprocess.porter.stem(word) for word in words]
            if lemmatization is True:
                words = [tweet_preprocess.lemmatizer.lemmatize(word) for word
                         in words]
            if pos is False:
                words = [word for word in words]

            return words  

    def preprocess_sent(self, data, lowercase=True, pos=True, stemming=False,
                        lemmatization=True):
        """
        Preprocess and tokenize the tweets, but keep sentence structure.

        Args:
            lowercase: Lower case indicator
            pos: Part-of-Speech tagging indicator
            stemming: Stemming indicator
            lemmatization: Lemmatization indicator
        Returns:
            The preprocessed and tokenized tweets
        """
        def fun(row, lowercase, pos, stemming, lemmatization):
            words_list = []
            tokenized = sent_tokenize(row.text)
            for i in tokenized:
                words = self._treat(i, lowercase, pos, stemming, lemmatization)
                if words:
                    words_list.append(words)
                words_list = flatten(words_list)
            return words_list

        data.text = data.apply(lambda x: fun(x, lowercase, pos, stemming, lemmatization), axis=1)
        return data

    def clean(self, data):
        """
        Clean the tweets
        """
        def fun(row):
            row.text = row.text.replace("*", "")
            words_list = []
            chars = ["/", '-', '%', '.com', '&', '=']
            tokens = row.text.split()
            index = 0
            indicator = False
            while index <= (len(tokens)-1):
                if (any(e in tokens[index] for e in chars) and indicator is False):
                    words_list.append(tokens[index])
                    indicator = True
                    index = index + 1
                elif (any(e in tokens[index] for e in chars) and indicator is True and
                      words_list[-1].startswith(('http', 'https:// ',
                                                 'http:// ', 'https://www. ',
                                                 'http://www. ', '*http', 'www.', 'pic'))):
                    word = words_list[-1] + tokens[index]
                    del words_list[-1]
                    words_list.append(word)
                    indicator = True
                    index = index + 1
                else:
                    words_list.append(tokens[index])
                    if tokens[index].startswith(('http', 'https:// ',
                                                 'http:// ', 'https://www. ',
                                                 'http://www. ', '*http', 'www.')):
                        indicator = True
                    else:
                        indicator = False
                    index = index + 1
            text = " ".join(words_list)
            stripped = re.sub(tweet_preprocess.combined_pat, '', text)
            handled = tweet_preprocess.pattern.sub(lambda x: tweet_preprocess.dic[x.group()], stripped)
            #handled = handled.replace("#", "")
            return handled
            
        data = data.dropna(subset=['text'])
        data.text = data.apply(lambda x: fun(x), axis=1)
        return data

class tweet_analysis():

    def data_analysis(self, data_tweets_init, freq):
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
            elif frequency == 'min':
                df['date_'] = df.apply(lambda x: func(x, 'min', False), axis=1)
            else:
                print("invalid frequency")

            return df

        data_tweets = data_tweets_init.copy()
        years = mdates.YearLocator()  # every year
        years_fmt = mdates.DateFormatter('%Y-%m-%d')

        # Number of Tweets per day/hour
        if data_tweets.index.name != 'date':
            data_tweets = data_tweets.set_index('date')
        data_tweets = resample(data_tweets, freq)
        count = data_tweets.groupby(['date_']).size()
        # Plot
        fig, axis = plt.subplots()
        axis.plot(count.index, count)
        # format the ticks
        axis.xaxis.set_major_locator(years)
        axis.xaxis.set_major_formatter(years_fmt)
        plt.xticks(rotation=45)
        axis.grid(which='both', axis='x')
        plt.ylabel('Number of Tweets')
        plt.title('Evolution of the number of Tweets per month')
        plt.savefig('./figures/n_tweets2.jpg', bbox_inches='tight', pad_inches=1)
        data_tweets.drop('date_', inplace=True, axis=1)
        data_tweets.text = [str(t) for t in data_tweets.text]
        data_tweets['pre_clean_len'] = [len(t) for t in data_tweets.text]
        data_dict = {'text':{\
                     'type':data_tweets.text.dtype,\
                     'description':'tweet text'},\
                     'pre_clean_len':{\
                     'type':data_tweets.pre_clean_len.dtype,\
                     'description':'Length of the tweet before cleaning'},\
                     'dataset_shape':data_tweets.shape}
        pprint(data_dict)
        fig, axis = plt.subplots(figsize=(5, 5))
        plt.boxplot(data_tweets.pre_clean_len)
        plt.ylabel('Number of characters')
        plt.title('Distribution of the number of characters of each tweet')
        axis.set_xticklabels([])
        plt.savefig('./figures/length_tweets2.jpg', bbox_inches='tight', pad_inches=1)
        print(data_tweets[data_tweets.pre_clean_len > 200].text.head(10))
        self._wordcloud_draw(data_tweets)
        plt.savefig('./figures/cloud2.jpg', bbox_inches='tight', pad_inches=1)
        cvec = CountVectorizer()
        cvec.fit(data_tweets.text)
        print(len(cvec.get_feature_names()))
        document_matrix = cvec.transform(data_tweets.text)
        batches = np.linspace(0, data_tweets.text.size, 1000).astype(int)
        i = 0
        term_freq = []
        while i < len(batches)-1:
            batch_result = np.sum(document_matrix[batches[i]:batches[i+1]].toarray(), axis=0)
            term_freq.append(batch_result)
            if (i % 10 == 0) | (i == len(batches)-2):
                print(batches[i+1]) #"entries term frequency calculated"
            i += 1
        tot = np.sum(term_freq, axis=0)
        term_freq_df = pd.DataFrame([tot], columns=cvec.get_feature_names()).transpose()
        term_freq_df.columns = ['total']
        # Zip's law
        y_axis = np.arange(500)
        plt.figure(figsize=(10, 8))
        s_var = 1
        expected_zipf = [term_freq_df.sort_values(by='total',\
                         ascending=False)['total'][0]/(i+1)**s_var for i in y_axis]
        plt.bar(y_axis, term_freq_df.sort_values(by='total', ascending=False)\
                ['total'][:500], align='center', alpha=0.5)
        plt.plot(y_axis, expected_zipf, color='r', linestyle='--', linewidth=2, alpha=0.5)
        plt.ylabel('Occurrences')
        plt.title('Top 500 tokens in tweets')
        plt.gca().legend(("Zipf's Law", "Occurrences"))
        plt.savefig('./figures/zipf_law2.jpg', bbox_inches='tight', pad_inches=1)
        term_freq_df = term_freq_df.sort_values(by='total', ascending=False)
        y_axis = np.arange(50)
        plt.figure(figsize=(12, 10))
        plt.bar(y_axis, term_freq_df.sort_values(by='total', ascending=False)\
                ['total'][:50], align='center', alpha=0.5)
        plt.xticks(y_axis, term_freq_df.sort_values(by='total', ascending=False)\
                   ['total'][:50].index, rotation='vertical')
        plt.ylabel('Frequency')
        plt.xlabel('Top 50 tokens')
        plt.title('Top 50 tokens in sample tweets')
        plt.savefig('./figures/top_tweet2.jpg', bbox_inches='tight', pad_inches=1)

    def _wordcloud_draw(self, data_tweets, color='black'):
        """
        Represent the most frequent words on a cloud

        Args:
            color: Background color of the cloud
        """
        words = ' '.join(data_tweets.text)
        cleaned_word = " ".join([word for word in words.split()
                                 if 'http' not in word
                                 and not word.startswith('@')
                                 and not word.startswith('#')
                                 and word != 'RT'])
        wordcloud = WordCloud(stopwords=STOPWORDS,
                              background_color=color,
                              width=1600, height=800, max_font_size=200
                              ).generate(cleaned_word)
        plt.figure(1, figsize=(13, 13))
        plt.imshow(wordcloud)
        plt.axis('off')


class tweet():

    def resample(self, data, frequency, joiner):
        """
        Resample the data by concatenating the tweets at the given frequency
        """
        if frequency == 'H':
            data.index = data.apply(lambda x: func(x, 'H', True), axis=1)
        elif frequency == 'D':
            data.index = data.apply(lambda x: func(x, 'D', True), axis=1)
        elif frequency == 'min':
            data.index = data.apply(lambda x: func(x, 'min', True), axis=1)
        else:
            print("invalid frequency")
        data = data.dropna(subset=['text'])
        #data = data.groupby(data.index)['text'].agg({\
        #                    'text' : lambda col: joiner.join(col), 'count': 'count'})
        data = data.groupby(data.index)['text'].agg(text = lambda col: joiner.join(col), count= 'count')
        return data

    def resample_sentiment(self, data_tweets, frequency):
        """
        Resample the data by concatenating the tweets at the given frequency,
        but keeping the distinctio:
            frequency: The resampling frequency
        """
        if frequency == 'H':
            data_tweets.index = data_tweets.apply(lambda x: func(x, 'H', True), axis=1)
        elif frequency == 'D':
            data_tweets.index = data_tweets.apply(lambda x: func(x, 'D', True), axis=1)
        elif frequency == 'min':
            data_tweets.index = data_tweets.apply(lambda x: func(x, 'min', True), axis=1)
        else:
            print("invalid frequency")

        grouped = data_tweets.groupby(data_tweets.index)
        data = defaultdict(list)
        for name, group in grouped:
            data['Date'].append(name)
            data['count'].append(len(group))
            temp = []
            for i in range(len(group)):
                if len(group['text'][i]) != 0:
                    temp.append(group['text'][i])
            data['text'].append(temp)

        data = pd.DataFrame(data)
        data_tweets = data

        return data_tweets