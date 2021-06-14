import pdb
import time
import numpy as np
import pandas as pd
import sentiment.association as s
import sentiment.MasterDictionary as MasterDictionary 
from termcolor import colored
import data.data_tweets as dat
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import sentiment.vader as v
import sentiment.wordnet as w
import sentiment.poms as po

def main():
    """
    Execute matching action for testing
    """
    start = time.time()
    nb_processes = 4
    
    #Vader sentiment analysis
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/sentiment140_vader.txt"
    print('\n')
    print(colored('-----------------------------------', 'red'))
    print(colored('Sentiment140 data for Vader', 'red'))
    print(colored('-----------------------------------', 'red'))
    freq = 's'
    print('\n')
    print(colored('Downloading data...\n', 'magenta'))
    data = dat.data_tweets(fname, freq)
    y_truth = data.get_df()['sentiment']
    print(colored('done! \n', 'magenta'))
    print(colored('sentiment analysis...\n', 'magenta'))
    vader = v.vader()
    print(colored(str(vader.scaling) + ' batches to run \n', 'magenta'))
    vader.run("calculate_sentiments", nb_processes, data, type_='sentiment')
    print(colored('\ndone! \n', 'magenta'))
    #comparing predicted polarity with truth
    y_pred = vader.get_index('compound')
    y_pred.loc[y_pred['value'] > 0, 'value'] = 4
    y_pred.loc[y_pred['value'] == 0, 'value'] = 2
    y_pred.loc[y_pred['value'] < 0, 'value'] = 0
    print('Accuracy score Vader: ' + "%.2f" % (accuracy_score(y_truth,y_pred)*100) + '%')
    index = np.where(y_pred["value"] != 2)
    print("Non neutral prediction length:" + "%.2f" % (index[0].size/data.get_length()*100) + '%')
    y_pred = y_pred.loc[y_pred['value'] != 2]
    y_truth = y_truth.iloc[index]
    print('Adjusted Accuracy score Vader: ' + "%.2f" % (accuracy_score(y_truth,y_pred)*100) + '%')

    #Bing sentiment analysis
    print(colored('-------------------------------------------------------------------------', 'red'))
    print(colored('                     Sentiment140  data with for Bing                    |', 'red'))
    print(colored('-------------------------------------------------------------------------', 'red'))
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/sentiment140.txt"
    freq = 's'
    print('\n')
    print(colored('preprocessing...\n', 'magenta'))
    data = dat.data_tweets(fname, freq)
    y_truth = data.get_df()['sentiment']
    print(colored('done! \n', 'magenta'))
    print(colored('sentiment analysis...\n', 'magenta'))
    dictionary = 'Bing'
    master_dictionary = MasterDictionary.MasterDictionary(dictionary).get_dictionary()
    sentiment = s.association(master_dictionary, False)
    print(colored(str(sentiment.scaling) + ' batches to run \n', 'magenta'))
    sentiment.run("calculate_sentiments", nb_processes, data, type_='sentiment')
    print(colored('\ndone! \n', 'magenta'))
    #comparing predicted polarity with truth
    y_pred = sentiment.get_index('positive') - sentiment.get_index('negative')
    y_pred.loc[y_pred['value'] > 0, 'value'] = 4
    y_pred.loc[y_pred['value'] == 0, 'value'] = 2
    y_pred.loc[y_pred['value'] < 0, 'value'] = 0
    print('Accuracy score Bing index: '+ "%.2f" % (accuracy_score(y_truth,y_pred)*100) + '%')
    index = np.where(y_pred["value"] != 2)
    print("Non neutral prediction length:" + "%.2f" % (index[0].size/data.get_length()*100) + '%')
    y_pred = y_pred.loc[y_pred['value'] != 2]
    y_truth = y_truth.iloc[index]
    print('Adjusted Accuracy score Bing index: '+ "%.2f" % (accuracy_score(y_truth,y_pred)*100) + '%')

    #POMS sentiment analysis
    print('\n')
    print(colored('------------------------------------------------------------------', 'red'))
    print(colored('           Daily data with for Profile of Mood States            |', 'red'))
    print(colored('------------------------------------------------------------------', 'red'))
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/sentiment140.txt"
    freq = 's'
    dictionary = 'POMS'
    print('\n')
    print(colored('preprocessing...\n', 'magenta'))
    data = dat.data_tweets(fname, freq)
    y_truth = data.get_df()['sentiment']
    print(colored('done! \n', 'magenta'))
    print(colored('creating lexicons...\n', 'magenta'))
    master_dictionary = MasterDictionary.MasterDictionary(dictionary).get_dictionary()
    sentiment = po.poms(master_dictionary, False)
    print(colored(str(sentiment.scaling) + ' batches to run for POMS and OF \n', 'magenta'))
    sentiment.run("calculate_sentiments", nb_processes, data, type_='sentiment')
    print(colored('\ndone! \n', 'magenta'))
    #comparing predicted polarity with truth
    y_pred = sentiment.get_index('positive_OF') - sentiment.get_index('negative') 
    y_pred.loc[y_pred['value'] > 0, 'value'] = 4
    y_pred.loc[y_pred['value'] == 0, 'value'] = 2
    y_pred.loc[y_pred['value'] < 0, 'value'] = 0
    print('Accuracy score OF index: ' + "%.2f" % (accuracy_score(y_truth,y_pred)*100) + '%')
    index = np.where(y_pred["value"] != 2)
    print("Non neutral prediction length:" + "%.2f" % (index[0].size/data.get_length()*100) + '%')
    y_pred = y_pred.loc[y_pred['value'] != 2]
    y_truth = y_truth.iloc[index]
    print('Adjusted Accuracy score OF index: ' + "%.2f" % (accuracy_score(y_truth,y_pred)*100) + '%')
  
    #WordNet sentiment analysis
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/disambiguation_sentiment140_synset_simplified_lesk.txt"
    print('\n')
    print(colored('-----------------------------------', 'red'))
    print(colored('Sentiment140 data for SentiWordNet', 'red'))
    print(colored('-----------------------------------', 'red'))
    freq = 's'
    print('\n')
    print(colored('Downloading data...\n', 'magenta'))
    data = dat.data_tweets(fname, freq)
    y_truth = data.get_df()['sentiment']
    print(colored('done! \n', 'magenta'))
    print(colored('sentiment analysis...\n', 'magenta'))
    wordnet = w.wordnet()
    print(colored(str(wordnet) + ' batches to run \n', 'magenta'))
    wordnet.run("calculate_sentiments", nb_processes, data, type_='sentiment')
    print(colored('\ndone! \n', 'magenta'))
    #comparing predicted polarity with truth
    y_pred = wordnet.get_index('positive') - wordnet.get_index('negative')
    y_pred.loc[y_pred['value'] > 0, 'value'] = 4
    y_pred.loc[y_pred['value'] == 0, 'value'] = 2
    y_pred.loc[y_pred['value'] < 0, 'value'] = 0
    print('Accuracy score SentiWordNet: ' + "%.2f" % (accuracy_score(y_truth,y_pred)*100) + '%')
    index = np.where(y_pred["value"] != 2)
    print("Non neutral prediction length:" + "%.2f" % (index[0].size/data.get_length()*100) + '%')
    y_pred = y_pred.loc[y_pred['value'] != 2]
    y_truth = y_truth.iloc[index]
    print('Adjusted Accuracy score SentiWordNet: ' + "%.2f" % (accuracy_score(y_truth,y_pred)*100) + '%')

if __name__ == '__main__':
    main()
