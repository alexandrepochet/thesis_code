import pdb
import time
import data.data_tweets as d
import sentiment.wordnet as w
from termcolor import colored
import pandas as pd
import sentiment.poms as po
import sentiment.MasterDictionary as MasterDictionary 
import sentiment.association as s
import sentiment.vader as v

def main():
    """
    Execute matching action for testing
    """
    nb_processes = 4
    start = time.time()
    path = 'C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/'
  
    # Dataframe that will contain all relevant sentiment time series
    sentiment_series = pd.DataFrame() 
    ########################################################################## still have to standardize
    ####### Daily
    # Daily Vader
   
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_sentiment_vader_2D.txt"
    print('\n')
    print(colored('-----------------------------------', 'red'))
    print(colored('Daily data for Vader', 'red'))
    print(colored('-----------------------------------', 'red'))
    freq = 'D'
    print('\n')
    print(colored('preprocessing...\n', 'magenta'))
    data = d.data_tweets(fname, freq)
    print(colored('done! \n', 'magenta'))
    print(colored('sentiment analysis...\n', 'magenta'))
    vader_daily = v.vader()
    print(colored(str(vader_daily.scaling) + ' batches to run \n', 'magenta'))
    vader_daily.run("calculate_sentiments", nb_processes, data, type_='sentiment')
    print(colored('\ndone! \n', 'magenta'))

    sentiment_series = sentiment_series.append(data.get_df())
    sentiment_series = sentiment_series.drop('text', axis=1)
    sentiment_series = sentiment_series.drop('count', axis=1)
    temp = vader_daily.get_index('positive').rename(columns = {'value':'vader_positive'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = vader_daily.get_index('negative').rename(columns = {'value':'vader_negative'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = vader_daily.get_index('objective').rename(columns = {'value':'vader_objective'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    file = 'daily_sentiment_series.txt'
    sentiment_series.to_csv(str(path) + str(file), sep='\t', index=True)
    
    # Daily data Lesk
    print('\n')
    print(colored('---------------------------------------------------------', 'red'))
    print(colored('Daily data with Lesk method for disambiguation', 'red'))
    print(colored('---------------------------------------------------------', 'red'))
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/disambiguation_D_synset_simplified_lesk.txt"
    freq = 'D'
    print('\n')
    print(colored('preprocessing...\n', 'magenta'))
    data = d.data_tweets(fname, freq)
    print(colored('done! \n', 'magenta'))
    print(colored('sentiment analysis...\n', 'magenta'))
    wordnet_daily_lesk = w.wordnet()
    print(colored(str(wordnet_daily_lesk.scaling) + ' batches to run \n', 'magenta'))
    wordnet_daily_lesk.run("calculate_sentiments", nb_processes, data, type_='sentiment')
    print(colored('\ndone! \n', 'magenta'))
    
    temp = wordnet_daily_lesk.get_index('positive').rename(columns = {'value':'wordnet_positive'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = wordnet_daily_lesk.get_index('negative').rename(columns = {'value':'wordnet_negative'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = wordnet_daily_lesk.get_index('objective').rename(columns = {'value':'wordnet_objective'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    
    file = 'daily_sentiment_series.txt'
    sentiment_series.to_csv(str(path) + str(file), sep='\t', index=True)
    
    # Daily data index
    print('\n')
    print(colored('------------------------------------------------------------------', 'red'))
    print(colored('           Daily data with for Profile of Mood States            |', 'red'))
    print(colored('------------------------------------------------------------------', 'red'))
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_sentiment_2D.txt"
    freq = 'D'
    print('\n')
    print(colored('preprocessing...\n', 'magenta'))
    data = d.data_tweets(fname, freq)
    print(colored('done! \n', 'magenta'))
    print(colored('sentiment analysis...\n', 'magenta'))
    dictionary = 'POMS'
    print(colored('creating lexicons...\n', 'magenta'))
    master_dictionary = MasterDictionary.MasterDictionary(dictionary).get_dictionary()
    sentiment_daily_index = po.poms(master_dictionary, False)
    print(colored(str(sentiment_daily_index.scaling) + ' batches to run for POMS and OF \n', 'magenta'))
    sentiment_daily_index.run("calculate_sentiments", nb_processes, data, type_='sentiment')
    print(colored('\ndone! \n', 'magenta'))
   
    temp = sentiment_daily_index.get_index('positive_OF').rename(columns = {'value':'poms_positive_OF'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_daily_index.get_index('negative').rename(columns = {'value':'poms_negative'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_daily_index.get_index('tension').rename(columns = {'value':'poms_tension'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_daily_index.get_index('anger').rename(columns = {'value':'poms_anger'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_daily_index.get_index('fatigue').rename(columns = {'value':'poms_fatigue'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_daily_index.get_index('depression').rename(columns = {'value':'poms_depression'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_daily_index.get_index('vigour').rename(columns = {'value':'poms_vigour'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_daily_index.get_index('confusion').rename(columns = {'value':'poms_confusion'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_daily_index.get_index('positive').rename(columns = {'value':'poms_positive'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    
    file = 'daily_sentiment_series.txt'
    sentiment_series.to_csv(str(path) + str(file), sep='\t', index=True)

    # Daily data, Loughran-McDonald
    print('\n')
    print(colored('----------------------------------------------------------------------------------', 'red'))
    print(colored('                     Daily data with for Loughran-McDonald POS                   |', 'red'))
    print(colored('----------------------------------------------------------------------------------', 'red'))
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_sentiment_pos_filter_2D.txt"
    freq = 'D'
    print('\n')
    print(colored('preprocessing...\n', 'magenta'))
    data = d.data_tweets(fname, freq)
    print(colored('done! \n', 'magenta'))
    dictionary = 'Loughran-McDonald'
    print(colored('sentiment analysis...\n', 'magenta'))
    master_dictionary = MasterDictionary.MasterDictionary(dictionary).get_dictionary()
    sentiment_daily_fin_pos = s.association(master_dictionary)
    print(colored(str(sentiment_daily_fin_pos.scaling) + ' batches to run \n', 'magenta'))
    sentiment_daily_fin_pos.run("calculate_sentiments", nb_processes, data, type_='sentiment')
    print(colored('\ndone! \n', 'magenta'))
    
    temp = sentiment_daily_fin_pos.get_index('positive').rename(columns = {'value':'assoc_fin_pos_positive'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_daily_fin_pos.get_index('negative').rename(columns = {'value':'assoc_fin_pos_negative'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_daily_fin_pos.get_index('litigious').rename(columns = {'value':'assoc_fin_pos_litigious'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_daily_fin_pos.get_index('constraining').rename(columns = {'value':'assoc_fin_pos_constraining'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_daily_fin_pos.get_index('uncertainty').rename(columns = {'value':'assoc_fin_pos_uncertainty'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_daily_fin_pos.get_index('strong_modal').rename(columns = {'value':'assoc_fin_pos_strong_modal'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_daily_fin_pos.get_index('moderate_modal').rename(columns = {'value':'assoc_fin_pos_moderate_modal'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_daily_fin_pos.get_index('weak_modal').rename(columns = {'value':'assoc_fin_pos_weak_modal'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    
    file = 'daily_sentiment_series.txt'
    sentiment_series.to_csv(str(path) + str(file), sep='\t', index=True)

    # Daily data, Loughran-McDonald
    print('\n')
    print(colored('-------------------------------------------------------------------------------', 'red'))
    print(colored('                     Daily data with for Loughran-McDonald                    |', 'red'))
    print(colored('-------------------------------------------------------------------------------', 'red'))
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_sentiment_2D.txt"
    freq = 'D'
    print('\n')
    print(colored('preprocessing...\n', 'magenta'))
    data = d.data_tweets(fname, freq)
    print(data.get_length())
    print(colored('done! \n', 'magenta'))
    dictionary = 'Loughran-McDonald'
    print(colored('sentiment analysis...\n', 'magenta'))
    master_dictionary = MasterDictionary.MasterDictionary(dictionary).get_dictionary()
    sentiment_daily_fin = s.association(master_dictionary)
    print(colored(str(sentiment_daily_fin.scaling) + ' batches to run \n', 'magenta'))
    sentiment_daily_fin.run("calculate_sentiments", nb_processes, data, type_='sentiment')
    print(colored('\ndone! \n', 'magenta'))
   
    temp = sentiment_daily_fin.get_index('positive').rename(columns = {'value':'assoc_fin_positive'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_daily_fin.get_index('negative').rename(columns = {'value':'assoc_fin_negative'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_daily_fin.get_index('litigious').rename(columns = {'value':'assoc_fin_litigious'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_daily_fin.get_index('constraining').rename(columns = {'value':'assoc_fin_constraining'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_daily_fin.get_index('uncertainty').rename(columns = {'value':'assoc_fin_uncertainty'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_daily_fin.get_index('strong_modal').rename(columns = {'value':'assoc_fin_strong_modal'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_daily_fin.get_index('moderate_modal').rename(columns = {'value':'assoc_fin_moderate_modal'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_daily_fin.get_index('weak_modal').rename(columns = {'value':'assoc_fin_weak_modal'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')

    file = 'daily_sentiment_series.txt'
    sentiment_series.to_csv(str(path) + str(file), sep='\t', index=True)

    # Daily data, Bing POS
    print('\n')
    print(colored('----------------------------------------------------------------------------------', 'red'))
    print(colored('                     Daily data with for Bing POS                   |', 'red'))
    print(colored('----------------------------------------------------------------------------------', 'red'))
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_sentiment_pos_filter_2D.txt"
    freq = 'D'
    print('\n')
    print(colored('preprocessing...\n', 'magenta'))
    data = d.data_tweets(fname, freq)
    print(colored('done! \n', 'magenta'))
    dictionary = 'Bing'
    print(colored('sentiment analysis...\n', 'magenta'))
    master_dictionary = MasterDictionary.MasterDictionary(dictionary).get_dictionary()
    sentiment_daily_bing_pos = s.association(master_dictionary)
    print(colored(str(sentiment_daily_bing_pos.scaling) + ' batches to run \n', 'magenta'))
    sentiment_daily_bing_pos.run("calculate_sentiments", nb_processes, data, type_='sentiment')
    print(colored('\ndone! \n', 'magenta'))

    temp = sentiment_daily_bing_pos.get_index('positive').rename(columns = {'value':'assoc_bing_pos_positive'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_daily_bing_pos.get_index('negative').rename(columns = {'value':'assoc_bing_pos_negative'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')

    file = 'daily_sentiment_series.txt'
    sentiment_series.to_csv(str(path) + str(file), sep='\t', index=True)

    # Daily data, BING
    print('\n')
    print(colored('-------------------------------------------------------------------------------', 'red'))
    print(colored('                     Daily data with for Bing                  |', 'red'))
    print(colored('-------------------------------------------------------------------------------', 'red'))
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_sentiment_2D.txt"
    freq = 'D'
    print('\n')
    print(colored('preprocessing...\n', 'magenta'))
    data = d.data_tweets(fname, freq)
    print(data.get_length())
    print(colored('done! \n', 'magenta'))
    dictionary = 'Bing'
    print(colored('sentiment analysis...\n', 'magenta'))
    master_dictionary = MasterDictionary.MasterDictionary(dictionary).get_dictionary()
    sentiment_daily_bing = s.association(master_dictionary)
    print(colored(str(sentiment_daily_bing.scaling) + ' batches to run \n', 'magenta'))
    sentiment_daily_bing.run("calculate_sentiments", nb_processes, data, type_='sentiment')
    print(colored('\ndone! \n', 'magenta'))

    temp = sentiment_daily_bing.get_index('positive').rename(columns = {'value':'assoc_bing_positive'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_daily_bing.get_index('negative').rename(columns = {'value':'assoc_bing_negative'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')

    file = 'daily_sentiment_series.txt'
    sentiment_series.to_csv(str(path) + str(file), sep='\t', index=True)
  
    ###################################################################################################################################
    ###################################################################################################################################
    ###################################################################################################################################

    ####### Hourly
    sentiment_series = pd.DataFrame() 
    # Hourly Vader
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_sentiment_vader_2H.txt"
    print('\n')
    print(colored('-----------------------------------', 'red'))
    print(colored('Hourly data for Vader', 'red'))
    print(colored('-----------------------------------', 'red'))
    freq = 'H'
    print('\n')
    print(colored('preprocessing...\n', 'magenta'))
    data = d.data_tweets(fname, freq)
    print(colored('done! \n', 'magenta'))
    print(colored('sentiment analysis...\n', 'magenta'))
    vader_hourly = v.vader()
    print(colored(str(vader_hourly.scaling) + ' batches to run \n', 'magenta'))
    vader_hourly.run("calculate_sentiments", nb_processes, data, type_='sentiment')
    print(colored('\ndone! \n', 'magenta'))

    sentiment_series = sentiment_series.append(data.get_df())
    sentiment_series = sentiment_series.drop('text', axis=1)
    sentiment_series = sentiment_series.drop('count', axis=1)
    temp = vader_hourly.get_index('positive').rename(columns = {'value':'vader_positive'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = vader_hourly.get_index('negative').rename(columns = {'value':'vader_negative'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = vader_hourly.get_index('objective').rename(columns = {'value':'vader_objective'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')

    file = 'hourly_sentiment_series.txt'
    sentiment_series.to_csv(str(path) + str(file), sep='\t', index=True)

    # Hourly data Lesk
    print('\n')
    print(colored('---------------------------------------------------------', 'red'))
    print(colored('Hourly data with Lesk method for disambiguation', 'red'))
    print(colored('---------------------------------------------------------', 'red'))
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/disambiguation_H_synset_simplified_lesk.txt"
    freq = 'H'
    print('\n')
    print(colored('preprocessing...\n', 'magenta'))
    data = d.data_tweets(fname, freq)
    print(colored('done! \n', 'magenta'))
    print(colored('sentiment analysis...\n', 'magenta'))
    wordnet_hourly_lesk = w.wordnet()
    print(colored(str(wordnet_hourly_lesk.scaling) + ' batches to run \n', 'magenta'))
    wordnet_hourly_lesk.run("calculate_sentiments", nb_processes, data, type_='sentiment')
    print(colored('\ndone! \n', 'magenta'))

    temp = wordnet_hourly_lesk.get_index('positive').rename(columns = {'value':'wordnet_positive'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = wordnet_hourly_lesk.get_index('negative').rename(columns = {'value':'wordnet_negative'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = wordnet_hourly_lesk.get_index('objective').rename(columns = {'value':'wordnet_objective'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')

    file = 'hourly_sentiment_series.txt'
    sentiment_series.to_csv(str(path) + str(file), sep='\t', index=True)
   
    # Hourly data index
    print('\n')
    print(colored('------------------------------------------------------------------', 'red'))
    print(colored('           Hourly data with for Profile of Mood States            |', 'red'))
    print(colored('------------------------------------------------------------------', 'red'))
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_sentiment_2H.txt"
    freq = 'H'
    print('\n')
    print(colored('preprocessing...\n', 'magenta'))
    data = d.data_tweets(fname, freq)
    print(colored('done! \n', 'magenta'))
    print(colored('sentiment analysis...\n', 'magenta'))
    dictionary = 'POMS'
    print(colored('creating lexicons...\n', 'magenta'))
    master_dictionary = MasterDictionary.MasterDictionary(dictionary).get_dictionary()
    sentiment_hourly_index = po.poms(master_dictionary, False)
    print(colored(str(sentiment_hourly_index.scaling) + ' batches to run for POMS and OF \n', 'magenta'))
    sentiment_hourly_index.run("calculate_sentiments", nb_processes, data, type_='sentiment')
    print(colored('\ndone! \n', 'magenta'))

    temp = sentiment_hourly_index.get_index('positive_OF').rename(columns = {'value':'poms_positive_OF'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_hourly_index.get_index('negative').rename(columns = {'value':'poms_negative'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_hourly_index.get_index('tension').rename(columns = {'value':'poms_tension'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_hourly_index.get_index('anger').rename(columns = {'value':'poms_anger'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_hourly_index.get_index('fatigue').rename(columns = {'value':'poms_fatigue'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_hourly_index.get_index('depression').rename(columns = {'value':'poms_depression'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_hourly_index.get_index('vigour').rename(columns = {'value':'poms_vigour'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_hourly_index.get_index('confusion').rename(columns = {'value':'poms_confusion'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_hourly_index.get_index('positive').rename(columns = {'value':'poms_positive'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')

    file = 'hourly_sentiment_series.txt'
    sentiment_series.to_csv(str(path) + str(file), sep='\t', index=True)

    # Hourly data, Loughran-McDonald
    print('\n')
    print(colored('----------------------------------------------------------------------------------', 'red'))
    print(colored('                     Hourly data with for Loughran-McDonald POS                   |', 'red'))
    print(colored('----------------------------------------------------------------------------------', 'red'))
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_sentiment_pos_filter_2H.txt"
    freq = 'H'
    print('\n')
    print(colored('preprocessing...\n', 'magenta'))
    data = d.data_tweets(fname, freq)
    print(colored('done! \n', 'magenta'))
    dictionary = 'Loughran-McDonald'
    print(colored('sentiment analysis...\n', 'magenta'))
    master_dictionary = MasterDictionary.MasterDictionary(dictionary).get_dictionary()
    sentiment_hourly_fin_pos = s.association(master_dictionary)
    print(colored(str(sentiment_hourly_fin_pos.scaling) + ' batches to run \n', 'magenta'))
    sentiment_hourly_fin_pos.run("calculate_sentiments", nb_processes, data, type_='sentiment')
    print(colored('\ndone! \n', 'magenta'))

    temp = sentiment_hourly_fin_pos.get_index('positive').rename(columns = {'value':'assoc_fin_pos_positive'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_hourly_fin_pos.get_index('negative').rename(columns = {'value':'assoc_fin_pos_negative'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_hourly_fin_pos.get_index('litigious').rename(columns = {'value':'assoc_fin_pos_litigious'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_hourly_fin_pos.get_index('constraining').rename(columns = {'value':'assoc_fin_pos_constraining'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_hourly_fin_pos.get_index('uncertainty').rename(columns = {'value':'assoc_fin_pos_uncertainty'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_hourly_fin_pos.get_index('strong_modal').rename(columns = {'value':'assoc_fin_pos_strong_modal'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_hourly_fin_pos.get_index('moderate_modal').rename(columns = {'value':'assoc_fin_pos_moderate_modal'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_hourly_fin_pos.get_index('weak_modal').rename(columns = {'value':'assoc_fin_pos_weak_modal'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')

    file = 'hourly_sentiment_series.txt'
    sentiment_series.to_csv(str(path) + str(file), sep='\t', index=True)

    # Hourly data, Loughran-McDonald
    print('\n')
    print(colored('-------------------------------------------------------------------------------', 'red'))
    print(colored('                     Hourly data with for Loughran-McDonald                    |', 'red'))
    print(colored('-------------------------------------------------------------------------------', 'red'))
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_sentiment_2H.txt"
    freq = 'H'
    print('\n')
    print(colored('preprocessing...\n', 'magenta'))
    data = d.data_tweets(fname, freq)
    print(data.get_length())
    print(colored('done! \n', 'magenta'))
    dictionary = 'Loughran-McDonald'
    print(colored('sentiment analysis...\n', 'magenta'))
    master_dictionary = MasterDictionary.MasterDictionary(dictionary).get_dictionary()
    sentiment_hourly_fin = s.association(master_dictionary)
    print(colored(str(sentiment_hourly_fin.scaling) + ' batches to run \n', 'magenta'))
    sentiment_hourly_fin.run("calculate_sentiments", nb_processes, data, type_='sentiment')
    print(colored('\ndone! \n', 'magenta'))

    temp = sentiment_hourly_fin.get_index('positive').rename(columns = {'value':'assoc_fin_positive'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_hourly_fin.get_index('negative').rename(columns = {'value':'assoc_fin_negative'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_hourly_fin.get_index('litigious').rename(columns = {'value':'assoc_fin_litigious'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_hourly_fin.get_index('constraining').rename(columns = {'value':'assoc_fin_constraining'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_hourly_fin.get_index('uncertainty').rename(columns = {'value':'assoc_fin_uncertainty'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_hourly_fin.get_index('strong_modal').rename(columns = {'value':'assoc_fin_strong_modal'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_hourly_fin.get_index('moderate_modal').rename(columns = {'value':'assoc_fin_moderate_modal'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_hourly_fin.get_index('weak_modal').rename(columns = {'value':'assoc_fin_weak_modal'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')

    file = 'hourly_sentiment_series.txt'
    sentiment_series.to_csv(str(path) + str(file), sep='\t', index=True)

    # Hourly data, Bing POS
    print('\n')
    print(colored('----------------------------------------------------------------------------------', 'red'))
    print(colored('                     Hourly data with for Bing POS                   |', 'red'))
    print(colored('----------------------------------------------------------------------------------', 'red'))
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_sentiment_pos_filter_2H.txt"
    freq = 'H'
    print('\n')
    print(colored('preprocessing...\n', 'magenta'))
    data = d.data_tweets(fname, freq)
    print(colored('done! \n', 'magenta'))
    dictionary = 'Bing'
    print(colored('sentiment analysis...\n', 'magenta'))
    master_dictionary = MasterDictionary.MasterDictionary(dictionary).get_dictionary()
    sentiment_hourly_bing_pos = s.association(master_dictionary)
    print(colored(str(sentiment_hourly_bing_pos.scaling) + ' batches to run \n', 'magenta'))
    sentiment_hourly_bing_pos.run("calculate_sentiments", nb_processes, data, type_='sentiment')
    print(colored('\ndone! \n', 'magenta'))

    temp = sentiment_hourly_bing_pos.get_index('positive').rename(columns = {'value':'assoc_bing_pos_positive'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_hourly_bing_pos.get_index('negative').rename(columns = {'value':'assoc_bing_pos_negative'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')

    file = 'hourly_sentiment_series.txt'
    sentiment_series.to_csv(str(path) + str(file), sep='\t', index=True)

    # Hourly data, BING
    print('\n')
    print(colored('-------------------------------------------------------------------------------', 'red'))
    print(colored('                     Hourly data with for Bing                  |', 'red'))
    print(colored('-------------------------------------------------------------------------------', 'red'))
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_sentiment_2H.txt"
    freq = 'H'
    print('\n')
    print(colored('preprocessing...\n', 'magenta'))
    data = d.data_tweets(fname, freq)
    print(data.get_length())
    print(colored('done! \n', 'magenta'))
    dictionary = 'Bing'
    print(colored('sentiment analysis...\n', 'magenta'))
    master_dictionary = MasterDictionary.MasterDictionary(dictionary).get_dictionary()
    sentiment_hourly_bing = s.association(master_dictionary)
    print(colored(str(sentiment_hourly_bing.scaling) + ' batches to run \n', 'magenta'))
    sentiment_hourly_bing.run("calculate_sentiments", nb_processes, data, type_='sentiment')
    print(colored('\ndone! \n', 'magenta'))

    temp = sentiment_hourly_bing.get_index('positive').rename(columns = {'value':'assoc_bing_positive'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')
    temp = sentiment_hourly_bing.get_index('negative').rename(columns = {'value':'assoc_bing_negative'})
    temp.index.names = ['Date']
    sentiment_series = sentiment_series.merge(temp, left_index=True, right_on='Date')

    file = 'hourly_sentiment_series.txt'
    sentiment_series.to_csv(str(path) + str(file), sep='\t', index=True)

if __name__ == '__main__':
    main()