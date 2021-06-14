import pdb
import time
import data.data_tweets as d
import sentiment.association as s
import sentiment.MasterDictionary as MasterDictionary 
from termcolor import colored


def main():
    """
    Execute matching action for testing
    """
    nb_processes = 4
    start = time.time()
    sentiments = ['positive', 'negative', 'litigious', 'constraining', 'uncertainty',
                  'strong_modal', 'moderate_modal', 'weak_modal']
    
    # No POS treatment
    # Daily data, Bing
    print('\n')
    print(colored('------------------------------------------------------------------', 'red'))
    print(colored('                     Daily data with for Bing                    |', 'red'))
    print(colored('------------------------------------------------------------------', 'red'))
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_sentiment_2D.txt"
    freq = 'D'
    print('\n')
    print(colored('preprocessing...\n', 'magenta'))
    data = d.data_tweets(fname, freq)
    print(colored('done! \n', 'magenta'))
    print(colored('sentiment analysis...\n', 'magenta'))
    dictionary = 'Bing'
    master_dictionary = MasterDictionary.MasterDictionary(dictionary).get_dictionary()
    sentiment_daily_bing = s.association(master_dictionary, False)
    print(colored(str(sentiment_daily_bing.scaling) + ' batches to run \n', 'magenta'))
    sentiment_daily_bing.run("calculate_sentiments", nb_processes, data, type_='sentiment')
    print(colored('\ndone! \n', 'magenta'))
    print(colored('Daily Correlation Bing with positive score: ' + str(sentiment_daily_bing.get_correlation('index', 'positive')) + '\n', 'yellow'))
    print(colored('Daily Correlation Bing with negative score: ' + str(sentiment_daily_bing.get_correlation('index', 'negative')) + '\n', 'yellow'))
    print(colored('-----------------------------------------------------------------------------------', 'white'))
    sentiment_daily_bing.trailing_correl(20, type_='index')
    sentiment_daily_bing.plot_trailing_corr(title='trailing_corr_D_Bing_20_index')
    sentiment_daily_bing.trailing_correl(50, type_='index')
    sentiment_daily_bing.plot_trailing_corr(title='trailing_corr_D_Bing_50_index')
    sentiment_daily_bing.trailing_correl(200, type_='index')
    sentiment_daily_bing.plot_trailing_corr(title='trailing_corr_daily_fin_D_Bing_200_index')
    sentiment_daily_bing.trailing_correl(500, type_='index')
    sentiment_daily_bing.plot_trailing_corr(title='trailing_corr_D_Bing_500_index')
    print(colored('\n---------------------------------------------------------', 'cyan'))
    print(colored('Granger causality test: \n', 'cyan'))
    sentiment_daily_bing.granger_causality(type_='index')
    sentiment_daily_bing.standardize(type_='index')

    # Daily data, Loughran-McDonald
    print('\n')
    print(colored('-------------------------------------------------------------------------------', 'red'))
    print(colored('                     Daily data with for Loughran-McDonald                    |', 'red'))
    print(colored('-------------------------------------------------------------------------------', 'red'))
    dictionary = 'Loughran-McDonald'
    print('\n')
    print(colored('sentiment analysis...\n', 'magenta'))
    master_dictionary = MasterDictionary.MasterDictionary(dictionary).get_dictionary()
    sentiment_daily_fin = s.association(master_dictionary, False)
    print(colored(str(sentiment_daily_fin.scaling) + ' batches to run \n', 'magenta'))
    sentiment_daily_fin.run("calculate_sentiments", nb_processes, data, type_='sentiment')
    print(colored('\ndone! \n', 'magenta'))
    for senti in sentiments:
        print(colored('Daily Loughran-McDonald Correlation with ' +  str(senti) + ' score: ' + str(sentiment_daily_fin.get_correlation('index', senti)) + '\n', 'yellow'))
    print(colored('-----------------------------------------------------------------------------------', 'white'))
    sentiment_daily_fin.trailing_correl(20, type_='index')
    sentiment_daily_fin.plot_trailing_corr(title='trailing_corr_D_Fin_20_index')
    sentiment_daily_fin.trailing_correl(50, type_='index')
    sentiment_daily_fin.plot_trailing_corr(title='trailing_corr_D_Fin_50_index')
    sentiment_daily_fin.trailing_correl(200, type_='index')
    sentiment_daily_fin.plot_trailing_corr(title='trailing_corr_D_Fin_200_index')
    sentiment_daily_fin.trailing_correl(500, type_='index')
    sentiment_daily_fin.plot_trailing_corr(title='trailing_corr_D_Fin_500_index')
    print(colored('\n---------------------------------------------------------', 'cyan'))
    print(colored('Granger causality test: \n', 'cyan'))
    sentiment_daily_fin.granger_causality(type_='index')
    sentiment_daily_fin.standardize(type_='index')

    # Hourly data, Bing
    print('\n')
    print(colored('------------------------------------------------------------------', 'red'))
    print(colored('                     Hourly data with for Bing                    |', 'red'))
    print(colored('------------------------------------------------------------------', 'red'))
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_sentiment_2H.txt"
    freq = 'H'
    print('\n')
    print(colored('preprocessing...\n', 'magenta'))
    data = d.data_tweets(fname, freq)
    print(colored('done! \n', 'magenta'))
    print(colored('sentiment analysis...\n', 'magenta'))
    dictionary = 'Bing'
    master_dictionary = MasterDictionary.MasterDictionary(dictionary).get_dictionary()
    sentiment_hourly_bing = s.association(master_dictionary, False)
    print(colored(str(sentiment_hourly_bing.scaling) + ' batches to run \n', 'magenta'))
    sentiment_hourly_bing.run("calculate_sentiments", nb_processes, data, type_='sentiment')
    print(colored('\ndone! \n', 'magenta'))
    print(colored('Hourly Correlation Bing with positive score: ' + str(sentiment_hourly_bing.get_correlation('index', 'positive')) + '\n', 'yellow'))
    print(colored('Hourly Correlation Bing with negative score: ' + str(sentiment_hourly_bing.get_correlation('index', 'negative')) + '\n', 'yellow'))
    print(colored('-----------------------------------------------------------------------------------', 'white'))
    sentiment_hourly_bing.trailing_correl(50, type_='index')
    sentiment_hourly_bing.plot_trailing_corr(title='trailing_corr_H_Bing_50_index')
    sentiment_hourly_bing.trailing_correl(200, type_='index')
    sentiment_hourly_bing.plot_trailing_corr(title='trailing_corr_H_Bing_200_index')
    sentiment_hourly_bing.trailing_correl(1000, type_='index')
    sentiment_hourly_bing.plot_trailing_corr(title='trailing_corr_H_Bing_1000_index')
    sentiment_hourly_bing.trailing_correl(5000, type_='index')
    sentiment_hourly_bing.plot_trailing_corr(title='trailing_corr_H_Bing_5000_index')
    print(colored('\n---------------------------------------------------------', 'cyan'))
    print(colored('Granger causality test: \n', 'cyan'))
    sentiment_hourly_bing.granger_causality(type_='index')
    sentiment_hourly_bing.standardize(type_='index')

    # Hourly data, Loughran-McDonald
    print('\n')
    print(colored('--------------------------------------------------------------------------------', 'red'))
    print(colored('                     Hourly data with for Loughran-McDonald                    |', 'red'))
    print(colored('--------------------------------------------------------------------------------', 'red'))
    dictionary = 'Loughran-McDonald'
    print('\n')
    print(colored('sentiment analysis...\n', 'magenta'))
    master_dictionary = MasterDictionary.MasterDictionary(dictionary).get_dictionary()
    sentiment_hourly_fin = s.association(master_dictionary, False)
    print(colored(str(sentiment_hourly_fin.scaling) + ' batches to run \n', 'magenta'))
    sentiment_hourly_fin.run("calculate_sentiments", nb_processes, data, type_='sentiment')
    print(colored('\ndone! \n', 'magenta'))
    for senti in sentiments:
         print(colored('Hourly Loughran-McDonald Correlation with ' +  str(senti) + ' score: ' + str(sentiment_hourly_fin.get_correlation('index', senti)) + '\n', 'yellow'))
    print(colored('-----------------------------------------------------------------------------------', 'white'))
    sentiment_hourly_fin.trailing_correl(50, type_='index')
    sentiment_hourly_fin.plot_trailing_corr(title='trailing_corr_H_Fin_50_index')
    sentiment_hourly_fin.trailing_correl(200, type_='index')
    sentiment_hourly_fin.plot_trailing_corr(title='trailing_corr_H_Fin_200_index')
    sentiment_hourly_fin.trailing_correl(1000, type_='index')
    sentiment_hourly_fin.plot_trailing_corr(title='trailing_corr_H_Fin_1000_index')
    sentiment_hourly_fin.trailing_correl(5000, type_='index')
    sentiment_hourly_fin.plot_trailing_corr(title='trailing_corr_H_Fin_5000_index')
    print(colored('\n---------------------------------------------------------', 'cyan'))
    print(colored('Granger causality test: \n', 'cyan'))
    sentiment_hourly_fin.granger_causality(type_='index')
    sentiment_hourly_fin.standardize(type_='index')

    ###################################################################
    # Well POS treatment
    # Daily data, Bing
    print('\n')
    print(colored('---------------------------------------------------------------------', 'red'))
    print(colored('                     Daily data with for Bing POS                   |', 'red'))
    print(colored('---------------------------------------------------------------------', 'red'))
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_sentiment_pos_filter_2D.txt"
    freq = 'D'
    print('\n')
    print(colored('preprocessing...\n', 'magenta'))
    data = d.data_tweets(fname, freq)
    print(colored('done! \n', 'magenta'))
    print(colored('sentiment analysis...\n', 'magenta'))
    dictionary = 'Bing'
    master_dictionary = MasterDictionary.MasterDictionary(dictionary).get_dictionary()
    sentiment_daily_bing_pos = s.association(master_dictionary, False)
    print(colored(str(sentiment_daily_bing_pos.scaling) + ' batches to run \n', 'magenta'))
    sentiment_daily_bing_pos.run("calculate_sentiments", nb_processes, data, type_='sentiment')
    print(colored('\ndone! \n', 'magenta')) 
    print(colored('Daily Correlation Bing POS with positive score: ' + str(sentiment_daily_bing_pos.get_correlation('index', 'positive')) + '\n', 'yellow'))
    print(colored('Daily Correlation Bing POS with negative score: ' + str(sentiment_daily_bing_pos.get_correlation('index', 'negative')) + '\n', 'yellow'))
    print(colored('-----------------------------------------------------------------------------------', 'white'))
    sentiment_daily_bing_pos.trailing_correl(20, type_='index')
    sentiment_daily_bing_pos.plot_trailing_corr(title='trailing_corr_D_pos_Bing_20_index')
    sentiment_daily_bing_pos.trailing_correl(50, type_='index')
    sentiment_daily_bing_pos.plot_trailing_corr(title='trailing_corr_D_pos_Bing_50_index')
    sentiment_daily_bing_pos.trailing_correl(200, type_='index')
    sentiment_daily_bing_pos.plot_trailing_corr(title='trailing_corr_D_pos_Bing_200_index')
    sentiment_daily_bing_pos.trailing_correl(500, type_='index')
    sentiment_daily_bing_pos.plot_trailing_corr(title='trailing_corr_D_pos_Bing_500_index')
    print(colored('\n---------------------------------------------------------', 'cyan'))
    print(colored('Granger causality test: \n', 'cyan'))
    sentiment_daily_bing_pos.granger_causality(type_='index')
    sentiment_daily_bing_pos.standardize(type_='index')

    # Daily data, Loughran-McDonald
    print('\n')
    print(colored('----------------------------------------------------------------------------------', 'red'))
    print(colored('                     Daily data with for Loughran-McDonald POS                   |', 'red'))
    print(colored('----------------------------------------------------------------------------------', 'red'))
    dictionary = 'Loughran-McDonald'
    print('\n')
    print(colored('sentiment analysis...\n', 'magenta'))
    master_dictionary = MasterDictionary.MasterDictionary(dictionary).get_dictionary()
    sentiment_daily_fin_pos = s.association(master_dictionary, False)
    print(colored(str(sentiment_daily_fin_pos.scaling) + ' batches to run \n', 'magenta'))
    sentiment_daily_fin_pos.run("calculate_sentiments", nb_processes, data, type_='sentiment')
    print(colored('\ndone! \n', 'magenta'))
    for senti in sentiments:
         print(colored('Daily Loughran-McDonald POS Correlation with ' +  str(senti) + ' score: ' + str(sentiment_daily_fin_pos.get_correlation('index', senti)) + '\n', 'yellow'))
    print(colored('-----------------------------------------------------------------------------------', 'white'))     
    sentiment_daily_fin_pos.trailing_correl(20, type_='index')
    sentiment_daily_fin_pos.plot_trailing_corr(title='trailing_corr_D_pos_Fin_20_index')
    sentiment_daily_fin_pos.trailing_correl(50, type_='index')
    sentiment_daily_fin_pos.plot_trailing_corr(title='trailing_corr_D_pos_Fin_50_index')
    sentiment_daily_fin_pos.trailing_correl(200, type_='index')
    sentiment_daily_fin_pos.plot_trailing_corr(title='trailing_corr_D_pos_Fin_200_index')
    sentiment_daily_fin_pos.trailing_correl(500, type_='index')
    sentiment_daily_fin_pos.plot_trailing_corr(title='trailing_corr_D_pos_Fin_500_index')
    print(colored('\n---------------------------------------------------------', 'cyan'))
    print(colored('Granger causality test: \n', 'cyan'))
    sentiment_daily_fin_pos.granger_causality(type_='index')
    sentiment_daily_fin_pos.standardize(type_='index')

    # Hourly data, Bing
    print('\n')
    print(colored('---------------------------------------------------------------------', 'red'))
    print(colored('                     Hourly data with for Bing POS                   |', 'red'))
    print(colored('---------------------------------------------------------------------', 'red'))
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_sentiment_pos_filter_2H.txt"
    freq = 'H'
    print('\n')
    print(colored('preprocessing...\n', 'magenta'))
    data = d.data_tweets(fname, freq)
    print(colored('done! \n', 'magenta'))
    print(colored('sentiment analysis...\n', 'magenta'))
    dictionary = 'Bing'
    master_dictionary = MasterDictionary.MasterDictionary(dictionary).get_dictionary()
    sentiment_hourly_bing_pos = s.association(master_dictionary, False)
    print(colored(str(sentiment_hourly_bing_pos.scaling) + ' batches to run \n', 'magenta'))
    sentiment_hourly_bing_pos.run("calculate_sentiments", nb_processes, data, type_='sentiment')
    print(colored('\ndone! \n', 'magenta'))
    print(colored('Hourly Correlation Bing POS with positive score: ' + str(sentiment_hourly_bing_pos.get_correlation('index', 'positive')) + '\n', 'yellow'))
    print(colored('Hourly Correlation Bing POS with negative score: ' + str(sentiment_hourly_bing_pos.get_correlation('index', 'negative')) + '\n', 'yellow'))
    print(colored('-----------------------------------------------------------------------------------', 'white'))
    sentiment_hourly_bing_pos.trailing_correl(50, type_='index')
    sentiment_hourly_bing_pos.plot_trailing_corr(title='trailing_corr_H_pos_Bing_50_index')
    sentiment_hourly_bing_pos.trailing_correl(200, type_='index')
    sentiment_hourly_bing_pos.plot_trailing_corr(title='trailing_corr_H_pos_Bing_200_index')
    sentiment_hourly_bing_pos.trailing_correl(1000, type_='index')
    sentiment_hourly_bing_pos.plot_trailing_corr(title='trailing_corr_H_pos_Bing_1000_index')
    sentiment_hourly_bing_pos.trailing_correl(5000, type_='index')
    sentiment_hourly_bing_pos.plot_trailing_corr(title='trailing_corr_H_pos_Bing_5000_index')
    print(colored('\n---------------------------------------------------------', 'cyan'))
    print(colored('Granger causality test: \n', 'cyan'))
    sentiment_hourly_bing_pos.granger_causality(type_='index')
    sentiment_hourly_bing_pos.standardize(type_='index')
   
    # Hourly data, Loughran-McDonald
    print('\n')
    print(colored('----------------------------------------------------------------------------------', 'red'))
    print(colored('                     Hourly data with for Loughran-McDonald POS                   |', 'red'))
    print(colored('----------------------------------------------------------------------------------', 'red'))
    dictionary = 'Loughran-McDonald'
    print('\n')
    print(colored('sentiment analysis...\n', 'magenta'))
    master_dictionary = MasterDictionary.MasterDictionary(dictionary).get_dictionary()
    sentiment_hourly_fin_pos = s.association(master_dictionary, False)
    print(colored(str(sentiment_hourly_fin_pos.scaling) + ' batches to run \n', 'magenta'))
    sentiment_hourly_fin_pos.run("calculate_sentiments", nb_processes, data, type_='sentiment')
    print(colored('\ndone! \n', 'magenta'))
    for senti in sentiments:
         print(colored('Hourly Loughran-McDonald POS Correlation with ' +  str(senti) + ' score: ' + str(sentiment_hourly_fin_pos.get_correlation('index', senti)) + '\n', 'yellow'))
    sentiment_hourly_fin_pos.trailing_correl(50, type_='index')
    sentiment_hourly_fin_pos.plot_trailing_corr(title='trailing_corr_H_pos_Fin_50_index')
    sentiment_hourly_fin_pos.trailing_correl(200, type_='index')
    sentiment_hourly_fin_pos.plot_trailing_corr(title='trailing_corr_H_pos_Fin_200_index')
    sentiment_hourly_fin_pos.trailing_correl(1000, type_='index')
    sentiment_hourly_fin_pos.plot_trailing_corr(title='trailing_corr_H_pos_Fin_1000_index')
    sentiment_hourly_fin_pos.trailing_correl(5000, type_='index')
    sentiment_hourly_fin_pos.plot_trailing_corr(title='trailing_corr_H_pos_Fin_5000_index')
    print(colored('\n---------------------------------------------------------', 'cyan'))
    print(colored('Granger causality test: \n', 'cyan'))
    sentiment_hourly_fin_pos.granger_causality(type_='index')
    sentiment_hourly_fin_pos.standardize(type_='index')

    #plot figures at last because of this shit parallel package that does strange stuffs
    sentiment_daily_bing.plot_sentiment(type_='index', title1='orientation_D_Bing', title2='sentiment_index_D_Bing')
    sentiment_daily_fin.plot_sentiment(type_='index', title1='orientation_D_Fin', title2='sentiment_index_D_Fin')
    sentiment_hourly_bing.plot_sentiment(type_='index', title1='orientation_H_Bing', title2='sentiment_index_H_Bing')
    sentiment_hourly_fin.plot_sentiment(type_='index', title1='orientation_H_Fin', title2='sentiment_index_H_Fin')
    sentiment_daily_bing_pos.plot_sentiment(type_='index', title1='orientation_D_pos_Bing', title2='sentiment_index_D_pos_Bing')
    sentiment_daily_fin_pos.plot_sentiment(type_='index', title1='orientation_D_pos_Fin', title2='sentiment_index_D_pos_Fin')
    sentiment_hourly_bing_pos.plot_sentiment(type_='index', title1='orientation_H_pos_Bing', title2='sentiment_index_H_pos_Bing')
    sentiment_hourly_fin_pos.plot_sentiment(type_='index', title1='orientation_H_pos_Fin', title2='sentiment_index_H_pos_Fin')

    end = time.time()
    print('-----------------------------------------------------------')
    print('Running time in seconds: ')
    print(end - start)

if __name__ == '__main__':
    main()
