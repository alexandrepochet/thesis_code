import pdb
import time
import data.data_tweets as d
import sentiment.poms as po
from termcolor import colored
import sentiment.MasterDictionary as MasterDictionary 


def main():
    """
    Execute matching action for testing
    """
    nb_processes = 4
    start = time.time()
    sentiments = ['anger', 'confusion', 'depression', 'fatigue', 'tension', 'vigour', 'positive']
        
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
    print(colored('Daily Correlation with positive score: ' + str(sentiment_daily_index.get_correlation('index', 'positive')), 'yellow'))
    print(colored('Daily Correlation with tension score: ' + str(sentiment_daily_index.get_correlation('index', 'tension')), 'yellow'))
    print(colored('Daily Correlation with anger score: ' + str(sentiment_daily_index.get_correlation('index', 'anger')), 'yellow'))
    print(colored('Daily Correlation with fatigue score: ' + str(sentiment_daily_index.get_correlation('index', 'fatigue')), 'yellow'))
    print(colored('Daily Correlation with depression score: ' + str(sentiment_daily_index.get_correlation('index', 'depression')) , 'yellow'))
    print(colored('Daily Correlation with vigour score: ' + str(sentiment_daily_index.get_correlation('index', 'vigour')), 'yellow'))
    print(colored('Daily Correlation with confusion score: ' + str(sentiment_daily_index.get_correlation('index', 'confusion')), 'yellow'))
    print(colored('Daily Correlation with positive OF score: ' + str(sentiment_daily_index.get_correlation('index', 'positive_OF')), 'yellow'))
    print(colored('Daily Correlation with negative OF score: ' + str(sentiment_daily_index.get_correlation('index', 'negative')) + '\n', 'yellow'))
    print(colored('-----------------------------------------------------------------------------------', 'white'))
    sentiment_daily_index.trailing_correl(20, type_='index')
    sentiment_daily_index.plot_trailing_corr(title='trailing_corr_D_POMS_20')
    sentiment_daily_index.trailing_correl(50, type_='index')
    sentiment_daily_index.plot_trailing_corr(title='trailing_corr_D_POMS_50')
    sentiment_daily_index.trailing_correl(200, type_='index')
    sentiment_daily_index.plot_trailing_corr(title='trailing_corr_D_POMS_200')
    sentiment_daily_index.trailing_correl(500, type_='index')
    sentiment_daily_index.plot_trailing_corr(title='trailing_corr_D_POMS_500')
    print(colored('\n---------------------------------------------------------', 'cyan'))
    print(colored('Granger causality test: \n', 'cyan'))
    sentiment_daily_index.granger_causality(type_='index')
    sentiment_daily_index.standardize(type_='index')

    # Hourly data index
    print('\n')
    print(colored('------------------------------------------------------------------', 'red'))
    print(colored('           Hourly data with for Profile of Mood States           |', 'red'))
    print(colored('------------------------------------------------------------------', 'red'))
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_sentiment_2H.txt"
    freq = 'H'
    print('\n')
    print(colored('preprocessing...\n', 'magenta'))
    data = d.data_tweets(fname, freq)
    print(colored('done! \n', 'magenta'))
    print(colored('sentiment analysis...\n', 'magenta'))
    sentiment_hourly_index = po.poms(master_dictionary, False)
    print(colored(str(sentiment_hourly_index.scaling) + ' batches to run for POMS and OF \n', 'magenta'))
    sentiment_hourly_index.run("calculate_sentiments", nb_processes, data, type_='sentiment')
    print(colored('\ndone! \n', 'magenta'))
    print(colored('Hourly Correlation with positive score: ' + str(sentiment_hourly_index.get_correlation('index', 'positive')), 'yellow'))
    print(colored('Hourly Correlation with tension score: ' + str(sentiment_hourly_index.get_correlation('index', 'tension')), 'yellow'))
    print(colored('Hourly Correlation with anger score: ' + str(sentiment_hourly_index.get_correlation('index', 'anger')), 'yellow'))
    print(colored('Hourly Correlation with fatigue score: ' + str(sentiment_hourly_index.get_correlation('index', 'fatigue')), 'yellow'))
    print(colored('Hourly Correlation with depression score: ' + str(sentiment_hourly_index.get_correlation('index', 'depression')) , 'yellow'))
    print(colored('Hourly Correlation with vigour score: ' + str(sentiment_hourly_index.get_correlation('index', 'vigour')), 'yellow'))
    print(colored('Hourly Correlation with confusion score: ' + str(sentiment_hourly_index.get_correlation('index', 'confusion')), 'yellow'))
    print(colored('Hourly Correlation with positive OF score: ' + str(sentiment_hourly_index.get_correlation('index', 'positive_OF')) + '\n', 'yellow'))
    print(colored('Hourly Correlation with negative OF score: ' + str(sentiment_hourly_index.get_correlation('index', 'negative')) + '\n', 'yellow'))
    print(colored('-----------------------------------------------------------------------------------', 'white'))
    sentiment_hourly_index.trailing_correl(50, type_='index')
    sentiment_hourly_index.plot_trailing_corr(title='trailing_corr_H_POMS_50')
    sentiment_hourly_index.trailing_correl(200, type_='index')
    sentiment_hourly_index.plot_trailing_corr(title='trailing_corr_H_POMS_200')
    sentiment_hourly_index.trailing_correl(1000, type_='index')
    sentiment_hourly_index.plot_trailing_corr(title='trailing_corr_H_POMS_1000')
    sentiment_hourly_index.trailing_correl(5000, type_='index')
    sentiment_hourly_index.plot_trailing_corr(title='trailing_corr_H_POMS_5000')
    print(colored('\n---------------------------------------------------------', 'cyan'))
    print(colored('Granger causality test: \n', 'cyan'))
    sentiment_hourly_index.granger_causality(type_='index')
    sentiment_hourly_index.standardize(type_='index')

    # Daily data association
    print('\n')
    print(colored('-----------------------------------------------------------------------------', 'red'))
    print(colored('           Daily data with for Profile of Mood States association            |', 'red'))
    print(colored('-----------------------------------------------------------------------------', 'red'))
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_sentiment_2D.txt"
    freq = 'D'
    print('\n')
    print(colored('preprocessing...\n', 'magenta'))
    data = d.data_tweets(fname, freq)
    print(colored('done! \n', 'magenta'))
    print(colored('sentiment analysis...\n', 'magenta'))
    dictionary = 'POMS'
    print(colored('creating lexicons...\n', 'magenta'))
    sentiment_daily_assoc = po.poms(master_dictionary, True)
    print(colored(str(sentiment_daily_assoc.scaling) + ' batches to run for POMS and OF \n', 'magenta'))
    sentiment_daily_assoc.run("calculate_sentiments", nb_processes, data, type_='sentiment')
    print(colored('\ndone! \n', 'magenta'))
    print(colored('Daily Correlation with positive score: ' + str(sentiment_daily_assoc.get_correlation('association', 'positive')), 'yellow'))
    print(colored('Daily Correlation with tension score: ' + str(sentiment_daily_assoc.get_correlation('association', 'tension')), 'yellow'))
    print(colored('Daily Correlation with anger score: ' + str(sentiment_daily_assoc.get_correlation('association', 'anger')), 'yellow'))
    print(colored('Daily Correlation with fatigue score: ' + str(sentiment_daily_assoc.get_correlation('association', 'fatigue')), 'yellow'))
    print(colored('Daily Correlation with depression score: ' + str(sentiment_daily_assoc.get_correlation('association', 'depression')) , 'yellow'))
    print(colored('Daily Correlation with vigour score: ' + str(sentiment_daily_assoc.get_correlation('association', 'vigour')), 'yellow'))
    print(colored('Daily Correlation with confusion score: ' + str(sentiment_daily_assoc.get_correlation('association', 'confusion')), 'yellow'))
    print(colored('Daily Correlation with positive OF score: ' + str(sentiment_daily_assoc.get_correlation('association', 'positive_OF')) + '\n', 'yellow'))
    print(colored('Daily Correlation with negative OF score: ' + str(sentiment_daily_assoc.get_correlation('association', 'negative')) + '\n', 'yellow'))
    print(colored('-----------------------------------------------------------------------------------', 'white'))
    sentiment_daily_assoc.trailing_correl(20, type_='association')
    sentiment_daily_assoc.plot_trailing_corr(title='trailing_corr_D_POMS_20_assoc')
    sentiment_daily_assoc.trailing_correl(50, type_='association')
    sentiment_daily_assoc.plot_trailing_corr(title='trailing_corr_D_POMS_50_assoc')
    sentiment_daily_assoc.trailing_correl(200, type_='association')
    sentiment_daily_assoc.plot_trailing_corr(title='trailing_corr_D_POMS_200_assoc')
    sentiment_daily_assoc.trailing_correl(500, type_='association')
    sentiment_daily_assoc.plot_trailing_corr(title='trailing_corr_D_POMS_500_assoc')
    print(colored('\n---------------------------------------------------------', 'cyan'))
    print(colored('Granger causality test: \n', 'cyan'))
    sentiment_daily_assoc.granger_causality(type_='association')
    sentiment_daily_assoc.standardize(type_='association')
    sentiment_daily_assoc.calculate_orientation()

    # Hourly data association
    print('\n')
    print(colored('-----------------------------------------------------------------------------', 'red'))
    print(colored('           Hourly data with for Profile of Mood States association          |', 'red'))
    print(colored('-----------------------------------------------------------------------------', 'red'))
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_sentiment_2H.txt"
    freq = 'H'
    print('\n')
    print(colored('preprocessing...\n', 'magenta'))
    data = d.data_tweets(fname, freq)
    print(colored('done! \n', 'magenta'))
    print(colored('sentiment analysis...\n', 'magenta'))
    sentiment_hourly_assoc = po.poms(master_dictionary, True)
    print(colored(str(sentiment_hourly_assoc.scaling) + ' batches to run for POMS and OF \n', 'magenta'))
    sentiment_hourly_assoc.run("calculate_sentiments", nb_processes, data, type_='sentiment')
    print(colored('\ndone! \n', 'magenta'))
    print(colored('Hourly Correlation with positive score: ' + str(sentiment_hourly_assoc.get_correlation('association', 'positive')), 'yellow'))
    print(colored('Hourly Correlation with tension score: ' + str(sentiment_hourly_assoc.get_correlation('association', 'tension')), 'yellow'))
    print(colored('Hourly Correlation with anger score: ' + str(sentiment_hourly_assoc.get_correlation('association', 'anger')), 'yellow'))
    print(colored('Hourly Correlation with fatigue score: ' + str(sentiment_hourly_assoc.get_correlation('association', 'fatigue')), 'yellow'))
    print(colored('Hourly Correlation with depression score: ' + str(sentiment_hourly_assoc.get_correlation('association', 'depression')) , 'yellow'))
    print(colored('Hourly Correlation with vigour score: ' + str(sentiment_hourly_assoc.get_correlation('association', 'vigour')), 'yellow'))
    print(colored('Hourly Correlation with confusion score: ' + str(sentiment_hourly_assoc.get_correlation('association', 'confusion')), 'yellow'))
    print(colored('Hourly Correlation with positive OF score: ' + str(sentiment_hourly_assoc.get_correlation('association', 'positive_OF')) + '\n', 'yellow'))
    print(colored('Hourly Correlation with negative OF score: ' + str(sentiment_hourly_assoc.get_correlation('association', 'negative')) + '\n', 'yellow'))
    print(colored('-----------------------------------------------------------------------------------', 'white'))
    sentiment_hourly_assoc.trailing_correl(50, type_='association')
    sentiment_hourly_assoc.plot_trailing_corr(title='trailing_corr_H_POMS_50_assoc')
    sentiment_hourly_assoc.trailing_correl(200, type_='association')
    sentiment_hourly_assoc.plot_trailing_corr(title='trailing_corr_H_POMS_200_assoc')
    sentiment_hourly_assoc.trailing_correl(1000, type_='association')
    sentiment_hourly_assoc.plot_trailing_corr(title='trailing_corr_H_POMS_1000_assoc')
    sentiment_hourly_assoc.trailing_correl(5000, type_='association')
    sentiment_hourly_assoc.plot_trailing_corr(title='trailing_corr_H_POMS_5000_assoc')
    print(colored('\n---------------------------------------------------------', 'cyan'))
    print(colored('Granger causality test: \n', 'cyan'))
    sentiment_hourly_assoc.granger_causality(type_='association')
    sentiment_hourly_assoc.standardize(type_='association')
    sentiment_hourly_assoc.calculate_orientation()

    #plot figures at last because of this shit parallel package that does strange stuffs
    sentiment_daily_index.plot_sentiment(type_='index', title1='orientation_D_POMS', title2='sentiment_index_D_POMS')
    sentiment_hourly_index.plot_sentiment(type_='index', title1='orientation_H_POMS', title2='sentiment_index_H_POMS')
    sentiment_daily_assoc.plot_sentiment(type_='association', title1='orientation_D_POMS', title2='sentiment_association_D_POMS')
    sentiment_hourly_assoc.plot_sentiment(type_='association', title1='orientation_H_POMS', title2='sentiment_index_H_POMS')

    end = time.time()
    print('-----------------------------------------------------------')
    print('Running time in seconds: ')
    print(end - start)

if __name__ == '__main__':
    main()