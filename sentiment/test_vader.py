from termcolor import colored
import pdb
import time
import data.data_tweets as d
import sentiment.vader as v


def main():
    """
    Execute matching action for testing
    """
    nb_processes = 4
    start = time.time()
    sentiments = ['positive', 'negative', 'objective', 'compound']
 
    # Daily data
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
    print(colored('Daily Correlation with positive score: ' + str(vader_daily.get_correlation('positive')) + '\n', 'yellow'))
    print(colored('Daily Correlation with negative score: ' + str(vader_daily.get_correlation('negative')) + '\n', 'yellow'))
    print(colored('Daily Correlation with objective score: ' + str(vader_daily.get_correlation('objective')) + '\n', 'yellow'))
    print(colored('-----------------------------------------------------------------------------------', 'white'))
    
    vader_daily.trailing_correl(20)
    vader_daily.plot_trailing_corr(title='trailing_corr_D_Vader_20')
    vader_daily.trailing_correl(50)
    vader_daily.plot_trailing_corr(title='trailing_corr_D_Vader_50')
    vader_daily.trailing_correl(200)
    vader_daily.plot_trailing_corr(title='trailing_corr_D_Vader_200')
    vader_daily.trailing_correl(500)
    vader_daily.plot_trailing_corr(title='trailing_corr_D_Vader_500')
    print(colored('\n---------------------------------------------------------', 'cyan'))
    print(colored('Granger causality test: \n', 'cyan'))
    vader_daily.granger_causality()
    vader_daily.standardize()
    
    # Hourly data
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_sentiment_vader_2H.txt"
    print('\n')
    print(colored('-----------------------------------', 'red'))
    print(colored('Hourly data for Vader', 'red'))
    print(colored('-----------------------------------', 'red'))
    freq = 'H'
    threshold =  0.0000
    print('\n')
    print(colored('preprocessing...\n', 'magenta'))
    data = d.data_tweets(fname, freq)
    print(colored('done! \n', 'magenta'))
    print(colored('sentiment analysis...\n', 'magenta'))
    vader_hourly = v.vader()
    print(colored(str(vader_hourly.scaling) + ' batches to run \n', 'magenta'))
    vader_hourly.run("calculate_sentiments", nb_processes, data, type_='sentiment')
    print(colored('\ndone! \n', 'magenta'))
    print(colored('Hourly Correlation with positive score: ' + str(vader_hourly.get_correlation('positive')) + '\n', 'yellow'))
    print(colored('Hourly Correlation with negative score: ' + str(vader_hourly.get_correlation('negative')) + '\n', 'yellow'))
    print(colored('Hourly Correlation with objective score: ' + str(vader_hourly.get_correlation('objective')) + '\n', 'yellow'))
    print(colored('-----------------------------------------------------------------------------------', 'white'))
    vader_hourly.trailing_correl(50)
    vader_hourly.plot_trailing_corr(title='trailing_corr_H_Vader_50')
    vader_hourly.trailing_correl(200)
    vader_hourly.plot_trailing_corr(title='trailing_corr_H_Vader_200')
    vader_hourly.trailing_correl(1000)
    vader_hourly.plot_trailing_corr(title='trailing_corr_H_Vader_1000')
    vader_hourly.trailing_correl(5000)
    vader_hourly.plot_trailing_corr(title='trailing_corr_H_Vader_5000')
    print(colored('\n---------------------------------------------------------', 'cyan'))
    print(colored('Granger causality test: \n', 'cyan'))
    vader_hourly.granger_causality()
    vader_hourly.standardize()
    
    #plot figures at last because of this shit parallel package that does strange stuffs
    vader_daily.plot_sentiment(title='Daily_Vader')
    vader_hourly.plot_sentiment(title='Hourly_Vader')

    end = time.time()
    print('-----------------------------------------------------------')
    print('Running time in seconds: ')
    print(end - start)

if __name__ == '__main__':
    main()
