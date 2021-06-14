from termcolor import colored
import pdb
import time
import data.data_tweets as d
import sentiment.wordnet as w


def main():
    """
    Execute matching action for testing
    """
    nb_processes = 4
    start = time.time()
    sentiments = ['positive', 'negative', 'objective']

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
    print(colored('Daily Correlation with positive score: ' + str(wordnet_daily_lesk.get_correlation('positive')) + '\n', 'yellow'))
    print(colored('Daily Correlation with negative score: ' + str(wordnet_daily_lesk.get_correlation('negative')) + '\n', 'yellow'))
    print(colored('Daily Correlation with objective score: ' + str(wordnet_daily_lesk.get_correlation('objective')) + '\n', 'yellow'))
    print(colored('-----------------------------------------------------------------------------------', 'white'))
    wordnet_daily_lesk.trailing_correl(20)
    wordnet_daily_lesk.plot_trailing_corr(title='trailing_corr_D_wordnet_lesk_20')
    wordnet_daily_lesk.trailing_correl(50)
    wordnet_daily_lesk.plot_trailing_corr(title='trailing_corr_D_wordnet_lesk_50')
    wordnet_daily_lesk.trailing_correl(200)
    wordnet_daily_lesk.plot_trailing_corr(title='trailing_corr_D_wordnet_lesk_200')
    wordnet_daily_lesk.trailing_correl(500)
    wordnet_daily_lesk.plot_trailing_corr(title='trailing_corr_D_wordnet_lesk_500')
    print(colored('\n---------------------------------------------------------', 'cyan'))
    print(colored('Granger causality test: \n', 'cyan'))
    wordnet_daily_lesk.granger_causality()
    wordnet_daily_lesk.standardize()

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
    print(colored('Hourly Correlation with positive score: ' + str(wordnet_hourly_lesk.get_correlation('positive')) + '\n', 'yellow'))
    print(colored('Hourly Correlation with negative score: ' + str(wordnet_hourly_lesk.get_correlation('negative')) + '\n', 'yellow'))
    print(colored('Hourly Correlation with objective score: ' + str(wordnet_hourly_lesk.get_correlation('objective')) + '\n', 'yellow'))
    print(colored('-----------------------------------------------------------------------------------', 'white'))
    wordnet_hourly_lesk.trailing_correl(50)
    wordnet_hourly_lesk.plot_trailing_corr(title='trailing_corr_H_wordnet_lesk_50')
    wordnet_hourly_lesk.trailing_correl(200)
    wordnet_hourly_lesk.plot_trailing_corr(title='trailing_corr_H_wordnet_lesk_200')
    wordnet_hourly_lesk.trailing_correl(1000)
    wordnet_hourly_lesk.plot_trailing_corr(title='trailing_corr_H_wordnet_lesk_1000')
    wordnet_hourly_lesk.trailing_correl(5000)
    wordnet_hourly_lesk.plot_trailing_corr(title='trailing_corr_H_wordnet_lesk_5000')
    print(colored('\n---------------------------------------------------------', 'cyan'))
    print(colored('Granger causality test: \n', 'cyan'))
    wordnet_hourly_lesk.granger_causality()
    wordnet_hourly_lesk.standardize()

    #plot figures at last because of this shit parallel package that does strange stuffs
    wordnet_daily_lesk.plot_sentiment(title='Daily_wordnet_lesk')
    wordnet_hourly_lesk.plot_sentiment(title='Hourly_wordnet_lesk')

    end = time.time()
    print('-----------------------------------------------------------')
    print('Running time in seconds: ')
    print(end - start)

if __name__ == '__main__':
    main()
