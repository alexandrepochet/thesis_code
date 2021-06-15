import pdb
import time
import data.data_tweets as dat
import disambiguation.disambiguation as d


def main():
    """
    Execute matching action for testing
    """
    nb_processes = 4
    start = time.time()
    
    # Daily
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_sentiment_pos_2D.txt"
    freq = 'D'
    file = 'disambiguation_D'
    print('preprocessing...\n')
    data = dat.data_tweets(fname, freq)
    disambiguation = d.disambiguation()
    #disambiguation.run("disambiguateWordSenses", nb_processes, data, 'max_similarity', file, type_='disambiguation')
    disambiguation.run("disambiguateWordSenses", nb_processes, data, 'simplified_lesk', file, type_='disambiguation')

    # Hourly
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_sentiment_pos_2H.txt"
    freq = 'H'
    file = 'disambiguation_H'
    print('preprocessing...\n')
    data = dat.data_tweets(fname, freq)
    
    disambiguation = d.disambiguation()
    #disambiguation.run("disambiguateWordSenses", nb_processes, data, 'max_similarity', file, type_='disambiguation')
    disambiguation.run("disambiguateWordSenses", nb_processes, data, 'simplified_lesk', file, type_='disambiguation')

    end = time.time()
    print(end - start)

if __name__ == '__main__':
    main()
