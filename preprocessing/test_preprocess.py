import pdb
import time
import preprocessing.preprocess as p


def main():
    """
    Execute matching action for testing
    """
    start = time.time()
   
    fname_curr = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/forexDATA/currency.txt"
    fname_tweets = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/tweetsRawData/tweets.txt"
    
    freq = 'D'
    threshold = 0.0000
    print('preprocessing daily data...\n')

    #p.preprocess_sentiment(fname_curr, fname_tweets, freq, threshold, True)
    #p.preprocess_sentiment(fname_curr, fname_tweets, freq, threshold, False)
    p.preprocess(fname_curr, fname_tweets, freq, threshold, False)
    #p.preprocess_sentiment(fname_curr, fname_tweets, freq, threshold, True, True)
    #p.preprocess_vader(fname_curr, fname_tweets, freq, threshold)
    
    freq = 'H'
    threshold = 0.0000
    print('preprocessing hourly data...\n')

    #p.preprocess_sentiment(fname_curr, fname_tweets, freq, threshold, True)
    #p.preprocess_sentiment(fname_curr, fname_tweets, freq, threshold, False)
    #p.preprocess(fname_curr, fname_tweets, freq, threshold, False)
    #p.preprocess_sentiment(fname_curr, fname_tweets, freq, threshold, True, True)
    #p.preprocess_vader(fname_curr, fname_tweets, freq, threshold)
   
    #Sentiment 140 preprocess 
    #file = 'C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/test_data/sentiment140_raw.txt'
    #p.preprocess140(file)

    
    end = time.time()
    print(end - start)
    
if __name__ == '__main__':
    main()
