import pdb
import time
import preprocessing.preprocess_clone as p


def main():
    """
    Execute matching action for testing
    """
    start = time.time()
    fname_tweets = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/tweetsRawData/tweets.txt"
    fname_curr = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/forexDATA/currency.txt"
    p.preprocess_sentiment(fname_tweets, True)
    p.preprocess_sentiment(fname_tweets, False)
    p.preprocess(fname_tweets, False)
    p.preprocess_sentiment(fname_tweets, True, True)
    p.preprocess_vader(fname_tweets)
    p.preprocess_currency(fname_curr, "D", 0)
    p.preprocess_currency(fname_curr, "H", 0)
    end = time.time()
    print(end - start)
    
if __name__ == '__main__':
    main()
