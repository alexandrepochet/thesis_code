import time
import preprocessing.currency_preprocess as c
import preprocessing.tweet_preprocess as t
import pdb


def main():
    """
    Execute matching action for testing
    """
    start = time.time()
    raw_file_ask = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/forexDATA/eurusd_ask_full.csv"
    raw_file_bid = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/forexDATA/eurusd_bid_full.csv"
    c.Currency(raw_file_ask=raw_file_ask, raw_file_bid=raw_file_bid)
    raw_file = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/tweetsRawData/tweets_eurusd.txt"
    t.Tweets(raw_file=raw_file)
    end = time.time()
    print(end - start)

if __name__ == '__main__':
    main()
