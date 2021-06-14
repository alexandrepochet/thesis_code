import time
#import preprocessing.currency_preprocess as c
import preprocessing.tweet_preprocess as t


def main():
    """
    Execute matching action for testing
    """
    start = time.time()
    fname_ask = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/forexDATA/eurusd_ask_full.csv"
    fname_bid = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/forexDATA/eurusd_bid_full.csv"
    currency = c.Currency(fname_ask=fname_ask, fname_bid=fname_bid)
    currency.data_analysis()
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/tweetsRawData/tweets.txt"
    tweets = t.Tweets(fname=fname)
    tweets.data_analysis('M')
    end = time.time()
    print(end - start)

if __name__ == '__main__':
    main()
