from data.data_tweets import data_tweets
from data.data_sentiments import data_sentiments
import time
import pdb
import numpy as np


def main():
    """
    Execute matching action for testing
    """

    start = time.time()
    path = 'C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/'
    #Daily
    file_sentiment = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/daily_sentiment_series.txt"
    file = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_D.txt"
    freq = 'D'
    tweets = data_tweets(file, freq)
    sentiments = data_sentiments(file_sentiment, freq)
    df_tweets = tweets.get_df()
    df_sentiments = sentiments.get_df()
    bid_ask_previous = df_tweets.close_bid_ask_previous
    df_sentiments = df_sentiments.merge(bid_ask_previous.to_frame(), on='Date', how="left")
    df_sentiments['close_bid_ask_previous'] = df_sentiments['close_bid_ask_previous'].replace(np.nan, 0)
    title = 'daily_sentiment_series2.txt'
    df_sentiments.to_csv(str(path) + str(title), sep='\t', index=True)
    #Hourly
    file_sentiment = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/hourly_sentiment_series.txt"
    file = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_H.txt"
    freq = 'H'
    tweets = data_tweets(file, freq)
    sentiments = data_sentiments(file_sentiment, freq)
    df_tweets = tweets.get_df()
    df_sentiments = sentiments.get_df()
    bid_ask_previous = df_tweets.close_bid_ask_previous
    df_sentiments = df_sentiments.merge(bid_ask_previous.to_frame(), on='Date', how="left")
    df_sentiments['close_bid_ask_previous'] = df_sentiments['close_bid_ask_previous'].replace(np.nan, 0)
    title = 'hourly_sentiment_series2.txt'
    df_sentiments.to_csv(str(path) + str(title), sep='\t', index=True)

    end = time.time()
    print(end - start)


if __name__ == '__main__':
    main()