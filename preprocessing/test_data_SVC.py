from data.data_tweets import data_tweets
from data.data_currency import data_currency
from utils.utils import add_features, add_lags
import time
import pdb
import warnings


def main(): 
    """
    Execute matching action for testing
    """
    warnings.filterwarnings("ignore")
    start = time.time()
    
    # Daily
    print('------------------------------------------------------------------------')
    print('Daily data ')
    print('------------------------------------------------------------------------')
    file = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/forexDATA/currency.txt"
    data_curr = data_currency(file) 
    # Daily frequency
    print("resampling to daily...\n")
    data_curr.resample('D')
    threshold = 0.0000
    data_curr.define_threshold(threshold)
    size = data_curr.get_length()
    df = data_curr.get_df()
    df = add_features(df, 'close', 'high', 'low', 'volume')
    df_copy = df[['Direction', 'close_bid_ask', 'Return']]
    df = df.shift(1)
    df = df.rename(columns={"close_bid_ask": "close_bid_ask_previous"})
    df_copy2 = df[['close_bid_ask_previous']]
    df = df.drop(['open', 'high', 'low', 'volume', 'close_bid_ask_previous', 'open_bid_ask', 'Direction', 'Return'], axis=1)
    df = df.rename(columns={"close": "close_lagged"})
    df = add_lags(df, 10)
    df = df.merge(df_copy, on='Date')
    df = df.merge(df_copy2, on='Date')
    df_reduce = df['2013-01-02':"2017-09-19"]
    freq = 'D'
    path = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/"
    df_reduce.to_csv(str(path) + 'data_SVC_ST_10_full_' + str(freq) + '.txt', header=True, index=True, sep='\t', float_format='%.6f')
    df = df.drop(['Direction', 'close_bid_ask', 'close_bid_ask_previous', 'Return'], axis=1)
    file = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_D.txt"
    tweets = data_tweets(file, freq).get_df()
    tweets = tweets.drop(['count', 'text'], axis=1)
    data = df.merge(tweets, on='Date')
    data.to_csv(str(path) + 'data_SVC_test' + str(freq) + '.txt', header=True, index=True, sep='\t', float_format='%.6f')
    
    # Hourly
    print('------------------------------------------------------------------------')
    print('Hourly data ')
    print('------------------------------------------------------------------------')
    file = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/forexDATA/currency.txt"
    data_curr = data_currency(file) 
    # Hourly frequency
    print("resampling to hourly...\n")
    data_curr.resample('H')
    threshold = 0.0000
    data_curr.define_threshold(threshold)
    size = data_curr.get_length()
    df = data_curr.get_df()
    df = add_features(df, 'close', 'high', 'low', 'volume')
    df_copy = df[['Direction', 'close_bid_ask', 'Return']]
    df = df.shift(1)
    df = df.rename(columns={"close_bid_ask": "close_bid_ask_previous"})
    df_copy2 = df[['close_bid_ask_previous']]
    df = df.drop(['open', 'high', 'low', 'volume', 'close_bid_ask_previous', 'open_bid_ask', 'Direction', 'Return'], axis=1)
    df = df.rename(columns={"close": "close_lagged"})
    df = add_lags(df, 10)
    df = df.merge(df_copy, on='Date')
    df = df.merge(df_copy2, on='Date')
    df_reduce = df['2013-01-01 22:00:00':"2017-09-19 02:00:00"]
    freq = 'H'
    path = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/"
    df_reduce.to_csv(str(path) + 'data_SVC_ST_10_full_' + str(freq) + '.txt', header=True, index=True, sep='\t', float_format='%.6f')
    df = df.drop(['Direction', 'close_bid_ask', 'close_bid_ask_previous', 'Return'], axis=1)
    file = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_H.txt"
    tweets = data_tweets(file, freq).get_df()
    tweets = tweets.drop(['count', 'text'], axis=1)
    data = df.merge(tweets, on='Date')
    data.to_csv(str(path) + 'data_SVC_test' + str(freq) + '.txt', header=True, index=True, sep='\t', float_format='%.6f')
    
    end = time.time()
    print(end - start)
     
if __name__ == '__main__':
    main()



