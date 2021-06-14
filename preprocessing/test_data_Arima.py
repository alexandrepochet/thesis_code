from data.data_tweets import data_tweets
from data.data_currency import data_currency
from utils.utils import add_lags
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
    df = df.drop(['open', 'high', 'low', 'volume', 'open_bid_ask', 'close', 'close_bid_ask', 'Direction'], axis=1)
    df = add_lags(df, 10)
    df = df.drop(['Return'], axis=1)
    freq = 'D'
    path = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/"
    file = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_D.txt"
    tweets = data_tweets(file, freq).get_df()
    tweets = tweets.drop(['count', 'text'], axis=1)
    data = df.merge(tweets, on='Date')
    data.to_csv(str(path) + 'data_Arima_' + str(freq) + '.txt', header=True, index=True, sep='\t', float_format='%.6f')
    
    # Hourly
    print('------------------------------------------------------------------------')
    print('Hourly data ')
    print('------------------------------------------------------------------------')
    file = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/forexDATA/currency.txt"
    data_curr = data_currency(file) 
    # Daily frequency
    print("resampling to daily...\n")
    data_curr.resample('H')
    threshold = 0.0000
    data_curr.define_threshold(threshold)
    size = data_curr.get_length()
    df = data_curr.get_df()
    df = df.drop(['open', 'high', 'low', 'volume', 'open_bid_ask', 'close', 'close_bid_ask', 'Direction'], axis=1)
    df = add_lags(df, 10)
    df = df.drop(['Return'], axis=1)
    freq = 'H'
    path = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/"
    file = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_H.txt"
    tweets = data_tweets(file, freq).get_df()
    tweets = tweets.drop(['count', 'text'], axis=1)
    data = df.merge(tweets, on='Date')
    data.to_csv(str(path) + 'data_Arima_' + str(freq) + '.txt', header=True, index=True, sep='\t', float_format='%.6f')
    
    end = time.time()
    print(end - start)
     
if __name__ == '__main__':
    main()



