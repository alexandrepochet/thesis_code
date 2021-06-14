import time
from preprocessing.tweet import tweet_format_raw
from preprocessing.tweet import tweet_analysis
import pdb


def main():
    """
    Execute matching action for testing
    """
    start = time.time()
    raw_file = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/tweetsRawData/tweets_eurusd.txt"
    obj = tweet_format_raw()
    data = obj.format_file(raw_file)
    data = data.set_index('date')
    data_overlap = data['2013-01-02 00:00:00':"2017-09-19 09:00:00"]
    del data
    obj = tweet_analysis()
    obj.data_analysis(data_overlap, 'min')
    end = time.time()
    print(end - start)


if __name__ == '__main__':
    main()