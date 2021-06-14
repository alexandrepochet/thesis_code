import time
from preprocessing.currency import currency_preprocess
from preprocessing.currency import currency_analysis


def main():
    """
    Execute matching action for testing
    """
    start = time.time()
    raw_file_ask = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/forexDATA/eurusd_ask.txt"
    raw_file_bid = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/forexDATA/eurusd_bid.txt"
    obj = currency_preprocess()
    data_ask, data_bid = obj.format_file(raw_file_ask, raw_file_bid)
    data = obj.preprocess(data_ask, data_bid, 'currency')
    obj = currency_analysis()
    data_overlap = data['2013-01-02 00:00:00':"2017-09-19 09:00:00"]
    obj.data_analysis(data_overlap)


if __name__ == '__main__':
    main()