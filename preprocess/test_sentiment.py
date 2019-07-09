import pandas as pd
import preprocess as p
import sentiment as s
import pdb
import time



def main():
    
     """
     
     Execute matching action for testing
     
     """

     start = time.time()
     # Daily data
     fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_sentiment_D.txt"
     freq = 'D'
     print ('preprocessing...\n')
     df = p.get_preprocessed_data(fname, freq)
     dictionary = 'Bing'
     sentiment =  s.Sentiment(df)
     sentiment.calculate_sentiments(dictionary)
     sentiment.plot_('association')
     pdb.set_trace()

     end = time.time()
     print(end - start)


if __name__ == '__main__':
    main()   