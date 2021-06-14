import pdb
import time
import preprocessing.preprocess as p
import sentiment.sentiment as s


def main():

    """

    Execute matching action for testing

    """

    start = time.time()
    sentiments = ['positive', 'negative', 'litigious', 'constraining', 'uncertainty',
                  'strong_modal', 'moderate_modal', 'weak_modal']
    # No POS treatment
    # Daily data, Bing

    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_sentiment_D.txt"
    freq = 'D'
    print('preprocessing...\n')
    df_tweets = p.get_preprocessed_data(fname, freq)
    dictionary = 'Bing'
    sentiment = s.Sentiment(df_tweets)
    sentiment.calculate_sentiments(dictionary)
    sentiment.calculate_orientation()
    sentiment.correl()
    sentiment.plot_sentiment(type_='association', title1='association_D_Bing2', title2='sentiment_association_D_Bing2')
    print('Daily Bing Correlation with positive score: ' + str(sentiment.correlation['positive']) + '\n')
    print('Daily Bing Correlation with negative score: ' + str(sentiment.correlation['negative']) + '\n')
    sentiment.trailing_correl(20)
    sentiment.plot_trailing_corr(title='trailing_corr_D_Bing_202')
    sentiment.trailing_correl(50)
    sentiment.plot_trailing_corr(title='trailing_corr_D_Bing_502')
    sentiment.trailing_correl(200)
    sentiment.plot_trailing_corr(title='trailing_corr_D_Bing_2002')
    sentiment.trailing_correl(500)
    sentiment.plot_trailing_corr(title='trailing_corr_D_Bing_5002')
    pdb.set_trace()
    # Daily data, Loughran-McDonald
    dictionary = 'Loughran-McDonald'
    sentiment.calculate_sentiments(dictionary)
    sentiment.calculate_orientation()
    sentiment.correl()
    sentiment.plot_sentiment(type_='association', title1='association_D_Fin', title2='sentiment_association_D_Fin')
    for senti in sentiments:
         print('Daily Loughran-McDonald Correlation with ' +  str(senti) + ' score: ' + str(sentiment.correlation[senti]) + '\n')
    sentiment.trailing_correl(20)
    sentiment.plot_trailing_corr(title='trailing_corr_D_Fin_20')
    sentiment.trailing_correl(50)
    sentiment.plot_trailing_corr(title='trailing_corr_D_Fin_50')
    sentiment.trailing_correl(200)
    sentiment.plot_trailing_corr(title='trailing_corr_D_Fin_200')
    sentiment.trailing_correl(500)
    sentiment.plot_trailing_corr(title='trailing_corr_D_Fin_500')

     # Hourly data
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_sentiment_H.txt"
    freq = 'H'
    print('preprocessing...\n')
    df_tweets = p.get_preprocessed_data(fname, freq)
    dictionary = 'Bing'
    sentiment = s.Sentiment(df_tweets)
    sentiment.calculate_sentiments(dictionary)
    sentiment.calculate_orientation()
    sentiment.correl()
    sentiment.plot_sentiment(type_='association', title1='association_H_Bing', title2='sentiment_association_H_Bing')
    print('Hourly Bing Correlation with positive score: ' + str(sentiment.correlation['positive']) + '\n')
    print('Hourly Bing Correlation with negative score: ' + str(sentiment.correlation['negative']) + '\n')
    sentiment.trailing_correl(50)
    sentiment.plot_trailing_corr(title='trailing_corr_H_Bing_50')
    sentiment.trailing_correl(200)
    sentiment.plot_trailing_corr(title='trailing_corr_H_Bing_200')
    sentiment.trailing_correl(1000)
    sentiment.plot_trailing_corr(title='trailing_corr_H_Bing_1000')
    sentiment.trailing_correl(5000)
    sentiment.plot_trailing_corr(title='trailing_corr_H_Bing_5000')

    # Daily data, Loughran-McDonald
    dictionary = 'Loughran-McDonald'
    sentiment.calculate_sentiments(dictionary)
    sentiment.calculate_orientation()
    sentiment.correl()
    sentiment.plot_sentiment(type_='association', title1='association_H_Fin', title2='sentiment_association_H_Fin')
    for senti in sentiments:
         print('Hourly Loughran-McDonald Correlation with ' +  str(senti) + ' score: ' + str(sentiment.correlation[senti]) + '\n')
    sentiment.trailing_correl(50)
    sentiment.plot_trailing_corr(title='trailing_corr_H_Fin_50')
    sentiment.trailing_correl(200)
    sentiment.plot_trailing_corr(title='trailing_corr_H_Fin_200')
    sentiment.trailing_correl(1000)
    sentiment.plot_trailing_corr(title='trailing_corr_H_Fin_1000')
    sentiment.trailing_correl(5000)
    sentiment.plot_trailing_corr(title='trailing_corr_H_Fin_5000')

    ###################################################################
    # Well POS treatment
    # Daily data, Bing
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_sentiment_pos_filter_D.txt"
    freq = 'D'
    print('preprocessing...\n')
    df_tweets = p.get_preprocessed_data(fname, freq)
    dictionary = 'Bing'
    sentiment = s.Sentiment(df_tweets)
    sentiment.calculate_sentiments(dictionary)
    sentiment.calculate_orientation()
    sentiment.correl()
    sentiment.plot_sentiment(type_='association', title1='association_D_pos_Bing', title2='sentiment_association_D_pos_Bing')
    print('Daily Bing Correlation with positive score: ' + str(sentiment.correlation['positive']) + '\n')
    print('Daily Bing Correlation with negative score: ' + str(sentiment.correlation['negative']) + '\n')
    sentiment.trailing_correl(20)
    sentiment.plot_trailing_corr(title='trailing_corr_D_pos_Bing_20')
    sentiment.trailing_correl(50)
    sentiment.plot_trailing_corr(title='trailing_corr_D_pos_Bing_50')
    sentiment.trailing_correl(200)
    sentiment.plot_trailing_corr(title='trailing_corr_D_pos_Bing_200')
    sentiment.trailing_correl(500)
    sentiment.plot_trailing_corr(title='trailing_corr_D_pos_Bing_500')

    # Daily data, Loughran-McDonald
    dictionary = 'Loughran-McDonald'
    sentiment.calculate_sentiments(dictionary)
    sentiment.calculate_orientation()
    sentiment.correl()
    sentiment.plot_sentiment(type_='association', title1='association_D_pos_Fin', title2='sentiment_association_D_pos_Fin')
    for senti in sentiments:
         print('Daily Loughran-McDonald Correlation with ' +  str(senti) + ' score: ' + str(sentiment.correlation[senti]) + '\n')
    sentiment.trailing_correl(20)
    sentiment.plot_trailing_corr(title='trailing_corr_D_pos_Fin_20')
    sentiment.trailing_correl(50)
    sentiment.plot_trailing_corr(title='trailing_corr_D_pos_Fin_50')
    sentiment.trailing_correl(200)
    sentiment.plot_trailing_corr(title='trailing_corr_D_pos_Fin_200')
    sentiment.trailing_correl(500)
    sentiment.plot_trailing_corr(title='trailing_corr_D_pos_Fin_500')

     # Hourly data
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_sentiment_pos_filter_H.txt"
    freq = 'H'
    print('preprocessing...\n')
    df_tweets = p.get_preprocessed_data(fname, freq)
    dictionary = 'Bing'
    sentiment = s.Sentiment(df_tweets)
    sentiment.calculate_sentiments(dictionary)
    sentiment.calculate_orientation()
    sentiment.correl()
    sentiment.plot_sentiment(type_='association', title1='association_H_pos_Bing', title2='sentiment_association_H_pos_Bing')
    print('Hourly Bing Correlation with positive score: ' + str(sentiment.correlation['positive']) + '\n')
    print('Hourly Bing Correlation with negative score: ' + str(sentiment.correlation['negative']) + '\n')
    sentiment.trailing_correl(50)
    sentiment.plot_trailing_corr(title='trailing_corr_H_pos_Bing_50')
    sentiment.trailing_correl(200)
    sentiment.plot_trailing_corr(title='trailing_corr_H_pos_Bing_200')
    sentiment.trailing_correl(1000)
    sentiment.plot_trailing_corr(title='trailing_corr_H_pos_Bing_1000')
    sentiment.trailing_correl(5000)
    sentiment.plot_trailing_corr(title='trailing_corr_H_pos_Bing_5000')

    # Daily data, Loughran-McDonald
    dictionary = 'Loughran-McDonald'
    sentiment.calculate_sentiments(dictionary)
    sentiment.calculate_orientation()
    sentiment.correl()
    sentiment.plot_sentiment(type_='association', title1='association_H_pos_Fin', title2='sentiment_association_H_pos_Fin')
    for senti in sentiments:
         print('Hourly Loughran-McDonald Correlation with ' +  str(senti) + ' score: ' + str(sentiment.correlation[senti]) + '\n')
    sentiment.trailing_correl(50)
    sentiment.plot_trailing_corr(title='trailing_corr_H_pos_Fin_50')
    sentiment.trailing_correl(200)
    sentiment.plot_trailing_corr(title='trailing_corr_H_pos_Fin_200')
    sentiment.trailing_correl(1000)
    sentiment.plot_trailing_corr(title='trailing_corr_H_pos_Fin_1000')
    sentiment.trailing_correl(5000)
    sentiment.plot_trailing_corr(title='trailing_corr_H_pos_Fin_5000')

    end = time.time()
    print(end - start)


if __name__ == '__main__':
    main()
