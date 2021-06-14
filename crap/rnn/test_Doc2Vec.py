import numpy as np
import pandas as pd
import math
import warnings
import time
import matplotlib.pyplot as plt 
from rnn.Doc2Vec import Doc_2_Vec
import data.data_tweets as d
import pdb
import ast
import itertools
from termcolor import colored

def main():
    """
    Execute matching action for testing  
    """
    nb_processes = 4
    warnings.filterwarnings("ignore")
    start = time.time()
 
    print('\n')
    print(colored('-----------------------------------', 'red'))
    print(colored('            Daily data             ', 'red'))
    print(colored('-----------------------------------', 'red'))
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_sentiment_H.txt"
    freq = 'H'
    print ('preprocessing...\n')
    data = d.data_tweets(fname, freq)
    pdb.set_trace()
    size = data.get_length()
    data_train = data.slice(0, int(0.7*size))
    documents = [ast.literal_eval(x) for x in data_train.get_text().values.tolist()]
    documents = [list(itertools.chain.from_iterable(x)) for x in documents]
    print ('creating Doc2Vec instance...\n')
    model = Doc_2_Vec(documents)
   
    window_sizes = [2, 7, 10] 
    vec_sizes = [50, 100, 200, 500]
    print ('training the different models...\n')
    for window_size in window_sizes:
        for vec_size in vec_sizes:
            print ("training model with DM = 0 and with vector size " + str(vec_size) + "...")
            model.train(dm=0, vector_size=vec_size, window=window_size, epochs=500, callbacks=None)
            print ('saving model...\n')
            model.save("C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/src/model/Doc2Vec_D_0_" + str(vec_size) + "_" + str(window_size) + ".model")
            print ("training model with DM = 1 and with vector size " + str(vec_size) + "...")
            model.train(dm=1, vector_size=vec_size, window=window_size, epochs=500, callbacks=None)
            print ('saving model...\n')
            model.save("C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/src/model/Doc2Vec_D_1_" + str(vec_size) + "_" + str(window_size) + ".model")
   
    print('\n')
    print(colored('------------------------------ -----', 'red'))
    print(colored('            Hourly data             ', 'red'))
    print(colored('------------------------------ -----', 'red'))
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_sentiment_H.txt"
    freq = 'H'
    print ('preprocessing...\n')
    data = d.data_tweets(fname, freq)
    size = data.get_length()
    data_train = data.slice(0, int(0.7*size))
    documents = [ast.literal_eval(x) for x in data_train.get_text().values.tolist()]
    documents = [list(itertools.chain.from_iterable(x)) for x in documents]
    print ('creating Doc2Vec instance...\n')
    model = Doc_2_Vec(documents)
    
    window_sizes = [2, 7, 10]
    vec_sizes = [50, 100, 200, 500]
    print ('training the different models...\n')
    for window_size in window_sizes:
        for vec_size in vec_sizes:
            print ("training model with DM = 0 and with vector size " + str(vec_size) + "...")
            model.train(dm = 0, vector_size=vec_size, window=window_size, epochs=500, callbacks=None)
            print ('saving model...\n')
            model.save("C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/src/model/Doc2Vec_H_0_" + str(vec_size) + "_" + str(window_size) + ".model")
            print ("training model with DM = 1 and with vector size " + str(vec_size) + "...")
            model.train(dm = 1, vector_size=vec_size, window=window_size
                , epochs=500, callbacks=None)
            print ('saving model...\n')
            model.save("C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/src/model/Doc2Vec_H_1_" + str(vec_size) + "_" + str(window_size) + ".model")
   
if __name__ == '__main__':
    main()
