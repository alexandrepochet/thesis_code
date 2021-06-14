import pandas as pd
import numpy as np
from rnn.Doc2Vec import Doc_2_Vec
import data.data_tweets as d
import pdb
import time
import warnings
import ast
import itertools
from numpy import save


def main(): 
    """  
    Execute matching action for testing  
    """

    warnings.filterwarnings("ignore")
    start = time.time()
    ##Daily
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_sentiment_D.txt"
    freq = 'D'
    print ('preprocessing daily...\n')
    data = d.data_tweets(fname, freq)
    size = data.get_length()
    data_val = data.slice(int(0.7*size), int(0.85*size)) 
    data_test = data.slice(int(0.85*size), size) 
    text_val = data_val.get_text()
    text_test = data_test.get_text()
    methods = ["0","1"]
    dimensions = ["50", "100", "200", "500"]
    windows = ["2", "5", "7"]
    path = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/src/model/"
    path_save = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/src/Doc2Vec_data/"

    for method, dimension, window in itertools.product(methods, dimensions, windows):
        print(method)
        print(dimension)
        print(window)
        file  = str(path) + "Doc2Vec_D_" + str(method) + "_" + str(dimension) + "_" + str(window) + ".model"
        Doc2Vec_model = Doc_2_Vec()
        Doc2Vec_model.load(file)
        X_val = []
        X_test = []
        documents_val = [ast.literal_eval(x) for x in text_val.values.tolist()]
        documents_val = [list(itertools.chain.from_iterable(x)) for x in documents_val]
        documents_test = [ast.literal_eval(x) for x in text_test.values.tolist()]
        documents_test = [list(itertools.chain.from_iterable(x)) for x in documents_test]
        for i in range(len(text_val)):
            X_val.append(Doc2Vec_model.infer(documents_val[i]))
        for i in range(len(text_test)):
            X_test.append(Doc2Vec_model.infer(documents_test[i]))
        file_save_X_val = str(path_save) + "X_val_D_" + str(method) + "_" + str(dimension) + "_" + str(window) + ".npy"
        file_save_X_test = str(path_save) + "X_test_D_" + str(method) + "_" + str(dimension) + "_" + str(window) + ".npy"
        save(file_save_X_val, X_val)
        save(file_save_X_test, X_test)
    print ('done...\n')
    ##Hourly
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_sentiment_H.txt"
    freq = 'H'
    print ('preprocessing Hourly...\n')
    data = d.data_tweets(fname, freq)
    size = data.get_length()
    data_val = data.slice(int(0.7*size), int(0.85*size)) 
    data_test = data.slice(int(0.85*size), size) 
    text_val = data_val.get_text()
    text_test = data_test.get_text()
    methods = ["0","1"]
    dimensions = ["50", "100", "200", "500"]
    windows = ["2", "5", "7"]
    path = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/src/model/"
    path_save = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/src/Doc2Vec_data/"

    for method, dimension, window in itertools.product(methods, dimensions, windows):
        print(method)
        print(dimension)
        print(window)
        file  = str(path) + "Doc2Vec_H_" + str(method) + "_" + str(dimension) + "_" + str(window) + ".model"
        Doc2Vec_model = Doc_2_Vec()
        Doc2Vec_model.load(file)
        X_val = []
        X_test = []
        documents_val = [ast.literal_eval(x) for x in text_val.values.tolist()]
        documents_val = [list(itertools.chain.from_iterable(x)) for x in documents_val]
        documents_test = [ast.literal_eval(x) for x in text_test.values.tolist()]
        documents_test = [list(itertools.chain.from_iterable(x)) for x in documents_test]
        for i in range(len(text_val)):
            X_val.append(Doc2Vec_model.infer(documents_val[i]))
        for i in range(len(text_test)):
            X_test.append(Doc2Vec_model.infer(documents_test[i]))
        file_save_X_val = str(path_save) + "X_val_H_" + str(method) + "_" + str(dimension) + "_" + str(window) + ".npy"
        file_save_X_test = str(path_save) + "X_test_H_" + str(method) + "_" + str(dimension) + "_" + str(window) + ".npy"
        save(file_save_X_val, X_val)
        save(file_save_X_test, X_test)
    print ('done...\n')

if __name__ == '__main__':
    main()