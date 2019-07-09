# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 20:42:50 2019

@author: alexa
"""

import numpy as np
import pandas as pd
import preprocessing.preprocess as p
import math
import initial_analysis.classifier as c
import time
import matplotlib.pyplot as plt
import pdb


def main():
    
    """
     
    Execute matching action for testing
     
    """
    
    # Daily data
    # Compares the different classifiers 
    
    start = time.time()
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_D.txt"
    freq = 'D'
    print ('preprocessing...\n')
    df = p.get_preprocessed_data(fname, freq)
     
    # Separate in train and validation set and test set
    size = len(df)
    
    df_train = df.iloc[0:int(0.8*size)]
    model = c.Classifier(df_train)
    
    training_windows =[50, 100, 250, 500] 
    testing_windows = [1/8, 1/4, 1/2, 1] 
    weighting = 'Count'
    n_features = 10000
    ngram_range = (1, 3)
    classifiers = ["Random Forest", "K-Neighbors", "Decision Tree", "Logistic Regression", "SVC", "Multinomial NB", "Ridge Classifier", 
                    "AdaBoost", "Perceptron", "Passive-Aggresive", "Nearest Centroid", "Voting Classifier"]
    results = []
    for classifier in classifiers:
        print (classifier)
        print ('---------------\n')
        print ('training the model...\n')
        print ('---------------\n')
        result = model.run(training_windows, testing_windows, weighting, 
                              n_features, ngram_range, classifier)
        result = [classifier, result]
        results.append(result)

    print ('Final results: \n')
    print (results)
    end = time.time()
    print(end - start)
    
    # For the chosen classifier and windows, compares results for number of features
    training_windows =250
    testing_windows = 31
    weighting = 'Count'
    classifier = "K-Neighbors"
    n_features = [x for x in range(5000, 25000, 5000)]
    n_features.insert(0,2500)
    n_features.insert(0,1000)
    n_features.insert(0,500)
    n_features.insert(0,250)
    n_features.insert(0,100)

    accuracy = pd.DataFrame(index = n_features, columns=['unigrams', 'uni-bigrams', 'uni-bi-trigrams'])
    thisdict =  {
                "1": "unigrams",
                "2": "uni-bigrams",
                "3": "uni-bi-trigrams"
                }   
    for i in range(1,4):
        for features in n_features:
            print(features)
            if i == 1:
                ngram_range = (1, 1)
            elif i == 2:
                ngram_range = (1, 2)
            else:
                ngram_range = (1, 3)    
            result = model.run_classification(training_windows, testing_windows, weighting, 
                                               features, ngram_range, classifier) 
            accuracy.loc[features][thisdict[str(i)]] = result[0]
    
    accuracy['unigrams'].plot(color='green', linestyle='dashed')
    accuracy['uni-bigrams'].plot(color='blue', linestyle='dashed')
    accuracy['uni-bi-trigrams'].plot(color='red', linestyle='dashed')
    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel('number of features')
    plt.title('accuracy as a function of the number of features')  
    plt.savefig('./figures/grams_test.jpg', bbox_inches='tight', pad_inches=1)
    
    # Results on the test set
    df_test = df.iloc[(int(0.8*size)-250):]     
    model = c.Classifier(df_test)
    training_windows =250
    testing_windows = 31
    weighting = 'Count'
    n_features = 500
    ngram_range = (1, 2)
    classifier = "K-Neighbors"
    print ('training the model...\n')
    result = model.run_classification(training_windows, testing_windows, weighting, 
                                       n_features, ngram_range, classifier)
     
    model.plot_return(title = 'Daily_kneighbors', dash = False)
    print('results daily neighbors' + str(result))

    
    # Hourly data
    # Compares the different classifiers 
    start = time.time()
    fname = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/tweets_H.txt"
    freq = 'H'
    print ('preprocessing...\n')
    df = p.get_preprocessed_data(fname, freq)
    
    # Separate in train and validation set and test set
    size = len(df)
    
    df_train = df.iloc[0:int(0.8*size)]
    model = c.Classifier(df_train)
    
    training_windows =[50, 200, 500, 1000, 2000, 5000]
    testing_windows = [1/8, 1/4, 1/2, 1] 
    weighting = 'Count'
    n_features = 5000
    ngram_range = (1, 3)
    classifiers = ["Random Forest", "K-Neighbors", "Decision Tree", "Logistic Regression", "SVC", "Multinomial NB", "Ridge Classifier", 
                    "AdaBoost", "Perceptron", "Passive-Aggresive", "Nearest Centroid", "Voting Classifier"]
    results = []
    for classifier in classifiers:
        print (classifier)
        print ('---------------\n')
        print ('training the model...\n')
        print ('---------------\n')
        result = model.run(training_windows, testing_windows, weighting, 
                              n_features, ngram_range, classifier)
        result = [classifier, result]
        results.append(result)

    print ('Final results: \n')
    print (results)
    end = time.time()
    print(end - start)
    
    # For the chosen classifier and windows, compares results for number of features
    training_windows =2000
    testing_windows = 2000
    weighting = 'Count'
    classifier = "K-Neighbors"
    n_features = [x for x in range(5000, 25000, 5000)]
    n_features.insert(0,4000)
    n_features.insert(0,3000)
    n_features.insert(0,2000)
    n_features.insert(0,1000)
    n_features.insert(0,900)
    n_features.insert(0,750)
    n_features.insert(0,600)
    n_features.insert(0,500)
    n_features.insert(0,400)
    n_features.insert(0,300)
    n_features.insert(0,250)
    n_features.insert(0,200)
    n_features.insert(0,100)
    n_features.insert(0,50)
    accuracy = pd.DataFrame(index = n_features, columns=['unigrams', 'uni-bigrams', 'uni-bi-trigrams'])
    thisdict =  {
                "1": "unigrams",
                "2": "uni-bigrams",
                "3": "uni-bi-trigrams"
                }   
    for i in range(1,4):
        for features in n_features:
            print(features)
            if i == 1:
                ngram_range = (1, 1)
            elif i == 2:
                ngram_range = (1, 2)
            else:
                ngram_range = (1, 3)    
            result = model.run_classification(training_windows, testing_windows, weighting, 
                                               features, ngram_range, classifier) 
            accuracy.loc[features][thisdict[str(i)]] = result[0]
    
    print(result)
    accuracy['unigrams'].plot(color='green', linestyle='dashed')
    accuracy['uni-bigrams'].plot(color='blue', linestyle='dashed')
    accuracy['uni-bi-trigrams'].plot(color='red', linestyle='dashed')
    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel('number of features')
    plt.title('accuracy as a function of the number of features')  
    plt.savefig('./figures/grams2.jpg', bbox_inches='tight', pad_inches=1)
    
    # Results on the test set
    
    df_test = df.iloc[(int(0.8*size)-2000):]  
    print (len(df_test))   
    model = c.Classifier(df_test)
    training_windows =2000
    testing_windows = 2000
    weighting = 'Count'
    n_features = 5000
    ngram_range = (1, 3)
    classifier = "K-Neighbors"
    print ('training the model...\n')
    results = model.run_classification(training_windows, testing_windows, weighting, 
                                       n_features, ngram_range, classifier)
     
    model.plot_return(title = 'Hourly_kneighbors', dash = True)
    print('results hourly kneighbors' + str(results))
    
    end = time.time()
    print(end - start)
 
if __name__ == '__main__':
     main()