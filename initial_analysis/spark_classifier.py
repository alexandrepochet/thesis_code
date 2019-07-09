# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 18:06:37 2019

@author: alexa
"""


import findspark
findspark.init()
import pyspark as ps
import warnings
from pyspark.sql import SQLContext
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from time import time
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import MultilayerPerceptronClassifier
import pandas as pd
import matplotlib.pyplot as plt
from ../preprocessing import preprocess as p
import math
import numpy as np
import pdb


class Spark_Classifier(object):
    
    """
    
    Text classification object. Fit a classification model based on text in
    order to predict data series prices from Twitter conversations. 
    The classification algorithm can be chosen amongst usual machine learning
    methods. The model is being re-calibrated with most recent observations
    based on a shifting window. The model then compares the predicted 
    performance, based on long, short or neutral positions generated 
    by the value of the prediction with a buy and hold strategy. The framework
    is implemented with pyspark in order to improve calculation time
    
    Attributes:
        fname_curr: The preprocessed currency data
        fname_tweets: The tweets data
        freq: The frequency of the model (minute, hour or day)
        threshold: Threshold for the estimation of the long, short or neutral
        positions
        
    """
    
    def __init__(self, fname_curr, fname_tweets, freq, threshold):
        
        try:
            # create SparkContext on all CPUs available: in my case I have 4 CPUs on my laptop
            sc = ps.SparkContext('local[4]')
            sqlContext = SQLContext(sc)
            print("Just created a SparkContext")
        except ValueError:
            warnings.warn("SparkContext already exists in this scope")        
    
        self.threshold = threshold
        self.freq = freq
        self.accuracy = []
        self.eqCurves = []
        self.df = p.preprocess(fname_curr, fname_tweets, freq, threshold, False)
        self.sqlContext = sqlContext
                           
    
    def __setup__(self, vectorizer, n_features, classifier):
        
        """
        
        Sets up the classifier parameters
        
        Args:
            vectorizer: Weighting scheme of the tokens
            n_features: Number of max features
            classifier: Machine learning classifier
            
        Returns
            The parametrized classifier and features process
            
        """
    
        thisdict =	{
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Logistic Regression": LogisticRegression(),
                "Linear SVC": LinearSVC(),
                "Naive Bayes": NaiveBayes(),
                "Multilayer Perceptron": MultilayerPerceptronClassifier()
             }
        print (classifier)
        classifier_ = thisdict[classifier]
        print ("\n")
        
        if vectorizer == 'Hashing':
            weight = HashingTF(numFeatures=n_features, inputCol="text", outputCol='tf')
        elif vectorizer == 'Count':
            weight = CountVectorizer(vocabSize=n_features, inputCol="text", outputCol='tf')
        idf = IDF(inputCol='tf', outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms
        label_stringIdx = StringIndexer(inputCol = "Direction", outputCol = "label", handleInvalid = 'keep')
        pipeline = Pipeline(stages=[weight, idf, label_stringIdx])
            
        return classifier_, pipeline
    
    
    def __train_test_and_evaluate__(self, pipeline, classifier, df_train, df_test):
        
        """
        
        Classify and predict the given epoch
        
        Args:
            pipeline: The parametrized classifier
            classifier: The machine learning classifier
            df_train: The training dataset
            df_test: The test dataset
            
        Returns
            accuracy: The accuracy of the epoch
            train_test_time: The time for training the current epoch model
            y_pred: The prediction
            
        """
    
        prop_up = len(df_test.Direction[df_test.Direction == 'up']) / (len(df_test.Direction))
        prop_down = len(df_test.Direction[df_test.Direction == 'down']) / (len(df_test.Direction))
        prop_stable = len(df_test.Direction[df_test.Direction == 'stable']) / (len(df_test.Direction))
        null_strategy = ''
        if prop_up >= prop_down and prop_up >= prop_stable:
            null_accuracy = prop_up
            null_strategy = 'up'
        elif prop_down >= prop_up and prop_down >= prop_stable:
            null_accuracy = prop_down
            null_strategy = 'down'
        else:
            null_accuracy = prop_stable
            null_strategy = 'stable'
        t0 = time()
        df_train = self.sqlContext.createDataFrame(df_train)
        df_test = self.sqlContext.createDataFrame(df_test)
        pipelineFit = pipeline.fit(df_train)
        train_df = pipelineFit.transform(df_train)
        test_df = pipelineFit.transform(df_test)
        model = classifier.fit(train_df)
        predictions = model.transform(test_df)
        train_test_time = time() - t0
        accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(df_test.count())

        print ("null accuracy: {0:.2f}%".format(null_accuracy*100))
        print ("accuracy score: {0:.2f}%".format(accuracy*100))
        if accuracy > null_accuracy:
            print ("model is {0:.2f}% more accurate than null accuracy".format((accuracy-null_accuracy)*100))
        elif accuracy == null_accuracy:
            print ("model has the same accuracy with the null accuracy")
        else:
            print ("model is {0:.2f}% less accurate than null accuracy".format((null_accuracy-accuracy)*100))
        print ("-"*80)
        predictions = predictions.toPandas()
        direction = predictions.filter(like='Direction').values.T.ravel()
        label = predictions.filter(like='label').values.T.ravel()
        df = pd.DataFrame({'direction':direction, 'label':label}, columns=['direction','label'])
        df = df.drop_duplicates()
        data_dict = df.to_dict()['direction']
        y_pred = np.vectorize(data_dict.get)(predictions.prediction.values)
        
        return accuracy, train_test_time, y_pred, null_strategy
    
    
    def __classify__(self, df_train, df_test, weighting, n_features, classifier):
        
        """
        
        Classify and predict the given epoch
        
        Args:
            weighting: Weighting scheme of the tokens
            n_features: Number of max features
            classifier: Machine learning classifier
            
        Returns
            y_pred: The predictions
            accuracy: The accuracy of the epoch
            train_test_time: The time for training the current epoch model
            
        """
            
        classifier, pipeline = self.__setup__(weighting, n_features, classifier)
        accuracy, train_test_time, y_pred, null_strategy = self.__train_test_and_evaluate__(pipeline,
                                                                             classifier, df_train, df_test)
        
        return y_pred, accuracy, train_test_time, null_strategy
    
       
    def run_classification(self, training_window, testing_window, weighting, 
                           n_features, classifier):
        
        """
        
        Fit the model for all epochs and estimate the returns of the prediction
        and the return of the buy and hold strategy
        
        Args:
            training_window: The number of training observations
            testing_window: The number of prediction before the model
                            is updated with new observation
            weighting: Weighting scheme of the tokens
            n_features: Number of max features 
            classifier: Machine learning classifier
            
        Returns
            
        """
        df = self.df.drop(['count'], axis = 1)
        df.text = df.text.shift(periods=1)   
        df.dropna(inplace = True)
        size = len(df)
        signal = 0*self.df.Return[-(size - training_window + 1):]
        signal.index = self.df.Date[-(size - training_window + 1):]
        null_strategy_signal = 0*self.df.Return[-(size - training_window + 1):] 
        null_strategy_signal.index = self.df.Date[-(size - training_window + 1):]
        n_epochs = math.ceil((size - training_window)/testing_window)
         
        for i in range(0, n_epochs):
            print (str(i) + ' out of ' + str(n_epochs - 1))
            df_train = df.iloc[(i*testing_window):min(i*testing_window+
                        training_window, size)]
            df_test = df.iloc[(i*testing_window+training_window):min((i+1)
                        *testing_window+training_window, size)]
            
            y_pred, accuracy, train_test_time, null_strategy = self.__classify__(df_train, df_test, weighting,
                                                                   n_features, classifier)
            
            for j in range(0, len(df_test)):
                if y_pred[j] == 'up':
                    signal.iloc[i*testing_window + j + 1] = +1
                elif y_pred[j] == 'down':
                    signal.iloc[i*testing_window + j + 1] = -1
                else:
                    signal.iloc[i*testing_window + j + 1] = 0
            
            if null_strategy == 'up':
                null_strategy_signal.iloc[(i*testing_window + 1):(i*testing_window +
                                          len(df_test))] = +1
            elif y_pred[j] == 'down':
                null_strategy_signal.iloc[(i*testing_window + 1):(i*testing_window +
                                          len(df_test))] = -1
            else:
                null_strategy_signal.iloc[(i*testing_window + 1):(i*testing_window +
                                          len(df_test))] = +0
    
            self.accuracy.append(accuracy)                  
            
        returns = pd.DataFrame(index = signal.index, 
                               columns=['Buy and Hold', 'Strategy', 'Majority'])
        self.df = self.df.set_index('Date')
        returns['Buy and Hold'] = self.df.Return[-(size - training_window + 1):]
        returns['Strategy'] = signal*returns['Buy and Hold']
        returns['Majority'] = null_strategy_signal*returns['Buy and Hold']
        returns['Buy and Hold'].iloc[0] = 0
        returns['Strategy'].iloc[0] = 0
        returns['Majority'].iloc[0] = 0
        
        self.eqCurves = pd.DataFrame(index = signal.index, 
                           columns=['Buy and Hold', 'Strategy', 'Majority'])
        self.eqCurves['Buy and Hold']=returns['Buy and Hold'].cumsum()+1
        self.eqCurves['Strategy'] = returns['Strategy'].cumsum()+1
        self.eqCurves['Majority'] = returns['Majority'].cumsum()+1
        

    def plot_return(self):
        
        """
        
        Plot the return of the predicted strategy and the buy and hold
        strategy
        
        Args:
            
        Returns:
            
        """
        
        self.eqCurves['Strategy'].plot(figsize=(10,8))
        self.eqCurves['Buy and Hold'].plot()
        self.eqCurves['Majority'].plot()
        plt.legend()
        plt.show()


def main():
    
     """
     
     Execute matching action for testing
     
     """
     
     fname_curr = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/forexDATA/currency.txt"
     fname_tweets = "C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/tweetsRawData/tweets.txt"
     freq = 'D'
     threshold = 0.0000
     print ('preprocessing...\n')
     model = Spark_Classifier(fname_curr, fname_tweets, freq, threshold)
     training_window = 250
     testing_window = 50
     weighting = 'Count'
     n_features = 30000
     classifier = "Random Forest"
     print ('training the model...\n')
     model.run_classification(training_window, testing_window, weighting, 
                              n_features, classifier)
     print('Done! Plotting returns...\n')
     model.plot_return()
     print(model.accuracy)
     
     
if __name__ == '__main__':
    main()   