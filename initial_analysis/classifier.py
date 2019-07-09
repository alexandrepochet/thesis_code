# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 08:02:13 2019

@author: alexa
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import NearestCentroid
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker 
import math
import sys
from multiprocessing import Pool
import time
import pdb
from sklearn.feature_selection import SelectKBest, chi2
import ast
np.random.seed(42)


class Classifier(object):
    
    """
    
    Text classification object. Fit a classification model based on text in
    order to predict data series prices from Twitter conversations. 
    The classification algorithm can be chosen amongst usual machine learning
    methods. The model is being re-calibrated with most recent observations
    based on a shifting window. The model then compares the predicted 
    performance, based on long, short or neutral positions generated 
    by the value of the prediction with a buy and hold strategy.
    
    Attributes:
        fname_curr: The preprocessed currency data
        fname_tweets: The tweets data
        freq: The frequency of the model (minute, hour or day)
        threshold: Threshold for the estimation of the long, short or neutral
        positions
        
    """
    
    def __init__(self, df):
        
        self.accuracy = []
        self.eqCurves = []
        self.signal = []
        self.null_strategy_signal = []
        self.df = df
                           
    
    def __setup(self, vectorizer, n_features, ngram_range, classifier):
        
        """
        
        Sets up the classifier parameters
        
        Args:
            vectorizer: Weighting scheme of the tokens
            n_features: Number of max features
            ngram_range: The n-gram number 
            classifier: Machine learning classifier
            
        Returns
            The parametrized classifier
            
        """
        clf1 = Perceptron()
        clf2 = SVC()
        clf3 = AdaBoostClassifier()
        clf4 = LogisticRegression()
        clf5 = KNeighborsClassifier()
        eclf = VotingClassifier(estimators=[('lr', clf1),('svc', clf2), ('rf', clf3),
                                            ('rc', clf4), ('pa', clf5)], voting='hard')
    
        thisdict =	{
                "Random Forest": RandomForestClassifier(),
                "K-Neighbors": KNeighborsClassifier(weights = 'distance', p = 2, n_neighbors = 5, leaf_size = 30),
                "Decision Tree": DecisionTreeClassifier(),
                "Logistic Regression": LogisticRegression(),
                "SVC": SVC(),
                "LinearSVC with L1-based feature selection": Pipeline([
                                ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
                                ('classification', LinearSVC(penalty="l2"))]),
                "Multinomial NB": MultinomialNB(),
                "Bernoulli NB": BernoulliNB(),
                "Ridge Classifier": RidgeClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "Perceptron": Perceptron(penalty = 'l1'),
                "Passive-Aggresive": PassiveAggressiveClassifier(),
                "Nearest Centroid": NearestCentroid(),
                "Voting Classifier": eclf
        }
        print (classifier)
        classifier_ = thisdict[classifier]
        print ("\n")
        vectorizer.set_params(stop_words=None, max_features=50000, ngram_range=ngram_range)
        checker_pipeline = Pipeline([
                ('vectorizer', vectorizer),
                ('reduce_dim', SelectKBest(chi2, k  = n_features)),
                ('classify', classifier_)])
        return checker_pipeline   


    def __train_test_and_evaluate(self, x_train, y_train, x_test,
                                    y_test, vectorizer, n_features, ngram_range, classifier, params=None):
        
        """
        
        Classify and predict the given epoch
        
        Args:
            pipeline: The parametrized classifier
            x_train: Training explanatory variables
            y_train: Training dependent variable
            x_test: Test explanatory variables
            y_test: Test dependent variable
            
        Returns
            accuracy: The accuracy of the epoch
            train_test_time: The time for training the current epoch model
            y_pred: The prediction
            
        """

        # majority class strategy 
        null_strategy, null_accuracy = self.__get_null_strategy(x_test, y_test)
        
        t0 = time.time()
        # Remove timestamp
        x_train = np.delete(x_train, 0, 1)
        x_test = np.delete(x_test, 0, 1)
        # resize
        x_train = x_train.ravel()
        x_test = x_test.ravel()
        list_words = []
        x_train = [" ".join(words) if type(words) == list else words for words in x_train]
        for i in range(0, len(x_train)-1):
            words_array = ast.literal_eval(x_train[i])
            for word in words_array:
                list_words.append(word)
        count = len(set(list_words)) 
        x_test = [" ".join(words) if type(words) == list else words for words in x_test]
        pipeline = self.__setup(vectorizer, min(n_features, count), ngram_range, classifier)

        if params != None:
            search = GridSearchCV(pipeline, param_grid = params, cv = 2)
            search.fit(x_train, y_train)
            y_pred = search.predict(x_test)
        else:
            sentiment_fit = pipeline.fit(x_train, y_train)
            y_pred = sentiment_fit.predict(x_test)
        train_test_time = time.time() - t0
        accuracy = accuracy_score(y_test, y_pred)
        conmat = np.array(confusion_matrix(y_test, y_pred))
        confusion = pd.DataFrame(conmat)
        print ("null accuracy: {0:.2f}%".format(null_accuracy*100))
        print ("accuracy score: {0:.2f}%".format(accuracy*100))
        if accuracy > null_accuracy:
            print ("model is {0:.2f}% more accurate than null accuracy".format((accuracy-null_accuracy)*100))
        elif accuracy == null_accuracy:
            print ("model has the same accuracy with the null accuracy")
        else:
            print ("model is {0:.2f}% less accurate than null accuracy".format((null_accuracy-accuracy)*100))
        print ("-"*80)
        print ("Confusion Matrix\n")
        print (confusion)
        print ("-"*80)
        print ("Classification Report\n")
        print (classification_report(y_test, y_pred))
            
        return accuracy, train_test_time, y_pred, null_strategy, null_accuracy
    
    
    def __get_null_strategy(self, x_test, y_test):

        """
        
        Determines the null strategy and calculate accuracy
        
        Args:
            x_test: The test data (features)
            y_test: The direction of the test data
            
        Returns:
            null_strategy: The majority class strategy
            null_accuracy: The majority class accuracy

        """

        #majority class strategy 
        prop_up = len(x_test[y_test == 'up']) / (len(x_test)*1.)
        prop_down = len(x_test[y_test == 'down']) / (len(x_test)*1.)
        prop_stable = len(x_test[y_test == 'stable']) / (len(x_test)*1.)
        print ('up ' + str(prop_up))
        print ('down ' + str(prop_down))
        print ('stable ' + str(prop_stable))
        null_strategy = ''
        null_accuracy = 0
        if prop_up >= prop_down and prop_up >= prop_stable:
            null_accuracy = prop_up
            null_strategy = 'up'
        elif prop_down >= prop_up and prop_down >= prop_stable:
            null_accuracy = prop_down
            null_strategy = 'down'
        else:
            null_accuracy = prop_stable
            null_strategy = 'stable'
        
        return null_strategy, null_accuracy


    def __classify(self, x_train, y_train, x_test, y_test, vectorizer, n_features, ngram_range, classifier, params=None):
        
        """
        
        Classify and predict the given epoch
        
        Args:
            x_train: Training explanatory variables
            y_train: Training dependent variable
            x_test: Test explanatory variables
            y_test: Test dependent variable
            pipeline: Pipeline
            
        Returns
            y_pred: The predictions
            accuracy: The accuracy of the epoch
            train_test_time: The time for training the current epoch model
            
        """
        

        accuracy, train_test_time, y_pred, null_strategy, null_accuracy = self.__train_test_and_evaluate(x_train, y_train, x_test, y_test, 
                                                                                                           vectorizer, n_features, ngram_range, classifier, params)
        
        return y_pred, accuracy, train_test_time, null_strategy, null_accuracy
        
    
    def run_classification(self, training_window, testing_window, weighting, 
                           n_features, ngram_range, classifier , params=None):

        """
        
        Fit the model for all epochs and estimate the returns of the prediction
        and the return of the buy and hold strategy
        
        Args:
            training_window: The number of training observations
            testing_window: The number of prediction before the model
                            is updated with new observation
            weighting: Weighting scheme of the tokens
            n_features: Number of max features
            ngram_range: The n-gram number 
            classifier: Machine learning classifier
            
        Returns
            
        """

        # initialization
        X, y, bid_ask, size = self.__initialize(training_window)
        n_epochs = math.ceil((size - training_window)/testing_window)
        null_accuracy_tot = 0
        
        if weighting == 'Count':
            vectorizer = CountVectorizer()
        elif weighting == 'Tf-idf':
            vectorizer = TfidfVectorizer()

        for i in range(0, n_epochs):

            # epoch number
            print (str(i) + ' out of ' + str(n_epochs - 1))

            # data preparation
            x_train, y_train, x_test, y_test = self.__data_preparation(X, y, training_window, testing_window, size, i)
            
            # Classification of the prediction
            y_pred, accuracy, train_test_time, null_strategy, null_accuracy = self.__classify(x_train, y_train, x_test, y_test, 
                                                                                              vectorizer, n_features, ngram_range, classifier, params)
            # null accuracy
            null_accuracy_tot += null_accuracy / len(self.signal)*(len(y_test)*1.)
            
            # calculate the signal of the strategies
            self.__calculate_signal(y_test, y_pred, testing_window, null_strategy, i)
            
            # accuracy
            self.accuracy.append(accuracy)              

        # calculates the cumulative return of the strategies  
        self.__calculate_returns(size, training_window, bid_ask)
        
        return (np.mean(self.accuracy), null_accuracy_tot,self.eqCurves['Strategy'][len(self.eqCurves['Strategy'])-1], 
                self.eqCurves['Buy and Hold'][len(self.eqCurves['Buy and Hold'])-1],
                self.eqCurves['Majority'][len(self.eqCurves['Majority'])-1], training_window, testing_window)


    def __initialize(self, training_window):

        """
        
        Initialisation
        
        Args:
            training_window: The training_window
            
        Returns:
            X: The data with the features
            y: The direction of the data
            bid_ask: The bid ask spread
            size: The size of the data

        """

        del self.accuracy[:]
        del self.signal[:]
        del self.null_strategy_signal[:]
        y = self.df['Direction']   
        X = self.df.drop(['Direction', 'Return', 'count', 'close', 'close_bid_ask'], axis = 1)
        size = y.size
        self.signal = 0*self.df.Return[-(size - training_window + 1):]
        self.signal.index = self.df.Date[-(size - training_window + 1):]
        self.null_strategy_signal = 0*self.df.Return[-(size - training_window + 1):] 
        self.null_strategy_signal.index = self.df.Date[-(size - training_window + 1):]
        close = self.df['close'][-(size - training_window + 1):]
        close.index = self.df.Date[-(size - training_window + 1):]
        bid_ask = self.df['close_bid_ask'][-(size - training_window + 1):]
        bid_ask.index = self.df.Date[-(size - training_window + 1):]
        bid_ask = bid_ask.rolling(min_periods=1, window=2).sum()/(4*close)

        return X, y, bid_ask, size
        
    
    def __data_preparation(self, X, y, training_window, testing_window, size, i):

        """
        
        Data preparation
        
        Args:
            X: The test data (features)
            y: The direction of the test data
            training_window: The size of the training window
            testing_window: The size of the testing window
            size: The size of the data
            i: The number of the epoch
            
        Returns:
            X_train: The training features
            y_train: The direction of the training data
            X_test: The test features
            y_test: The direction of the test data

        """

        x_train = X[(i*testing_window):min(i*testing_window+
                    training_window, size)].values
        x_test = X[(i*testing_window+training_window):min((i+1)
                    *testing_window+training_window, size)].values
        y_train = y[(i*testing_window):min(i*testing_window+
                    training_window, size)].values
        y_test = y[(i*testing_window+training_window):min((i+1)*
                    testing_window+training_window, size)].values

        # remove class with less than 10 observations
        unique, counts = np.unique(y_train, return_counts=True)
        _class = unique[np.argwhere(counts < 10)]
        index = np.where(y_train==_class)
        y_train = np.delete(y_train, index)
        x_train = np.delete(x_train, index, axis = 0)
        index = np.where(y_test==_class)
        y_test = np.delete(y_test, index)
        x_test = np.delete(x_test, index, axis = 0)

        return x_train, y_train, x_test, y_test


    def __calculate_signal(self, y_test, y_pred, testing_window, null_strategy, i):

        """
        
        Calculate the signal of the strategies
        
        Args:
            y_test: The test data (direction)
            y_pred: The predicted directions
            testing_window: The size of testing window
            null_strategy: The majority class of the epoch
            i: The number of the epoch
            
        Returns:

        """    
        for j in range(0, y_test.size):
            if y_pred[j] == 'up':
                self.signal.iloc[i*testing_window + j + 1] = +1
            elif y_pred[j] == 'down':
                self.signal.iloc[i*testing_window + j + 1] = -1
            else:
                self.signal.iloc[i*testing_window + j + 1] = 0

        if null_strategy == 'up':
            self.null_strategy_signal.iloc[(i*testing_window + 1):(i*testing_window +
                                           y_test.size + 1)] = +1
        elif null_strategy == 'down':
            self.null_strategy_signal.iloc[(i*testing_window + 1):(i*testing_window +
                                           y_test.size + 1)] = -1
        else:
            self.null_strategy_signal.iloc[(i*testing_window + 1):(i*testing_window +
                                           y_test.size + 1)] = +0


    def __calculate_returns(self, size, training_window, bid_ask):

        """
        
        Calculate the return of the different strategies
        
        Args:
            size: The size of the data
            training_window: The size of the training window
            bid_ask: The bid ask spread
            
        Returns:
            
        """

        returns = pd.DataFrame(index = self.signal.index, 
                               columns=['Buy and Hold', 'Strategy', 'Majority'])
        self.df.set_index('Date', drop = False, inplace = True)
        returns['Buy and Hold'] = self.df.Return[-(size - training_window + 1):]
        cost = self.signal.diff()
        cost.iloc[0] = 0
        cost = np.abs(cost)* bid_ask
        
        returns['Strategy'] = self.signal*returns['Buy and Hold'] - cost
        returns['Majority'] = self.null_strategy_signal*returns['Buy and Hold']
        returns['Buy and Hold'].iloc[0] = 0
        returns['Strategy'].iloc[0] = 0
        returns['Majority'].iloc[0] = 0

        self.eqCurves = pd.DataFrame(index =self.signal.index, 
                           columns=['Buy and Hold', 'Strategy', 'Majority'])
        self.eqCurves['Buy and Hold']=returns['Buy and Hold'].cumsum()+1
        self.eqCurves['Strategy'] = returns['Strategy'].cumsum()+1
        self.eqCurves['Majority'] = returns['Majority'].cumsum()+1


    def plot_return(self, title = None, dash = False):
        
        """
        
        Plot the return of the predicted strategy and the buy and hold
        strategy
        
        Args:
            
        Returns:
            
        """
        years = mdates.YearLocator()   # every year
        months = mdates.MonthLocator()  # every month
        years_fmt = mdates.DateFormatter('%Y-%m')
        self.eqCurves.index = pd.to_datetime(self.eqCurves.index, format='%Y-%m')
        fig, ax = plt.subplots()
        fig.set_size_inches(5, 12)
        self.eqCurves['Strategy'].plot(figsize=(10,8))
        self.eqCurves['Buy and Hold'].plot()
        if dash == True:
            self.eqCurves['Majority'].plot(dashes = [2,6])
        else:
            self.eqCurves['Majority'].plot()
        plt.ylabel('Index value')
        plt.xlabel('Date')
        plt.legend()
        # format the ticks
        fig.autofmt_xdate()
        plt.savefig('C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/src/figures/' + str(title) + '.jpg', pad_inches=1)
        plt.close()


    def run(self, training_windows, testing_windows, weighting, 
                           n_features, ngram_range, classifier, params = None):
        
        """
        
        Run autocorrelation, partial ann QQ plots for autocorrelation
        analysis
        
        Args:
            y: The time series or the residuals 
            lags: The number of lags to consider in the plots
            figsize: The size of the figures
            style: The style
            
        Returns:
            
        """
        
        t = Pool(processes=4)
        rs = t.map(parallel_call, self.__prepare_call("run_classification", 
                                                    training_windows, testing_windows, weighting, 
                                                    n_features, ngram_range, classifier, params))
        t.close()
        return rs
    

    def __prepare_call(self, name, training_windows, testing_windows, weighting, 
                           n_features, ngram_range, classifier, params):  # creates a 'remote call' package for each argument
        
        """
        
        Prepare relevant arguments for parallel call
        
        Args:
            name: The name of the method to call for parallelization 
            training_window: The number of training observations
            testing_window: The number of prediction before the model is updated 
                            with new observation
            confidence: Confidence interval for the strategy
            plot: Boolean indicating if plotting the autocorrelation of residuals
                  of the arch model
            
        Returns:
            
        """
        for train in training_windows:
            for testing in testing_windows:
                test = int(train*testing)
                yield [self.__class__.__name__, self.__dict__, name, [train, test, weighting, 
                                                                      n_features, ngram_range, classifier, params]]


def parallel_call(params):   
    
    """
        
    A helper for calling 'remote' instances
        
    Args:
        params: list containing the class type, the object, the method to call
                for parallilization and the arguments of the method
            
    Returns:
        method(*args): expand arguments, call our method and return the result
        
    """
        
    cls = getattr(sys.modules[__name__], params[0])  # get our class type
    instance = cls.__new__(cls)  # create a new instance without invoking __init__
    instance.__dict__ = params[1]  # apply the passed state to the new instance
    method = getattr(instance, params[2])  # get the requested method
    args = params[3] if isinstance(params[3], (list, tuple)) else [params[3]]
    return method(*args)  # expand arguments, call our method and return the result