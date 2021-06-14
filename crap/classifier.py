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
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker 
import math
import time
import pdb
from sklearn.feature_selection import SelectKBest, chi2
import ast
from crap.algorithm import algorithm
from utils.parallel import parallel
np.random.seed(42)


class Classifier(algorithm, parallel):
    """
    Text classification object. Fit a classification model based on text in
    order to predict data series prices from Twitter conversations. 
    The classification algorithm can be chosen amongst usual machine learning
    methods. The model is being re-calibrated with most recent observations
    based on a shifting window. The model then compares the predicted 
    performance, based on long, short or neutral positions generated 
    by the value of the prediction with a buy and hold strategy.
    """
    #Defining module name as global variable for access in parent class (parallel)
    mod = sys.modules[__name__]

    def _setup(self, vectorizer, n_features, ngram_range, classifier):
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
    
        thisdict =  {
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

    def _train_test_and_evaluate(self, x_train, y_train, x_test,
                                 y_test, vectorizer, n_features, ngram_range, classifier, params=None):
        """
        Classify and predict the given epoch 
        """
        # majority class strategy 
        null_strategy = self._get_null_strategy(x_test, y_test)
        t0 = time.time()
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
        pipeline = self._setup(vectorizer, min(n_features, count), ngram_range, classifier)
        pdb.set_trace()
        if params != None:
            search = GridSearchCV(pipeline, param_grid = params, cv = 2)
            search.fit(x_train, y_train)
            y_pred = search.predict(x_test)
        else:
            sentiment_fit = pipeline.fit(x_train, y_train)
            y_pred = sentiment_fit.predict(x_test)
        train_test_time = time.time() - t0
        accuracy = accuracy_score(y_test, y_pred)
            
        return accuracy, train_test_time, y_pred, null_strategy

    def _classify(self, x_train, y_train, x_test, y_test, vectorizer, n_features, ngram_range, classifier, params=None):
        accuracy, train_test_time, y_pred, null_strategy = self._train_test_and_evaluate(x_train, y_train, x_test, y_test, 
                                                                                         vectorizer, n_features, ngram_range, classifier, params)
        
        return y_pred, accuracy, train_test_time, null_strategy
        
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
        X, y, bid_ask, size, Return = self._initialize(training_window)
        n_epochs = math.ceil((size - training_window)/testing_window)
        
        if weighting == 'Count':
            vectorizer = CountVectorizer()
        elif weighting == 'Tf-idf':
            vectorizer = TfidfVectorizer()

        test_up = 0
        test_up_length = 0

        for i in range(0, n_epochs):

            # epoch number
            print (str(i) + ' out of ' + str(n_epochs - 1))

            # data preparation
            x_train, y_train, x_test, y_test = self._data_preparation(X, y, training_window, testing_window, size, i)

            # Classification of the prediction
            y_pred, accuracy, train_test_time, null_strategy = self._classify(x_train, y_train, x_test, y_test, 
                                                                              vectorizer, n_features, ngram_range, classifier, params) 
            # calculate the signal of the strategies
            self._calculate_signal(y_test, y_pred, testing_window, null_strategy, i)
            
            # accuracy
            self.accuracy.append(accuracy)              
            test_up += len(y_test[y_test == self.up])
            test_up_length += len(y_test)

        # calculates the cumulative return of the strategies  
        self._calculate_returns(y, training_window, bid_ask, Return)
        accuracy_BH = test_up/test_up_length
        
        return (np.mean(self.accuracy), accuracy_BH, self.null_accuracy, self.eqCurves['Strategy'][len(self.eqCurves['Strategy'])-1], 
                self.eqCurves['Buy and Hold'][len(self.eqCurves['Buy and Hold'])-1],
                self.eqCurves['Majority'][len(self.eqCurves['Majority'])-1], 
                training_window, testing_window)


    
