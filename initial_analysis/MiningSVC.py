from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from utils.algorithm import algorithm
from sklearn.feature_selection import SelectKBest, chi2
from utils.utils import count_features
from sklearn.preprocessing import scale
import numpy as np
import pandas as pd
import math
import pdb
import spacy


class MiningSVC(algorithm): 
    
    max_features = 100000
    nlp = spacy.load("en_core_web_lg")

    def __init__(self):
        super().__init__()
        self.model = None

    def fit(self, X_train, Y_train, X_val=None, Y_val=None, X_test=None, Y_test=None,
            weighting="Count", n_features=10000, ngram_range=(1, 3), C=1):
        #try:
        if weighting == 'Count':
            vectorizer = CountVectorizer()
        elif weighting == 'Tf-idf':
            vectorizer = TfidfVectorizer()
        count = count_features(X_train)
        self.model = self._pipeline(vectorizer, min(n_features, count), ngram_range, C)
        self.model.fit(X_train, Y_train)
        return None
        #except: 
        #    return "issue"

    def predict(self, X_test):
        y_hat = self.model.predict(X_test)
        return y_hat

    def evaluate(self, X_test, Y_test):
        y_hat = self.predict(X_test)
        if y_hat is None:
            score = 0
        else:
            score = accuracy_score(Y_test, y_hat)
        return score

    def _pipeline(self, vectorizer, n_features, ngram_range, C):
        """
        Sets up the classifier parameters
        
        Args:
            vectorizer: Weighting scheme of the tokens
            n_features: Number of max features
            ngram_range: The n-gram number 
        Returns
            The parametrized classifier    
        """
        classifier = SVC(kernel="linear", C=C, max_iter=1000000, shrinking=1, tol=0.0001)
        vectorizer.set_params(stop_words=None, max_features=self.max_features, ngram_range=ngram_range)
        
        checker_pipeline = Pipeline([
                ('vectorizer', vectorizer),
                ('reduce_dim', SelectKBest(chi2, k=n_features)),
                ('classify', classifier)])

        return checker_pipeline  