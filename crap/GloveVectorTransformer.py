from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np


class GloveVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, nlp):
        self.nlp = nlp
        self.dim = 300

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([self.nlp(text).vector for text in X])