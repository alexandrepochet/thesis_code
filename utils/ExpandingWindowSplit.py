from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
import numpy as np
import math


class ExpandingWindowSplit():

    def __init__(self, test_size, train_size_init):
        self.test_size = test_size
        self.train_size_init = train_size_init

    def reset(self):
        return ShiftingWindowSplit(self.test_size, self.train_size_init) 

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        if self.train_size_init > n_samples:
            raise ValueError(
                ("Cannot have number of training observations {0} greater"
                 " than the total number of observations: {1}.").format(self.train_size_init,
                                                             n_samples))
        indices = np.arange(n_samples)
        folds_number = math.ceil((n_samples - self.train_size_init) / self.test_size)
        training_ends = range(0, n_samples-self.train_size_init, self.test_size) # splits based on fold_size instead of test_size
        for i in range(0,folds_number):
            train_size = self.train_size_init + i*self.test_size
            if train_size + self.test_size  < (n_samples-1):
                yield (indices[0:train_size],
                       indices[train_size:train_size + self.test_size])
            else:
                yield (indices[0:train_size],
                       indices[train_size:len(X)])