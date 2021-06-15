from sklearn.model_selection._split import TimeSeriesSplit
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
import numpy as np


class ShiftingWindowSplit():

    def __init__(self, fold_size, train_size):
        self.fold_size = fold_size
        self.train_size = train_size

    def reset(self):
        return ShiftingWindowSplit(self.fold_size, self.train_size) 

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        if self.fold_size > n_samples:
            raise ValueError(
                ("Cannot have number of observation per fold ={0} greater"
                 " than the total number of observations: {1}.").format(self.fold_size,
                                                             n_samples))
        indices = np.arange(n_samples)
        folds_number = (n_samples - self.train_size) // self.fold_size
        training_starts = range(0, n_samples-self.train_size, self.fold_size)
        for training_start in training_starts:
            if self.train_size and (self.train_size + training_start) < (n_samples-1):
                yield (indices[training_start:training_start + self.train_size],
                       indices[training_start + self.train_size:training_start + self.train_size + self.fold_size])
            else:
                yield (indices[training_start:training_start + self.train_size],
                       indices[training_start + self.train_size:len(X)])