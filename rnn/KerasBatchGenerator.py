import numpy as np
import pandas as pd
import pdb
from copy import deepcopy
#np.random.seed(42)
#import random as python_random
#python_random.seed(123)
# Seed value
# Apparently you may use different seed values at each stage
#seed_value= 0

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
#import os
#os.environ['PYTHONHASHSEED']=str(seed_value)


class KerasBatchGenerator(object):

    def __init__(self, X, Y, batch_size, sequence_length):
        #np.random.seed(42)
        #python_random.seed(123)
        self.X = deepcopy(X)
        self.Y = deepcopy(Y)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.counter = 0
        if X.ndim == 3:
            self.dim = X.shape[2]

    def generate(self):
        samples_per_epoch = self.X.shape[0]
        number_of_batches = int(samples_per_epoch/self.batch_size)
        if self.X.ndim == 3:
            x = np.zeros((self.batch_size, self.sequence_length, self.dim))
            y = np.zeros((self.batch_size, 1))
        else:
            x = np.zeros((self.batch_size, self.sequence_length))
            y = np.zeros((self.batch_size, 1))
        index = np.arange(np.shape(self.Y)[0])
        #np.random.shuffle(index)
        while True:
            if self.counter >= number_of_batches:
                #np.random.shuffle(index)
                self.counter = 0
            index_batch = index[self.batch_size*self.counter:self.batch_size*(self.counter+1)]
            x[:] = self.X[index_batch]
            y[:] = self.Y[index_batch]
            self.counter += 1
            yield x,y
