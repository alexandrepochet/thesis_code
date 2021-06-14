import warnings
from copy import deepcopy
from abc import ABCMeta, abstractmethod
from utils.initialize_np import initialize
from sklearn.preprocessing import LabelEncoder


class algorithm(metaclass=ABCMeta):
    """
    Abstract class which supports the machine learning methods implemented 
    for currency prediction
    """
    up = 'up'
    down = 'down'
    stable = 'stable'
    #warnings.filterwarnings("ignore")

    def __init__(self):
        initialize()

    def convert(self, X):
        temp = deepcopy(X)
        temp[temp<0]=-100
        temp[temp>0] = 1
        temp[temp==-100]=0
        return temp

    def prepare_targets(self, y_train, y_test=None):
        le = LabelEncoder()
        le.fit(y_train)
        y_train_enc = le.transform(y_train)
        if y_test is not None:
            y_test_enc = le.transform(y_test)
            return y_train_enc, y_test_enc
        else:
            return y_train_enc