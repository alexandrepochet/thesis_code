from multiprocessing import Pool
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from nltk.corpus import wordnet
import sys
import warnings
import copy
import pandas as pd
from data.data_tweets import data_tweets
import pdb


class parallel(metaclass=ABCMeta):

    scaling = 10
    warnings.filterwarnings("ignore")

    def _get_class(self):
        return self.__class__.__name__

    def _get_dict(self):
        return self.__dict__

    def run(self, name, nb_processes, *args, type_='initial', parallel=True):
        t = Pool(processes=int(nb_processes))
        rs = t.map(self.parallel_call, self._prepare_call(name, type_, parallel, *args))
        t.close()
        if type_ == 'sentiment':
            if isinstance(rs, list):
                if isinstance(rs[0], tuple):
                    self.association, self.index = self._assemble_sentiment(rs)
                else:
                    self.index = self._assemble_sentiment(rs)
            else:
                if isinstance(rs, tuple):
                    self.association, self.index = self._assemble_sentiment(rs)
                else:
                    self.index = self._assemble_sentiment(rs)
        elif type_ == 'disambiguation':
            self.data, self.data_synset = self._assemble_disambiguation(rs)
            if args[2]:
                file = args[2]
                method = args[1]
                self.data.to_csv(file, self.path, '\t', True)
                self.data_synset.to_csv(str(file) + '_synset_' + str(method), self.path, '\t', True)
        else:
            return rs

    def parallel_call(self, params):  
        """    
        A helper for calling 'remote' instances
                
        Args:
            params: list containing the class type, the object, the method to call
                    for parallilization and the arguments of the method     
        Returns:
            method(*args): expand arguments, call our method and return the result     
        """
        cls = getattr(self.mod, params[0])  # get our class type
        instance = cls.__new__(cls)  # create a new instance without invoking __init__
        instance.__dict__ = params[1]  # apply the passed state to the new instance
        method = getattr(instance, params[2])  # get the requested method
        args = params[3] if isinstance(params[3], (list, tuple)) else [params[3]] 

        return method(*args) 

    def _prepare_call(self, name, type_, parallel, *args): 
        """
        Creates a 'remote call' package for each argument
        """
        if type_ == 'initial':
            training_windows = args[0]
            testing_windows = args[1]
            args = list(args)
            args.pop(0)
            args.pop(0)

            for train in training_windows:
                for testing in testing_windows:
                    test = int(train*testing)
                    yield [self._get_class(), self._get_dict(), name, [train] + [test] + args]
        else:
            #divide data to allow faster run
            data_init = args[0]
            length = data_init.get_length()
            window = int(length/self.scaling) + 1 
            args = list(args)
            args.pop(0)
            for scale in range(0, min(self.scaling, length)):
                data_temp = data_init.slice((scale*window), (scale + 1)*window)
                if not args:
                    yield [self._get_class(), self._get_dict(), name, [data_temp] + [parallel]]
                else:
                    yield [self._get_class(), self._get_dict(), name, [data_temp] + list(args) + [parallel]]
            if type_ == 'sentiment':
                self.data = data_init

    def _assemble_sentiment(self, rs):
        if isinstance(rs, list):
            if isinstance(rs[0], tuple):
                tupel = ()
                nb_elem = len(rs[0])
                for i in range(0, nb_elem):
                    output = dict()
                    for key in rs[0][i]:
                        value = []
                        index = []
                        for elem in rs:
                            index += elem[i][key].index.tolist()
                            value += elem[i][key].value.tolist()
                        output[key] = pd.DataFrame(index=index, columns=['value'])
                        output[key].value = value
                    tupel += (output,)
                return tupel
            else:
                output = dict()    
                for key in rs[0]:
                    index = [] 
                    value = []
                    for elem in rs:
                        index += elem[key].index.tolist()
                        value += elem[key].value.tolist()
                    output[key] = pd.DataFrame(index=index, columns=['value'])
                    output[key].value = value
                return output
        else:
            return rs

    def _assemble_disambiguation(self, rs):
        data = data_tweets()
        data_synsets = data_tweets()
        for elem in rs:
            data.join(elem[0])
            data_synsets.join(elem[1])

        return data, data_synsets
