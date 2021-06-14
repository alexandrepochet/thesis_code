from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.callbacks import CallbackAny2Vec
from numpy import linalg as LA
import numpy as np
import pdb
np.random.seed(42)


class EpochLogger(CallbackAny2Vec):
        '''Callback to log information about training'''
        def __init__(self):
            self.epoch = 0

        def on_epoch_begin(self, model):
            print("Epoch #{} start".format(self.epoch))

        def on_epoch_end(self, model):
            print("Epoch #{} end".format(self.epoch))
            self.epoch += 1

class Doc_2_Vec():

    def __init__(self, data=None):
        np.random.seed(42)
        self.data = data
        self.model = None

    def get_model(self):
        return self.model

    def get_data(self):
        return self.data

    def _tag_document(self):
        tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(self.data)]
        return tagged_data

    def train(self, dm=1, vector_size=200, window=5, negative=5, hs=0, min_count=5,
              sample=10e-5, workers=4, epochs=100, callbacks="print"):
        if self.data is None:
            print("No data to train on")
        else:
            tagged_data = self._tag_document()
            epoch_logger = EpochLogger()
            if callbacks=="print":
                self.model = Doc2Vec(dm=dm, vector_size=vector_size, window=window, negative=negative,
                                     hs=hs, min_count=min_count, sample=sample, workers=workers, epochs=epochs,
                                     seed=42, callbacks=[epoch_logger])
            else:
                self.model = Doc2Vec(dm=dm, vector_size=vector_size, window=window, negative=negative,
                                     hs=hs, min_count=min_count, sample=sample, workers=workers, epochs=epochs,
                                     seed=42)
            # Build the Volabulary
            self.model.build_vocab(tagged_data)
            # Train the Doc2Vec model
            self.model.train(tagged_data, total_examples=self.model.corpus_count, epochs=epochs)

    # Save trained doc2vec model
    def save(self, file):
        self.model.save(file)

    def load(self, file):
    ## Load saved doc2vec model
        self.model = Doc2Vec.load(file)

    def get_vocabulary(self):
    ## Print model vocabulary
        return self.model.wv.vocab

    def infer(self, doc):
        return self.model.infer_vector(doc)

    def get_vectors(self, ID=None):
        if ID is None:
            return self.model.docvecs
        else:
            return self.model.docvecs[ID]

    def norm(self, vec):
        return LA.norm(vec)

    @staticmethod
    def keyedvectors_to_array(X):
        length = len(X)
        X_train = np.zeros((length,len(X[0])))
        for i in range(length):
            X_train[i,:] = X[i]  
        return X_train