from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from utils.parallel import parallel
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
import ast
import pdb
import sys
import copy


class disambiguation(parallel):
    """
    Word disambiguation sentiment object. Perform sentiment analysis based 
    on word disambiguation for the tweets passed as arguments. The algorithm
    perform word sense disambiguation, i.e. derived the most probable sense
    of each word based on the presence of the other words in the sentence, and
    then derive a sentiment score for each tweet. Word sense disambiguation is 
    performed with Wordnet and sentiment analysis with SentiwordNet. Results
    are available in the form of a time series of sentiment scores than can
    be plotted together with the evolution of the EUR-USD exchange rate.
    """
    mod = sys.modules[__name__]
    stopwords = set(stopwords.words('english'))
    path = 'C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/preprocessed_data/'
    warnings.filterwarnings("ignore")

    def __init__(self):
        self.data = None
        self.data_synset = None

    def disambiguateWordSenses(self, data, method, file=False, parallel=False):
        """
        Calculate word sense disambiguation 
        """
        self.data = data
        self.data_synset = copy.deepcopy(data)
        length = self.data.get_length()
        for i in tqdm(range(0, length), desc="progress"):
            text = self.data.get_text().iloc[i]
            output = []
            output_synset = []
            wordsList = ast.literal_eval(text)
            if len(wordsList) != 0:
                obj = wordsList[0]
                if isinstance(obj, list):
                    for sentence in wordsList:
                        for word, pos in sentence:
                            if method=='max_similarity':
                                synset = self._max_similarity(sentence, (word, pos))
                            elif method=='simplified_lesk':
                                synset = self._simplified_Lesk(word, pos, sentence)
                            else:
                                print('method not implemented')
                                return
                            if synset is not None:
                                output.append(synset.lemmas()[0].name())
                                output_synset.append(str(synset.name()))
                else:
                    sentence = wordsList
                    for word, pos in sentence:
                        if method=='max_similarity':
                            synset = self._max_similarity(sentence, (word, pos))
                        elif method=='simplified_lesk':
                            synset = self._simplified_Lesk(word, pos, sentence)
                        else:
                            print('method not implemented')
                            return
                        if synset is not None:
                            output.append(synset.lemmas()[0].name())
                            output_synset.append(str(synset.name()))
            self.data.set_text(output, i)
            self.data_synset.set_text(output_synset, i)

        if parallel:
            return self.data, self.data_synset
        if file:
            self.data.to_csv(file, self.path, '\t', True)
            self.data_synset.to_csv(str(file) + '_synset_' + str(method), self.path, '\t', True)
    
    def _max_similarity(self, sentence, word):
        """
        Perform word sense disambiguation
        """
        token = word[0]
        POS = self._wordnet_pos_code(word[1])
        wordsynsets = wn.synsets(token, POS)
        bestScore = 0.0
        result = None
        for synset in wordsynsets:
            for w, p in sentence:
                score = 0.0
                for wsynset in wn.synsets(w, self._wordnet_pos_code(p)):
                    sim = wn.path_similarity(wsynset, synset)
                    if sim is None:
                        continue
                    else:
                        score += sim
                if score > bestScore:
                    bestScore = score
                    result = synset

        return result


    def _wordnet_pos_code(self, tag):
        """
        Perform mapping of POS tagging between nltk and Wordnet
        
        Args:
            tag: The POS tag to map               
        Returns:
            The mapped POS tag
        """
        if tag.startswith('NN'):
            return wn.NOUN
        elif tag.startswith('VB'):
            return wn.VERB
        elif tag.startswith('JJ'):
            return wn.ADJ
        elif tag.startswith('RB'):
            return wn.ADV
        else:
            return ''


    def _simplified_Lesk(self, word, POS, sentence):
        """
        Return the best sense from wordnet for the word in given sentence.

        Args:
            word (string)       The word for which sense is to be found
            POS                 The part-of-speech
            sentence (string)   The sentence containing the word
        """
        POS = self._wordnet_pos_code(POS)
        word_senses = wn.synsets(word, POS)
        if word_senses:
            best_sense = word_senses[0]  # Assume that first sense is most freq.
            max_overlap = 0
            context = [i[0] for i in sentence]
            context = set(context)
            for sense in word_senses:
                signature = self._tokenized_gloss(sense)
                overlap = self._compute_overlap(signature, context)
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_sense = sense
            return best_sense
        else:
            return None

    def _tokenized_gloss(self, sense):
        """
        Return set of token in gloss and examples
        """
        tokens = set(word_tokenize(sense.definition()))
        for example in sense.examples():
            tokens.union(set(word_tokenize(example)))
        return tokens

    def _compute_overlap(self, signature, context):
        """
        Returns the number of words in common between two sets.
        This overlap ignores function words or other words on a stop word list
        """
        gloss = signature.difference(self.stopwords)
        return len(gloss.intersection(context))

