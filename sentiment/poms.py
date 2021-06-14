from collections import defaultdict
from collections import Counter
from sentiment.association import association
from utils.parallel import parallel
import pandas as pd
from tqdm import tqdm
import ast
import pdb
import sys
import warnings


class poms(association, parallel):

    #Defining module name as global variable for access in parent class (parallel)
    mod = sys.modules[__name__]
    location = 'C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/src/figures/'

    def __init__(self, dictionary, association=True):
        super().__init__(dictionary, association)
        self.sentiments = ['positive_OF', 'negative', 'tension', 'anger', 'fatigue', \
                           'depression', 'vigour', 'confusion', 'positive']

    def update_dictionary(self, master_dictionary):
        print('dictionary cannot be updated, please use object association')
        pass

    def calculate_sentiments(self, data, parallel=False):
        """
        Calculate sentiments scores based on the chosen dictionary
        """
        self.data = data
        for sentiment in self.sentiments:
            if self.boolean_association is True:
                self.association[sentiment] = pd.DataFrame(index=self.data.get_date(), columns=['value'])
            self.index[sentiment] = pd.DataFrame(index=self.data.get_date(), columns=['value'])
        length = self.data.get_length()

        for i in tqdm(range(0, length), desc="progress"):
            terms = ast.literal_eval(self.data.get_text().iloc[i])
            n_tweets = self.data.get_count().iloc[i]
            if self.boolean_association is True:
                pmi, p_t, semantic_index = self._semantic(terms, n_tweets, self.dictionary)
                self._aggregate_sentiments(semantic_index, i, pmi, p_t)
            else:
                semantic_index = self._semantic_index(terms, self.dictionary)
                self._aggregate_sentiments(semantic_index, i)
            
        for sentiment in self.sentiments:
            if self.boolean_association is True:
                self.association[sentiment] = self.association[sentiment].fillna(0)
            self.index[sentiment] = self.index[sentiment].fillna(0)

        if parallel:
            if self.boolean_association is True:
                return (self.association, self.index)
            else:
                return self.index

    def _aggregator(self, sentiment, sentiments_list, i, semantic_index, term_assoc=None):
        length = 0
        value = 0
        value_2 = 0
        for senti in sentiments_list:
            if senti == 'relaxed' or senti == 'efficient':
                if self.boolean_association is True:
                    value -= sum(term_assoc[senti].values())
                    length += len(term_assoc[senti])
                value_2 -= semantic_index.get(senti, 0)
            else:
                if self.boolean_association is True:
                    value += sum(term_assoc[senti].values())
                    length += len(term_assoc[senti])
                value_2 += semantic_index.get(senti, 0)
        if self.boolean_association is True:
            try:
                value = value / length
            except:
                value = 0
            return value, value_2
        else:
            return value_2

    def _aggregate_sentiments(self, semantic_index, i, pmi=None, p_t=None):
        if self.boolean_association is True:
            term_assoc = defaultdict(lambda: defaultdict(int))
            for sentiment, n in semantic_index.items():
                for term, n in p_t.items():
                    term_assoc[sentiment][term] = sum(pmi[term][key] for key, value \
                                                      in self.dictionary[sentiment].items())
            self.association['positive_OF'].iloc[i], self.index['positive_OF'].iloc[i] = self._aggregator('positive_OF', ['positive'], i, semantic_index, term_assoc)
            self.association['negative'].iloc[i], self.index['negative'].iloc[i] = self._aggregator('negative', ['negative'], i, semantic_index, term_assoc)
            self.association['tension'].iloc[i], self.index['tension'].iloc[i] = self._aggregator('tension', ['tense', 'shaky', 'irritable',
                                                                                                              'panicky', 'relaxed', 'uneasy',
                                                                                                              'restless', 'nervous', 'anxious'], i, semantic_index, term_assoc)
            self.association['anger'].iloc[i], self.index['anger'].iloc[i] = self._aggregator('anger', ['angry', 'peeved', 'grouchy',
                                                                                                        'spiteful', 'annoyed', 'resentful',
                                                                                                        'bitter', 'aggressive', 'aggressive',
                                                                                                        'misguided', 'furious', 'bad-tempered'], i, semantic_index, term_assoc)
            self.association['fatigue'].iloc[i], self.index['fatigue'].iloc[i] = self._aggregator('fatigue', ['worn-out', 'listless', 'fatigued',
                                                                                                              'exhausted', 'sluggish', 'weary',
                                                                                                              'bushed'], i, semantic_index, term_assoc)
            self.association['depression'].iloc[i], self.index['depression'].iloc[i] = self._aggregator('depression', ['unhappy', 'sorry', 'sad', 'blue',
                                                                                                                       'hopeless', 'unworthy', 'discouraged',
                                                                                                                       'lonely', 'miserable', 'gloomy', 'desperate',
                                                                                                                       'helpless', 'worthless', 'terrified',
                                                                                                                       'guilty'], i, semantic_index, term_assoc)
            self.association['vigour'].iloc[i], self.index['vigour'].iloc[i] = self._aggregator('vigour', ['lively', 'energetic', 'cheerful', 'alert',
                                                                                                           'peppy', 'carefree', 'vigorous', 'active'], i, semantic_index, term_assoc)
            self.association['confusion'].iloc[i], self.index['confusion'].iloc[i] = self._aggregator('confusion', ['confused', 'distracted', 'muddled', 'bewildered',
                                                                                                                    'efficient', 'forgetful','indecisive'], i, semantic_index, term_assoc)
            self.association['positive'].iloc[i], self.index['positive'].iloc[i] = self._aggregator('positive', ['trusting', 'good-natured', 'helpful', 'sympathetic',
                                                                                                                 'considerate', 'clearheaded', 'friendly'], i, semantic_index, term_assoc)

        else:
            self.index['positive_OF'].iloc[i] = self._aggregator('positive_OF', ['positive'], i, semantic_index)
            self.index['negative'].iloc[i] = self._aggregator('negative', ['negative'], i, semantic_index)
            self.index['tension'].iloc[i] = self._aggregator('tension', ['tense', 'shaky', 'irritable',
                                                                         'panicky', 'relaxed', 'uneasy',
                                                                         'restless', 'nervous', 'anxious'], i, semantic_index)
            self.index['anger'].iloc[i] = self._aggregator('anger', ['angry', 'peeved', 'grouchy',
                                                                     'spiteful', 'annoyed', 'resentful',
                                                                     'bitter', 'aggressive', 'aggressive',
                                                                     'misguided', 'furious', 'bad-tempered'], i, semantic_index)
            self.index['fatigue'].iloc[i] = self._aggregator('fatigue', ['worn-out', 'listless', 'fatigued',
                                                                         'exhausted', 'sluggish', 'weary',
                                                                         'bushed'], i, semantic_index)
            self.index['depression'].iloc[i] = self._aggregator('depression', ['unhappy', 'sorry', 'sad', 'blue',
                                                                               'hopeless', 'unworthy', 'discouraged',
                                                                               'lonely', 'miserable', 'gloomy', 'desperate',
                                                                               'helpless', 'worthless', 'terrified',
                                                                               'guilty'], i, semantic_index)
            self.index['vigour'].iloc[i] = self._aggregator('vigour', ['lively', 'energetic', 'cheerful', 'alert',
                                                                       'peppy', 'carefree', 'vigorous', 'active'], i, semantic_index)
            self.index['confusion'].iloc[i] = self._aggregator('confusion', ['confused', 'distracted', 'muddled', 'bewildered',
                                                                             'efficient', 'forgetful','indecisive'], i, semantic_index)
            self.index['positive'].iloc[i] = self._aggregator('positive', ['trusting', 'good-natured', 'helpful', 'sympathetic',
                                                                           'considerate', 'clearheaded', 'friendly'], i, semantic_index)