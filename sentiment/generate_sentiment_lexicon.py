from nltk.corpus import wordnet as wn
import pdb
import copy
        
class generate_sentiment_lexicon():

    def __init__(self):
        pass

    def get_weights(self, word, antonym):
        weights = {}
        dic_word = self._find_all_lemmas(word)
        dic_antonym = self._find_all_lemmas(antonym)        
        if antonym in dic_word: 
            norm = dic_word[antonym]
            for key, value in dic_word.items():
                if (dic_antonym[key] - dic_word[key]) / norm !=0 :
                    weights[key] = (dic_antonym[key] - dic_word[key]) / norm
        else:
            for key, value in dic_word.items():
                synsets_word = wn.synsets(word, wn.ADJ)
                synsets_key = wn.synsets(key, wn.ADJ)
                count = 0
                weight = 0
                for synset_word in synsets_word:
                    for synset_key in synsets_key:
                        val = self._drf_similarity(synset_word, synset_key) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Change to relatedness
                        count += 1 if val is not None else 0
                        weight += val if val is not None else 0
                if weight != 0 and count != 0:
                    weights[key] = weight / count
        weights = {k:v for k,v in weights.items() if v > 0}
        
        return weights

    def _find_all_lemmas(self, word):
        lemmas = dict()
        indicator = False
        unexplored = [word]
        level = 0

        while indicator is False: 
            level += 1
            next_level = dict()
            for w in unexplored:   
                syn = self._find_lemmas(w)
                for elem in syn:
                    next_level[elem] = 1
            new = set(next_level.keys()) - set(lemmas.keys())
            if not new:
                indicator = True
            else:
                for elem in new:
                    lemmas[elem] = level
            unexplored = copy.deepcopy(new)

        return lemmas

    def _drf_similarity(self, synset1, synset2):
        sim = 0
        for lemma1 in synset1.lemmas():
            for form1 in lemma1.derivationally_related_forms():
                    for lemma2 in synset2.lemmas():
                        for form2 in lemma2.derivationally_related_forms():
                            temp = wn.path_similarity(form1.synset(), form2.synset())
                            if temp is not None:
                                if temp > sim:
                                    sim = temp
                                    
        return sim

    def _find_lemmas(self, word):
        synsets = wn.synsets(word, wn.ADJ)
        syns = list()

        for syn in synsets:
            for lemma in syn.lemmas():
                syns.append(lemma.name())

        return syns