from nltk.corpus import sentiwordnet as swn


class Word():
    """
    Word object. Stores information about the sentiment and the synset of
    a word. Convenience class for performing word sense disambiguation
    
    Attributes:
        synset: The synset of the word 
    """
    
    def __init__(self, synset):
        try:
            self.pos_score = swn.senti_synset(synset.name()).pos_score()
            self.neg_score = swn.senti_synset(synset.name()).neg_score()
            self.obj_score = swn.senti_synset(synset.name()).obj_score()
            self.synset = synset
        except:
            self.pos_score = 0
            self.neg_score = 0
            self.obj_score = 1
            self.synset = synset

    def get_pos_score(self):
        return self.pos_score

    def get_neg_score(self):
        return self.neg_score

    def get_obj_score(self):
        return self.obj_score

    def __str__(self):
        """Prints just the Pos/Neg scores for now."""
        s = ""
        s += self.synset.name + "\t"
        s += "PosScore: %s\t" % self.pos_score
        s += "NegScore: %s" % self.neg_score
        return s
