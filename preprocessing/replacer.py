from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

class AntonymReplacer():

    def replace(self, word):
        antonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                for antonym in lemma.antonyms():
                    antonyms.add(antonym.name())
                    if len(antonyms) == 1:
                        return antonyms.pop()
        return None

    def negreplace(self, string):
        i = 0
        text = word_tokenize(string)
        len_text = len(text)
        ftext = ""
        while i < len_text:
            word = text[i]
            if word == 'not' and i+1 < len_text:
                ant = self.replace(text[i+1])
                if ant:
                    ftext +=ant + " "
                    i +=2
                    continue
            ftext += word + " "
            i +=1
        return ftext