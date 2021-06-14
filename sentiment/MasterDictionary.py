import warnings
from collections import defaultdict
import sentiment.generate_sentiment_lexicon as g


class MasterDictionary():
    """
    MasterDictionary object. Contains the words dictionaries relevant for
    each sentiments. For Bing, two sentiment dictionaries are available, i.e.
    positive and negative. For Loughran-McDonald, eight sentiment dictionaries
    are available

    Attributes:
        dictionary: Dictionary to use for the sentiment scores
    """
    bing_negative = 'C:/Users/alexa/OneDrive/Desktop/Thesis/Lexicon/Bing/negative-words.txt'
    bing_positive = 'C:/Users/alexa/OneDrive/Desktop/Thesis/Lexicon/Bing/positive-words.txt'
    OpinionFinder_negative = 'C:/Users/alexa/OneDrive/Desktop/Thesis/Lexicon/OpinionFinder/negative.txt'
    OpinionFinder_positive = 'C:/Users/alexa/OneDrive/Desktop/Thesis/Lexicon/OpinionFinder/positive.txt'
    path = 'C:/Users/alexa/OneDrive/Desktop/Thesis/Lexicon/Loughran-McDonald/'

    def __init__(self, dictionary):
        self.dictionary = {}#defaultdict(lambda: defaultdict(int))
        if dictionary in ['Bing', 'Loughran-McDonald', 'POMS']:
            self.__upload(dictionary)
        else:
            print('invalid dictionary')
            return

    def get_dictionary(self):
        return self.dictionary

    def __upload(self, dictionary):
        """
        Load sentiment dictionaries
        """
        if dictionary == 'Bing':
            file_negative = self.bing_negative 
            file_positive = self.bing_positive
            file = open(file_negative, 'r')
            negative_list = file.readlines()
            negative_list = [x.replace('\n', '') for x in negative_list]
            self.dictionary['negative'] = dict.fromkeys(negative_list, 1)
            file = open(file_positive, 'r')
            positive_list = file.readlines()
            positive_list = [x.replace('\n', '') for x in positive_list]
            self.dictionary['positive'] = dict.fromkeys(positive_list, 1)
        elif dictionary == 'Loughran-McDonald':
            sentiments = ['positive', 'negative', 'litigious', 'constraining', 'uncertainty',
                      'strong_modal', 'moderate_modal', 'weak_modal']    
            for sentiment in sentiments:
                file = open(str(self.path + sentiment + '.txt'), 'r')
                words_list = file.readlines()
                words_list = [x.replace('\n', '').lower() for x in words_list]
                self.dictionary[sentiment] = dict.fromkeys(words_list, 1)
        else:
            sentiments = ['friendly', 'tense', 'angry', 'worn-out', 'unhappy', 'clearheaded', 'lively', 'confused',
                          'sorry', 'shaky', 'listless', 'peeved', 'considerate', 'sad', 'active', 'irritable',
                          'grouchy', 'blue', 'energetic', 'panicky', 'hopeless', 'relaxed', 'unworthy', 'spiteful',
                          'sympathetic', 'uneasy', 'restless', 'distracted', 'fatigued', 'helpful', 'annoyed', 'discouraged',
                          'resentful', 'nervous', 'lonely', 'miserable', 'muddled', 'cheerful', 'bitter', 'exhausted', 
                          'anxious', 'aggressive', 'good-natured', 'gloomy', 'desperate', 'sluggish', 'rebellious', 'helpless',
                          'weary', 'bewildered', 'alert', 'misguided', 'furious', 'efficient', 'trusting', 'peppy', 
                          'bad-tempered', 'worthless', 'forgetful', 'carefree', 'terrified', 'guilty', 'vigorous', 'indecisive',
                          'bushed']
            antonyms = ['aggressive', 'relaxed', 'pleased', 'fresh', 'happy', 'confused', 'listless', 'clearheaded',
                        'glad', 'stable', 'lively', 'delighted', 'inconsiderate', 'joyful', 'inactive', 'good-natured',
                        'easygoing', 'cheerful', 'fatigued', 'calm', 'optimistic', 'tense', 'worthy', 'sympathetic',
                        'spiteful', 'carefree', 'peaceful', 'focused', 'energetic', 'helpless', 'satisfied', 'hopeful',
                        'kindhearted', 'confident', 'sociable', 'gleeful', 'organized', 'blue', 'pleasant', 'strong',
                        'cool', 'friendly', 'irritable','upbeat', 'content', 'vigorous', 'compliant', 'helpful',
                        'refreshed', 'alert', 'bewildered', 'trusting', 'mild', 'inefficient', 'misguided', 'lethargic',
                        'affable', 'valuable', 'reliable', 'uneasy', 'fearless', 'innocent', 'sluggish', 'decisive',
                        'invigorated']
            i = 0
            file_negative = self.OpinionFinder_negative 
            file_positive = self.OpinionFinder_positive
            file = open(file_negative, 'r')
            negative_list = file.readlines()
            negative_list = [x.replace('\n', '') for x in negative_list]
            self.dictionary['negative'] = dict.fromkeys(negative_list, 1)
            file = open(file_positive, 'r')
            positive_list = file.readlines()
            positive_list = [x.replace('\n', '') for x in positive_list]
            self.dictionary['positive'] = dict.fromkeys(positive_list, 1)
            obj = g.generate_sentiment_lexicon()
            for sentiment in sentiments:
                antonym = antonyms[i]
                self.dictionary[sentiment] = obj.get_weights(sentiment, antonym)
                i += 1
