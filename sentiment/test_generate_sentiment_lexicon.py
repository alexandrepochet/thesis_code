import pdb
import time
import sentiment.generate_sentiment_lexicon as p


def main():
    """
    Execute matching action for testing
    """
    start = time.time()
    words = ['friendly', 'tense', 'angry', 'worn-out', 'unhappy', 'clearheaded', 'lively', 'confused',
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
    for word in words:
        antonym = antonyms[i]
        if i==0:
            obj = p.generate_sentiment_lexicon(word, antonym)
        else:
            obj.set(word, antonym)
        weights = obj.get_weights()
        i += 1
        print(len(weights))

if __name__ == '__main__':
    main()
