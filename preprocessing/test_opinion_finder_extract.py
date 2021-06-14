import pdb
import time
import preprocessing.opinion_finder_extract as o


def main():
    """
    Execute matching action for testing
    """
    start = time.time()
    file = "C:/Users/alexa/OneDrive/Desktop/Thesis/Lexicon/OpinionFinder/subjclueslen1polar.txt"
    print('preprocessing...\n')
    o.extract_vocabulary(file)
    end = time.time()
    print(end - start)

if __name__ == '__main__':
    main()
