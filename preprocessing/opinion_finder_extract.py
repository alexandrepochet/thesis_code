import re
import pdb


output_file = "C:/Users/alexa/OneDrive/Desktop/Thesis/Lexicon/OpinionFinder/"
def extract_vocabulary(file):
    """
    Read OpinionFinder lexicon file and saves the sentiment vocabulary in a text file
        
    Args:
        file: the file sentiment vocabulary
    Returns:
    """
    if file is None:
        print("no file provided")
    else:
        positive = []
        negative = []
        with open(file, "r") as file:
            i = 0
            lines = file.read().split("\n")
            print(len(lines))
            for line in lines:
                currentline = line.split()
                try:
                    length = len(currentline)
                    if re.sub('mpqapolarity=', "", str(currentline[length - 1])) == 'strongneg' \
                        or re.sub('mpqapolarity=', "", str(currentline[length - 1])) == 'weakneg':
                        negative.append(re.sub('word1=', "", str(currentline[2])))
                    elif re.sub('mpqapolarity=', "", str(currentline[length - 1])) == 'strongpos' \
                        or re.sub('mpqapolarity=', "", str(currentline[length - 1])) == 'weakpos':
                        positive.append(re.sub('word1=', "", str(currentline[2])))
                    else:
                        continue
                except:
                    print('issue \n')
                    continue
                i = i + 1
        file.close()
        outfile = open(str(output_file) + 'positive.txt', "w")
        outfile.write("\n".join(positive))
        outfile = open(str(output_file) + 'negative.txt', "w")
        outfile.write("\n".join(negative))