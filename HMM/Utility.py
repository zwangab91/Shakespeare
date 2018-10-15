
import string

def load_dictionary():
    '''
    Returns:
        a word2index dictionary and a word2syllableList dictionary
    '''
    word2index = {}
    word2syllable = {}
    with open("../data/Syllable_dictionary.txt") as f:
        word_count = 0
        for line in f:
            lst = line.split()
            word = lst[0]
            syllables = lst[1:]
            word2index[word] = word_count
            word2syllable[word] = syllables
            word_count += 1
    return word2index, word2syllable


#def find_syllable_counts(sCounts, target = 10):





def load_poem_line_based(syllableDic, word2index):
    '''
    Load the file 'shakespeare.txt', with a line as a singular sequence.
    Arguments:
        syllableDic: a dictionary mapping each word to its list of syllable counts

    Returns:
        tokenList: list of tokens, where tokens represent tokens in a line.
        syllableList: corresponding list of syllable counts.
    '''
    tokenList = []
    syllableList = []
    strip = string.punctuation
    strip = strip.replace("'", "")
    strip = strip.replace("-", "")
    with open("../data/shakespeare.txt") as f:
        for line in f:
            tokens = []
            sCounts = []
            line = line.strip()
            if line == "" or line.isdigit():
                continue
            line = line.lower().split()
            for word in line:
                word = word.strip(strip)
                if word not in syllableDic:
                    word = word.strip("'"+strip)
                tokens.append(word2index[word])
                #print(word2index[word])
                sCounts.append(syllableDic[word])
            tokenList.append(tokens[::-1])
            #syllables =  find_syllable_counts(sCounts)
            #syllableList.append(syllables)
    return tokenList, syllableList

                    

         

