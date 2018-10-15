
from HMM import unsupervised_HMM
import random
import pronouncing

class SonnetWriter:

    def __init__(self, X, word2index, word2syllable, n_states, N_iters):

        self.HMM = unsupervised_HMM(X, n_states, N_iters)
        # can use syllable counts to label the words and do supervised training
        # self.HMM = supervised_HMM(X, Y)
        self.word2index = word2index
        self.word2syllable = word2syllable
        self.index2word = {val:key for key, val in self.word2index.items()}


    def get_syllable(self, index, end = False):

        word = self.index2word[index]
        syllables = self.word2syllable[word]
        index = None
        for i, s in enumerate(syllables):
            if s[0] == "E":
                if end:
                    return int(syllables[i][1:])
                else:
                    return int(random.choices(syllables[:i]+syllables[i+1:])[0])
        return int(random.choices(syllables)[0])


    def writeLine(self, rhyme = None, syllablesCount = 10):

        states = []
        obs = []
        sCount = 0
        while sCount < syllablesCount:
            if not states:
                if not rhyme:
                    states.append(random.choices(range(self.HMM.L), 
                        weights = self.HMM.A_start)[0])
                    obs.append(random.choices(range(self.HMM.D), 
                        weights = self.HMM.O[states[-1]])[0])
                    sCount += self.get_syllable(obs[-1], end = True)
                else:
                    obs.append(rhyme)
                    weights = [self.HMM.O[i][rhyme] for i in range(self.HMM.L)]
                    states.append(random.choices(range(self.HMM.L), 
                        weights = weights)[0])
                    sCount += self.get_syllable(rhyme, end = True)
            else:
                states.append(random.choices(range(self.HMM.L), 
                    weights = self.HMM.A[states[-1]])[0])
                obs.append(random.choices(range(self.HMM.D), 
                    weights = self.HMM.O[states[-1]])[0])
                sCount += self.get_syllable(obs[-1])
        return obs


    def chooseRhyme(self, word):

        rhymes = pronouncing.rhymes(word)
        intersection = set(rhymes) & set(self.word2index.keys())
        if intersection:
            return random.sample(intersection, 1)[0]



    def writePoem(self, syllablesCount = 10):
        # rhyme scheme abab cdcd efef gg
        #  generate the 3 quatrains
        poem = []
        for i in range(4):
            tmp = []
            it = 1 if i == 4 else 2
            for j in range(it):
                first = self.writeLine(syllablesCount = syllablesCount)[::-1]
                tmp.append([self.index2word[i] for i in first])
                end = first[-1]
                rhyme = self.chooseRhyme(self.index2word[end])
                if rhyme:
                    rhyme = self.word2index[rhyme]
                    second = self.writeLine(rhyme = rhyme, 
                        syllablesCount = syllablesCount)[::-1]
                else:
                    second = self.writeLine(syllablesCount = syllablesCount)[::-1]
                tmp.append([self.index2word[i] for i in second])
            if it == 2:
                poem += [tmp[0], tmp[2], tmp[1], tmp[3]]
            else:
                poem += tmp
        print(poem)
        return poem


