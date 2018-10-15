
from Utility import *
from sonnet import SonnetWriter



def write_to_file(file_path, poem):

    with open(file_path, 'w') as f:
        for i, lst in enumerate(poem):
            line = " ".join(lst)
            line += "\n"
            f.write(line)
            if i == 3 or i == 7 or i == 11:
                f.write("\n")



if __name__ == '__main__':

    word2index, word2syllable = load_dictionary()
    tokenList, syllableList = load_poem_line_based(word2syllable, word2index)
    n_states = 6
    N_iters = 1000
    writer = SonnetWriter(tokenList, word2index, word2syllable, n_states, N_iters)
    poem = writer.writePoem()
    write_to_file("results/poem.txt", poem)




                    

         

