import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import time



class Net(nn.Module):
    def __init__(self, num_char, num_hidden, num_layer):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size = num_char, hidden_size = num_hidden, 
            num_layers = num_layer, batch_first = True) 
        # input and output tensors are provided as (batch, seq, feature)
        # else if batch_first = False, they are provided as (seq, batch, feature)
        self.drop_out = nn.Dropout(p = 0.2)
        self.fc = nn.Linear(num_hidden, num_char)

    def forward(self, x):
        x = self.drop_out(self.lstm(x)[0][:,-1,:])
        x = F.log_softmax(self.fc(x), dim = 1)
        return x



def load_poem_character_based(seq_length = 40, step = 1):

    with open("../data/shakespeare.txt") as f:
        text = ''
        for line in f:
            if len(line.strip()) <= 0 or line.strip().isnumeric():
                continue
            else:
                text += line
    f.close()
    text = text.lower()

    X = []
    y = []
    head = seq_length
    while head < len(text):
        X.append(text[head-seq_length:head])
        y.append(text[head])
        head = head + step

    char_dic = {char:i for i, char in enumerate(sorted(set(text)))}
    int_dic = {i:char for i, char in enumerate(sorted(set(text)))}
    y_to_int = [char_dic[c] for c in y]
    X_to_int = [[char_dic[c] for c in row] for row in X]

    # one-hot encode X_to_int 
    y_cat = torch.tensor(y_to_int)
    X_cat = torch.zeros(len(y_to_int), seq_length, len(char_dic))
    for i in range(len(y_to_int)):
        for j in range(seq_length):
            X_cat[i, j, X_to_int[i][j]] = 1
    print("total number of samples: %d"% len(y))

    return char_dic, int_dic, X_cat, y_cat



def train_RNN(model, X, y, batch_size, n_epoch, drop_last = False, shuffle = True):

    # Define a loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.RMSprop(model.parameters(), lr = 0.001)
    dataset = torch.utils.data.TensorDataset(X, y.long())
    loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, 
        shuffle = shuffle, drop_last = drop_last)

    for epoch in range(n_epoch):
        running_loss = 0
        for i, data in enumerate(loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99: # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %(epoch+1, i+1, running_loss / 100))
                running_loss = 0.0
    return model



def sample(prob_tensor, temperature):
    
    prob_array = prob_tensor.to('cpu').detach().numpy()[0]
    prob_array = np.exp(prob_array)
    prob_array = prob_array ** (1 / temperature)
    prob_array = prob_array / np.sum(prob_array)
    return np.random.choice(prob_array.shape[0], 1, replace = False, p = prob_array)[0]



def generate_sequence(model, seed, char_dic, int_dic, seq_length, device, temperature = 1):

    seq = [' ']* seq_length
    seed_to_int = [char_dic[c] for c in seed]
    seed_cat = deque([[0] * len(char_dic) for _ in range(len(seed_to_int))])
    for i in range(len(seed_to_int)):
        seed_cat[i][seed_to_int[i]] = 1

    for i in range(seq_length):
        seed_cat_tensor = torch.Tensor(seed_cat).to(device)
        prob_tensor = model(seed_cat_tensor.reshape(1,
            seed_cat_tensor.shape[0],seed_cat_tensor.shape[1]))
        prediction = sample(prob_tensor, temperature = temperature)
        seq[i] = int_dic[prediction]
        seed_cat.popleft()
        seed_cat_row = [0]*len(char_dic)
        seed_cat_row[prediction] = 1
        seed_cat.append(seed_cat_row)
    print(seed+''.join(seq))
    return seq
    


if  __name__ == "__main__":

    device = torch.device('cuda:0')
    print('Loading data ...')
    seqLength, sampleStep, batch_size, n_epoch = 40, 1, 128, 25
    char_dic, int_dic, X_cat, y_cat = load_poem_character_based(seqLength, sampleStep)
    print('Finished data loading')

    print('Constructing the NN ...')
    net = Net(len(char_dic), 128, 1)
    net = net.to(device)
    X_cat, y_cat = X_cat.to(device), y_cat.to(device)
    print('Finished NN construction')

    print('Training ...')
    start = time.time()
    net_trained = train_RNN(net, X_cat, y_cat, batch_size, n_epoch, True)
    print('Finished Training')
    print('time taken:{:.3f}'.format(time.time()-start))

    print('Generating sequence ...')
    seed = "shall i compare thee to a summer's day?\n"
    temperature = 0.75
    seq = generate_sequence(net_trained, seed, char_dic, int_dic, 560, device, temperature)
    with open('poems/poem','w') as f:
        f.write("shall i compare thee to a summer's day?\n")
        for char in seq:
            f.write(char)
