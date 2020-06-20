import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import trange


data, target = [], []

# Number of features used as input. (Number of columns)
INPUT_SIZE = 1

# Number of previous time stamps taken into account.
SEQ_LENGTH = 20

# Number of features in last hidden state ie. number of output time-
# steps to predict.See image below for more clarity.
HIDDEN_SIZE = 10

# Number of stacked rnn layers.
NUM_LAYERS = 3

# We have total of 20 rows in our input. 
# We divide the input into 4 batches where each batch has only 1
# row. Each row corresponds to a sequence of length 5. 
BATCH_SIZE = 1



for i in range(BATCH_SIZE):
    data.append(np.random.rand(20))
    target.append([data[-1][0]]+[a+b for a,b in zip(data[-1], data[-1][1:])])
    

# Initialize the RNN.
rnn = nn.RNN(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, batch_first=True, num_layers = 1)
print(np.array(target).flatten().tolist())
target = torch.Tensor(np.array(target).flatten().tolist())

data = torch.Tensor(np.array(data).flatten().tolist())

# input size : (batch_size , seq_len, input_size)
data = data.view(BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)
target = target.view(BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)

# out shape = (batch, seq_len, num_directions * hidden_size)
# h_n shape  = (num_layers * num_directions, batch, hidden_size)
criterion = nn.functional.mse_loss
out, h_n = rnn(data)
n_epoch = 100
lr=1e-4
optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)

for i in range(20, n_epoch +1):
    optimizer.zero_grad()
    out, h_n = rnn(data)
    print('\nTarget: ', target.shape, '\n', target)
    print('\nOutput: ', out.shape, '\n', out)
    loss = criterion(out, target)
    loss.backward()
    optimizer.step()

print('\nTarget: ', target.shape, '\n', target)
print('Input: ', data.shape, '\n', data)
print('\nOutput: ', out.shape, '\n', out)
print('\nHidden: ', h_n.shape, '\n', h_n)
