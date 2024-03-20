import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class Poker_Model(nn.Module):
    def __init__(self, input_size, output_size):
        """
        Initialize the LSTM model.
        """
        super(Poker_Model, self).__init__()

        self.input_size = input_size
        self.hidden_size = 50
        self.output_size = output_size
        self.num_layers = 2
        self.dropout = 0.1
        self.hidden = None

        self.recurrent = torch.nn.RNN(input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.sigmoid = torch.nn.LogSigmoid()

        self.linear = torch.nn.Linear(self.hidden_size, output_size)
        print("linear params", sum(p.numel() for p in self.linear.parameters()))
        self.dropout = torch.nn.Dropout(self.dropout)


        
    def init_hidden(self, batch_size, device):
        self.hidden = torch.zeros((self.num_layers, batch_size, self.hidden_size)).to(device)
        return self.hidden
        
    def forward(self, seq):
        output, (self.hidden) = self.recurrent(seq, (self.hidden)) #layers are the values of the hidden and cell states
        x = self.dropout(output)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

