import torch
import torch.nn as nn
from torch.autograd import Variable

class Poker_Model(nn.Module):
    def __init__(self, input_size, output_size):
        """
        Initialize the LSTM model.
        """
        super(Poker_Model, self).__init__()

        self.input_size = input_size
        self.hidden_size = 150
        self.output_size = output_size
        self.num_layers = 3
        self.dropout = 0.3
        self.hidden = None
        self.cell = None
        
        # self.recurrent = torch.nn.LSTM(input_size, self.hidden_size, self.num_layers)
        self.recurrent = torch.nn.GRU(input_size, self.hidden_size, self.num_layers)
        # self.recurrent = torch.nn.RNN(input_size, self.hidden_size, self.num_layers)

        self.linear = torch.nn.Linear(self.hidden_size, output_size)
        self.dropout = torch.nn.Dropout(self.dropout)


        
    def init_hidden(self, device):
        """
        Initializes the hidden state for the recurrent neural network.

        TODO:
        Check the model type to determine the initialization method:
        (i) If model_type is LSTM, initialize both cell state and hidden state.
        

        Initialise with zeros.
        """
        self.hidden = torch.zeros((self.num_layers, self.hidden_size)).to(device)
        self.cell = torch.zeros(self.num_layers, self.hidden_size).to(device)
        
    def forward(self, seq):
        """
        Forward pass of the SongRNN model.
        (Hint: In teacher forcing, for each run of the model, input will be a single character
        and output will be pred-probability vector for the next character.)

        Returns:
        - output (Tensor): Output tensor of shape (output_size)
        - activations (Tensor): Hidden layer activations to plot heatmap values


        TODOs:
        (i) Embed the input sequence
        (ii) Forward pass through the recurrent layer
        (iii) Apply dropout (if needed)
        (iv) Pass through the linear output layer
        """
        # x1 = self.embedding(seq)
        # output, (self.hidden, self.cell) = self.recurrent(seq.unsqueeze(0), (self.hidden, self.cell)) #layers are the values of the hidden and cell states
        output, (self.hidden) = self.recurrent(seq.unsqueeze(0), (self.hidden)) #layers are the values of the hidden and cell states
        x3 = self.dropout(output)
        return self.linear(x3.squeeze())

