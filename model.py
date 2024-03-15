import torch
import torch.nn as nn
from torch.autograd import Variable

class Poker_Model(nn.Module):
    def __init__(self, input_size, output_size, config):
        """
        Initialize the LSTM model.
        """
        super(Poker_Model, self).__init__()

        HIDDEN_SIZE = config["hidden_size"]
        NUM_LAYERS = config["no_layers"]
        MODEL_TYPE = config["model_type"]
        DROPOUT_P = config["dropout"]

        self.model_type = MODEL_TYPE
        self.input_size = input_size
        self.hidden_size = HIDDEN_SIZE
        self.output_size = output_size
        self.num_layers = NUM_LAYERS
        self.dropout = DROPOUT_P
        self.hidden = None
        self.cell = None
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"

        
        """
        Complete the code

        TODO: 
        (i) Initialize embedding layer with input_size and hidden_size
        (ii) Initialize the recurrent layer based on model type (i.e., LSTM or RNN) using hidden size and num_layers
        (iii) Initialize linear output layer using hidden size and output size
        (iv) Initialize dropout layer with dropout probability
        """
        # self.embedding = torch.nn.Embedding(input_size, HIDDEN_SIZE)

        self.recurrent = None
        if MODEL_TYPE == "lstm":
            self.recurrent = torch.nn.LSTM(input_size, HIDDEN_SIZE, NUM_LAYERS)

        self.linear = torch.nn.Linear(HIDDEN_SIZE, output_size)
        self.dropout = torch.nn.Dropout(DROPOUT_P)


        
    def init_hidden(self, device):
        """
        Initializes the hidden state for the recurrent neural network.

        TODO:
        Check the model type to determine the initialization method:
        (i) If model_type is LSTM, initialize both cell state and hidden state.
        

        Initialise with zeros.
        """
        self.hidden = torch.randn(self.hidden_size).unsqueeze(dim=0).to(device)
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
        output, (self.hidden, self.cell) = self.recurrent(seq.unsqueeze(0), (self.hidden, self.cell)) #layers are the values of the hidden and cell states
        x3 = self.dropout(output)
        return self.linear(x3.squeeze())
