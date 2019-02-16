import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.functional import relu, elu


class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            bidirectional=True)

    def forward(self, x):
        # initial state
        h0 = torch.zeros(self.num_layers * 2, self.batch_size,
                         self.hidden_size)  # 2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_size)

        # forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        # out: tensor (batch_size, seq_length, hidden_state * 2)
        # seq_len: seq_len info in the batch
        out, seq_len = pad_packed_sequence(out, batch_first=True, padding_value=-1)

        # (x_M, h_{M-1}) pair for the cause specific input (batch_size * hidden_state)
        # cs_input = torch.stack(tuple([out[ii, seq_len[ii] - 1, :] for ii in range(len(seq_len))]))
        # longitudinal output (sum(seq_len) - batch_size) vector
        # marker_output = torch.cat(tuple([out[ii, :seq_len[ii] - 1, 0] for ii in range(len(seq_len))]))

        return out, seq_len


class CsNet(nn.Module):
    """cause specific network"""
    def __init__(self, input_size, layer1_size, layer2_size, output_size, dropout=.6):
        super(CsNet, self).__init__()
        self.input_size = input_size
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.output_size = output_size
        self.dropout = dropout
        self.layer1 = nn.Linear(self.input_size, self.layer1_size)
        self.layer2 = nn.Linear(self.layer1_size, self.layer2_size)
        self.layer3 = nn.Linear(self.layer2_size, self.output_size)

    def forward(self, x):
        # x: batch_size * input_features
        x = self.layer1(x)
        x = relu(x)
        x = self.layer2(x)
        x = elu(x)
        x = self.layer3(x)
        x = torch.tanh(x)

        return x
