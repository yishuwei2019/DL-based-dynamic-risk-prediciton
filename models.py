import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence


class LongRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size):
        super(LongRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.out_linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # initial state
        h0 = torch.zeros(self.num_layers, self.batch_size,
                         self.hidden_size)
        c0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))
        # out: tensor (batch_size, seq_length, hidden_state)
        out, _ = pad_packed_sequence(out, batch_first=True, padding_value=0)
        return out[:, :, 0]
