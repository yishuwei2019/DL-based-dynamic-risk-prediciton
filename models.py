import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence


class LongRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size):
        super(LongRNN, self).__init__()
        self.lstm_layer = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.out_layer = nn.Linear(hidden_size, 1)
        # initial values
        self.h0 = torch.zeros(num_layers, batch_size, hidden_size)
        self.c0 = torch.zeros(num_layers, batch_size, hidden_size)

    def forward(self, x):
        out, _ = self.lstm_layer(x, (self.h0, self.c0))
        # out: tensor (batch_size, seq_length, hidden_state)
        out, _ = pad_packed_sequence(out, batch_first=True, padding_value=0)
        out = self.out_layer(out)
        return out[:, :, 0]


class CoxPH(nn.Module):
    # noinspection PyArgumentList
    def __init__(self, input_size, size_1, output_size, n_time_units):
        super(CoxPH, self).__init__()
        self.Sigmoid = nn.Sigmoid()
        self.fc_layer = nn.Sequential(
            nn.Linear(input_size, size_1),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(size_1, output_size)
        )
        self.fc_layer2 = nn.Linear(1, n_time_units)
        self.beta = nn.Parameter(torch.Tensor(output_size, 1))
        self.beta.data.uniform_(-.001, .001)

    def forward(self, x):
        x = self.fc_layer(x)
        # time invariant hazard ratio
        hazard_ratio = torch.exp(x.mm(self.beta))
        # predicted probability for each time unit
        preds = self.Sigmoid(self.fc_layer2(hazard_ratio))
        return hazard_ratio[:, 0], preds

