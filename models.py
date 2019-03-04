import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.parameter import Parameter


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


class SurvDl(nn.Module):
    """used in coxph.py"""
    def __init__(self, d_in, h, d_out, num_time_units):
        super(SurvDl, self).__init__()
        # self.sigmoid = nn.Sigmoid()
        self.fc_layer = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(0.5),
                                      nn.Linear(h, d_out))
        # self.fc_layer2 = nn.Linear(1, num_time_units)
        self.beta = Parameter(torch.Tensor(d_out, 1))
        self.beta.data.uniform_(-0.001, 0.001)

    def score_1(self, x):
        return torch.exp(x.mm(self.beta))

    # def score_2(self, score1):
    #     return self.sigmoid(self.fc_layer2(score1))

    def forward(self, x):
        new_x = self.fc_layer(x)
        score1 = self.score_1(new_x)
        # score2 = self.score_2(score1)
        # return score1, score2
        return score1, None


class CNet(nn.Module):
    """classification network"""

    def __init__(self, d_in, h_1, h_2, d_out):
        super(CNet, self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(d_in, h_1),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(h_1, h_2),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(h_2, d_out)
        )

    def forward(self, x):
        x = self.fc_layer(x)
        return F.log_softmax(x, dim=1)
