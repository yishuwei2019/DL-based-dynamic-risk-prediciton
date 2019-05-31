import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.parameter import Parameter


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        # m.bias.data.fill_(0.01)


class LongRNN(nn.Module):
    def __init__(self, d_in, d_hidden, num_layers, batch_size):
        super(LongRNN, self).__init__()
        self.lstm_layer = nn.LSTM(d_in, d_hidden, num_layers, batch_first=True)
        self.out_layer = nn.Sequential(
            nn.Linear(d_hidden, 1),
            nn.ReLU(),
        )
        # initial values
        self.h0 = torch.zeros(num_layers, batch_size, d_hidden)
        self.c0 = torch.zeros(num_layers, batch_size, d_hidden)

    def forward(self, x):
        out, (h_n, _) = self.lstm_layer(x, (self.h0, self.c0))
        h_n = h_n.permute(1, 0, 2)  # batch, num_layer * num_directions, d_hidden
        # batch, feature size (num_layer * num_directions * d_hidden)
        h_n = h_n.contiguous().view(50, -1)

        # out: tensor (batch_size, seq_length, hidden_state * num_directions)
        out, _ = pad_packed_sequence(out, batch_first=True, padding_value=0)
        out = self.out_layer(out)  # batch_size * seq_length * 1
        return out[:, :, 0], h_n


class CsNet(nn.Module):
    """cause specific network in deep hit"""
    def __init__(self, d_in, h, d_out):
        super(CsNet, self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(d_in, h),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(h, d_out),
        )

    def forward(self, x):
        x = self.fc_layer(x)
        return F.softmax(x, dim=1)


class DeepHit(nn.Module):
    """combination of RNN longitudinal and CsNet for competing events"""
    def __init__(self, rnn_input, csnet_input):
        super(DeepHit, self).__init__()
        self.rnn = LongRNN(
            rnn_input['d_in'],
            rnn_input['d_hidden'],
            rnn_input['num_layers'],
            rnn_input['batch_size'],
        )
        self.csnet1 = CsNet(
            rnn_input['num_layers'] * rnn_input['d_hidden'],  # * 1 (num directions),
            csnet_input['h'],
            csnet_input['d_out'],
        )

    def forward(self, x):
        marker_out, feature_out = self.rnn(x)
        cs1_out = self.csnet1(feature_out)
        return marker_out, cs1_out


class SurvDl(nn.Module):
    """used in coxph.py"""
    def __init__(self, d_in, h, d_out, num_time_units):
        super(SurvDl, self).__init__()
        # self.sigmoid = nn.Sigmoid()
        self.fc_layer = nn.Sequential(
            nn.Linear(d_in, h),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(h, d_out),
        )
        # self.fc_layer.apply(init_weights)
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
    def __init__(self, d_in, h_1, d_out):
        super(CNet, self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(d_in, h_1),
            nn.ReLU(),
            nn.Linear(h_1, d_out),
            nn.LogSoftmax(dim=1),
        )
        self.fc_layer.apply(init_weights)

    def forward(self, x):
        return self.fc_layer(x)


class DSNet(nn.Module):
    """discreate survival network"""
    def __init__(self, d_in, h_1, d_out):
        super(DSNet, self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(d_in, h_1),
            nn.ReLU(),
            nn.Linear(h_1, d_out),
            nn.Sigmoid(),
        )
        self.fc_layer.apply(init_weights)

    def forward(self, x):
        return self.fc_layer(x)