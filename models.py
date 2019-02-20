import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.functional import relu, elu, softmax


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
        # x = self.layer2(x)
        # x = elu(x)
        x = self.layer3(x)
        x = torch.tanh(x)
        return x


class DynamicDeepHit(nn.Module):
    def __init__(self, num_event, rnn_param, cs_param, target_len, dropout=.6):
        # target_len: discrete future time slots
        super(DynamicDeepHit, self).__init__()
        if len(rnn_param) != 4:
            raise ValueError("rnn parameter number is wrong")
        if len(cs_param) != 2:
            raise ValueError("cs parameter number is wrong")
        self.num_event = num_event
        self.rnn_param = rnn_param
        self.cs_param = cs_param
        self.target_len = target_len
        self.dropout = dropout
        self.rnn_net = BiRNN(
            input_size=rnn_param[0],
            hidden_size=rnn_param[1],
            num_layers=rnn_param[2],
            batch_size=rnn_param[3]
        )
        self.cs_nets = [
            CsNet(
                input_size=self.rnn_param[1] * 2,
                layer1_size=self.cs_param[0],
                layer2_size=self.cs_param[1],
                output_size=self.target_len,
                dropout=self.dropout
            )
            for _ in range(self.num_event)
        ]

    # noinspection PyTypeChecker
    def forward(self, x):
        """
        :param x: packed sequence
        :return:
        """
        out, seq_len = self.rnn_net(x)
        # (x_M, h_{M-1}) pair for the cause specific input (batch_size, hidden_state * 2)
        cs_input = torch.stack(
            [out[ii, seq_len[ii] - 1, :] for ii in range(len(seq_len))]
        )
        # longitudinal output (sum(seq_len) - batch_size) vector
        marker_output = torch.cat(
            [out[ii, :seq_len[ii] - 1, 0] for ii in range(len(seq_len))]
        )

        cs_output = [
            self.cs_nets[ii](cs_input) for ii in range(self.num_event)
        ]
        cs_output = torch.unbind(
            torch.stack(
                cs_output,
                dim=2
            )  # batch_size * target_len * num_event
        )
        cs_output = torch.stack([
            softmax(
                cc.reshape((1, -1)),
                dim=1
            ).reshape(
                (self.target_len, self.num_event)
            )
            for cc in cs_output
        ])  # batch_size * target_len * num_event
        return marker_output, cs_output


# noinspection PyTypeChecker
def deephit_loss(cs_output, label, alpha=0.00001):
    """survival likelihood loss + concordance index loss

    :param cs_output: batch_size * len(target_time) * num_event
    :param label: batch_size * len(target_time) * (num_event + 1)
    :param alpha: weight for loss_2
    :return:
    """
    censor_prob = torch.add(
        torch.ones(*cs_output.size()[:-1]),
        torch.neg(torch.cumsum(torch.sum(cs_output, dim=2), dim=1))
    )
    cs_output = torch.cat(
        (cs_output,
         torch.ones_like(cs_output[:, :, 0:1])),
        dim=2
    )
    p_obs = torch.mul(
        torch.mul(
            torch.unsqueeze(censor_prob, dim=2).expand_as(label),
            cs_output
        ),
        label
    )
    p_obs = torch.sum(p_obs, dim=2)  # vector of batch_size
    loss_1 = torch.neg(torch.log(torch.sum(p_obs, dim=1)).sum())  # survival likelihood

    loss_2 = torch.zeros(1)

    event_times = torch.nonzero(label)[:, 1].tolist()
    event_index = torch.nonzero(
        torch.nonzero(label)[:, 2] - label.shape[2] + 1
    )[:, 0].tolist()

    for ii in event_index:
        for jj in range(len(event_times)):
            tii, tjj = event_times[ii], event_times[jj]
            loss_2 = loss_2 + np.sign(tii - tjj) * torch.exp(p_obs[ii, tii] - p_obs[jj, tii])

    return torch.add(loss_1, alpha * loss_2)









