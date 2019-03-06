import math
import torch
import numpy as np
from common import TOL

"""Please compare this with loss_original in docstring
"""


def coxph_logparlk(event_time, event, hazard_ratio):
    """calculate partial likelihood in Cox model
        time invariant hazard, discrete time units

    :param event_time: numpy[batch_size]
    :param event: numpy[batch_size]
    :param hazard_ratio: tensor(1 * batch_size)
    """
    total = 0.0
    for j in np.unique(event_time).astype(int):
        # H in original code (which subject has event at that time)
        index_j = torch.min(event_time == j, event == 1).nonzero().data.numpy().flatten()
        """original paper didn't consider censored sample
        sum_plus = hazard_ratio[event_time >= j].sum()
        """
        sum_plus = hazard_ratio[event_time >= j].sum()

        """original paper's version 
        sum_plus = hazard_ratio[torch.min(event_time >= j, event == 1)].sum()
        """
        subtotal_1 = torch.log(hazard_ratio)[index_j].sum() if len(index_j) > 0 else 0

        # subtotal_2 = len(index_j) * torch.log(sum_plus)  # if no Efron correction considered
        # the Efron correction
        subtotal_2 = 0.0
        sum_j = hazard_ratio[index_j].sum() if len(index_j) > 0 else 0
        for l in range(len(index_j)):
            subtotal_2 = torch.add(torch.log(sum_plus - l * 1.0 / len(index_j) * sum_j),
                                   subtotal_2)

        total = subtotal_1 - subtotal_2 + total
    return torch.neg(total)


def acc_pairs2(event_time, event):
    """calculate accepted pair (i, j)
        i: non-censored event
        j: alive at event_time[i]

    :param event_time: numpy[batch_size]
    :param event: numpy[batch_size]
    """
    event_index = np.nonzero(event)[0]
    acc_pair = []
    for i in event_index:
        """original paper didn't consider censor case
        acc_pair += [(i, j) for j in np.where(
            np.logical_and(event_time >= event_time[i], event == 0))[0]]
        acc_pair += [(i, j) for j in event_index if event_time[j] > event_time[i]]
        
        In addition: 
        missing: i and j are both event but tie (same event_time)
        """
        acc_pair += [(i, j) for j in range(len(event)) if event_time[j] > event_time[i]]
    acc_pair.sort(key=lambda x: x[0])
    return acc_pair


def sigmoid_concordance_loss(event_time, event, preds):
    """calculate sigmoid concordance loss function

    :param event_time: numpy[batch_size]
    :param event: numpy[batch_size]
    :param preds: tensor(batch_size * len_time_units)
    """
    acc_pair = acc_pairs2(event_time, event)
    preds = preds.data.numpy()
    m = len(event)  # batch_size

    # noinspection PyShadowingNames
    def sigmoid_loss(x, preds=preds):
        """
        :param x: tuple(i, j)
        :param preds: tensor(batch_size * len_time_units)
        """
        i, j = x[0], x[1]
        fi, fj = preds[i][event_time[i]], preds[j][event_time[j]]
        return 1 - np.log(1 + math.exp(fi - fj)) / np.log(2)

    total = sum(list(map(sigmoid_loss, acc_pair))) * 1.0 / (m ** 2 - m)
    return torch.tensor(-total, requires_grad=True)


def c_index2(event_time, event, hazard_ratio):
    """calculate c-index
    :param event_time: numpy[batch_size]
    :param event: numpy[batch_size]
    :param hazard_ratio: tensor(1 * batch_size)
    """
    hazard_ratio = hazard_ratio.data.numpy()
    acc_pair = acc_pairs2(event_time, event)
    return sum([hazard_ratio[x[0]] >= hazard_ratio[x[1]] for x in acc_pair]) * 1.0 / len(acc_pair)
