import math
import torch
import numpy as np
from torch.autograd import Variable


def coxph_logparlk(event_time, event, hazard_ratio):
    """calculate partial likelihood in Cox model
        time invariant hazard, discrete time units

    :param event_time: tensor[batch_size]
    :param event: tensor[batch_size]
    :param hazard_ratio: tensor(1 * batch_size)
    """
    event_time, event, hazard_ratio = \
        event_time.data.numpy(), event.data.numpy(), hazard_ratio.data.numpy()
    m = len(event)

    total = 0
    for j in np.unique(event_time):
        index_j = np.array([
            (abs(event_time[ii] - j) < .0001 and event[ii] > 0) for ii in range(m)
        ])
        sum_plus = sum(hazard_ratio[np.array([(x - j) > -.0001 for x in event_time])])

        subtotal_1 = sum(np.log(hazard_ratio[index_j]))
        subtotal_2 = np.sum(index_j) * np.log(sum_plus)
        # # the Efron correction is almost no effect
        # subtotal_2 = 0
        # sum_j = sum(hazard_ratio[index_j])
        # for l in range(np.sum(index_j)):
        #     subtotal_2 += np.log(sum_plus - l * 1.0 / m * sum_j)

        total = total + subtotal_1 - subtotal_2
    total = Variable(
        torch.from_numpy(np.array([total])).type(torch.FloatTensor),
        requires_grad=True
    )
    return torch.neg(total)


def acc_pairs(event_time, event):
    """calculate accepted pair (i, j)
        i: non-censored event
        j: alive at event_time[i]

    :param event_time: tensor[batch_size]
    :param event: tensor[batch_size]
    """
    event_time, event = event_time.data.numpy(), event.data.numpy()
    event_index = np.nonzero(event)[0]
    acc_pair = []
    for i in event_index:
        acc_pair += [(i, j) for j in np.where(
            np.logical_and(event_time >= event_time[i], event == 0))[0]]
        acc_pair += [(i, j) for j in event_index if event_time[j] > event_time[i]]
        # missing: i and j are both event but tie (same event_time)
    return acc_pair


def sigmoid_concordance_loss(event_time, event, preds):
    """calculate sigmoid concordance loss function

    :param event_time: tensor[batch_size]
    :param event: tensor[batch_size]
    :param preds: tensor(batch_size * len_time_units)
    """
    acc_pair = acc_pairs(event_time, event)
    event_time, event, preds = \
        event_time.data.numpy(), event.data.numpy(), preds.data.numpy()
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

    total = sum(list(map(sigmoid_loss, acc_pair))) / (m ** 2 - m)
    total = Variable(
        torch.from_numpy(np.array([total])).type(torch.FloatTensor),
        requires_grad=True
    )
    return torch.neg(total)


def c_index(event_time, event, hazard_ratio):
    """calculate c-index

    :param event_time: tensor[batch_size]
    :param event: tensor[batch_size]
    :param hazard_ratio: tensor(1 * batch_size)
    """
    hazard_ratio = hazard_ratio.data.numpy()
    acc_pair = acc_pairs(event_time, event)
    return sum([hazard_ratio[x[0]] >= hazard_ratio[x[1]] for x in acc_pair]) * 1.0 / len(acc_pair)



