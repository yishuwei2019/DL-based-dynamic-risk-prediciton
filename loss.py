import math
import torch
import numpy as np
from common import TOL


def coxph_logparlk(event_time, event, hazard_ratio):
    """calculate partial likelihood in Cox model
        time invariant hazard, discrete time units

    :param event_time: numpy[batch_size]
    :param event: numpy[batch_size]
    :param hazard_ratio: tensor(1 * batch_size)
    """
    hazard_ratio = hazard_ratio.data.numpy()
    total = 0.0
    for j in np.unique(event_time):
        index_j = np.array([
            (abs(event_time[ii] - j) < TOL and event[ii] > 0) for ii in range(len(event))
        ])  # H in original code (which subject has event at that time)

        # didn't consider censored sample
        # sum_plus = sum(hazard_ratio[np.array(
        #     [(event_time[ii] - j) > -TOL for ii in range(len(event))])])
        sum_plus = sum(hazard_ratio[np.array([(event_time[ii] - j) > -TOL and event[ii] == 1 for ii in range(len(event))])])
        subtotal_1 = sum(np.log(hazard_ratio[index_j]))

        # subtotal_2 = np.sum(index_j) * np.log(sum_plus)  # if no Efron correction considered
        # the Efron correction
        subtotal_2 = 0
        sum_j = sum(hazard_ratio[index_j])
        for l in range(np.sum(index_j)):
            subtotal_2 += np.log(sum_plus - l * 1.0 / np.sum(index_j) * sum_j)

        total = total + subtotal_1 - subtotal_2
    return torch.tensor(- total, requires_grad=True)


def acc_pairs(event_time, event):
    """calculate accepted pair (i, j)
        i: non-censored event
        j: alive at event_time[i]

    :param event_time: numpy[batch_size]
    :param event: numpy[batch_size]
    """
    event_index = np.nonzero(event)[0]
    acc_pair = []
    for i in event_index:
        # original paper didn't consider censor case
        # acc_pair += [(i, j) for j in np.where(
        #     np.logical_and(event_time >= event_time[i], event == 0))[0]]
        # acc_pair += [(i, j) for j in event_index if event_time[j] > event_time[i]]
        acc_pair += [(i, j) for j in range(len(event)) if event_time[j] > event_time[i]]
        # missing: i and j are both event but tie (same event_time)
    acc_pair.sort(key=lambda x: x[0])
    return acc_pair


def sigmoid_concordance_loss(event_time, event, preds):
    """calculate sigmoid concordance loss function

    :param event_time: numpy[batch_size]
    :param event: numpy[batch_size]
    :param preds: tensor(batch_size * len_time_units)
    """
    acc_pair = acc_pairs(event_time, event)
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


def c_index(event_time, event, hazard_ratio):
    """calculate c-index
    :param event_time: numpy[batch_size]
    :param event: numpy[batch_size]
    :param hazard_ratio: tensor(1 * batch_size)
    """
    hazard_ratio = hazard_ratio.data.numpy()
    acc_pair = acc_pairs(event_time, event)
    return sum([hazard_ratio[x[0]] >= hazard_ratio[x[1]] for x in acc_pair]) * 1.0 / len(acc_pair)
