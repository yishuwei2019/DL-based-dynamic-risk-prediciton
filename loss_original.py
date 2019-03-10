import numpy as np
import torch
from torch.autograd import Variable
from functools import reduce

""" This file is copied from deepsurv's original code
We have revised our loss functions to this version, but there are some misleading part,
which is labeled both here and in loss.py  in docstring
"""


def unique_set(lifetime):
    a = lifetime.data.cpu().numpy()
    t, idx = np.unique(a, return_inverse=True)
    sort_idx = np.argsort(a)
    a_sorted = a[sort_idx]
    unq_first = np.concatenate(([True], a_sorted[1:] != a_sorted[:-1]))
    unq_count = np.diff(np.nonzero(unq_first)[0])
    unq_idx = np.split(sort_idx, np.cumsum(unq_count))
    return t, unq_idx


def acc_pairs(censor, lifetime):
    noncensor_index = np.nonzero(censor.data.cpu().numpy())[0]
    lifetime = lifetime.data.cpu().numpy()
    acc_pair = []
    for i in noncensor_index:
        """didn't consider equal time with second subject censored
        """
        all_j = np.array(range(len(lifetime)))[lifetime > lifetime[i]]
        acc_pair.append([(i, j) for j in all_j])

    acc_pair = reduce(lambda x, y: x + y, acc_pair)
    return acc_pair


def c_index(censor, lifetime, score1):
    score1 = score1.data.cpu().numpy()
    acc_pair = acc_pairs(censor, lifetime)
    prob = sum([score1[i] > score1[j] for (i, j) in acc_pair])[0]*1.0/len(acc_pair)
    return prob


def log_parlik(lifetime, censor, score1):
    t, H = unique_set(lifetime)
    keep_index = np.nonzero(censor.data.cpu().numpy())[0]  # censor = 1
    H = [list(set(h) & set(keep_index)) for h in H]
    n = [len(h) for h in H]
    # t is unique obs time, H: subject who has event at time in t

    total = 0.0
    for j in range(len(t)):
        total_1 = torch.log(score1)[H[j]].sum()
        m = n[j]
        total_2 = 0

        for i in range(m):
            """completely ignores censored subjects (np.sum(score1[sum(H[j:], [])])) in partial likelihood
            """
            subtotal = score1[sum(H[j:], [])].sum() - (i * 1.0 / m) * (score1[H[j]].sum())
            subtotal = torch.log(subtotal)
            total_2 = subtotal + total_2
        total = total_1 - total_2 + total
    return torch.neg(total)


def rank_loss(lifetime, censor, score2, t, time_bin):
    # score2 (n(samples)*24) at time unit t = 1,2,...,24
    acc_pair = acc_pairs(censor, lifetime)
    lifetime = lifetime.data.cpu().numpy()
    total = 0
    for i,j in acc_pair:
        yi = (lifetime[i] >= (t-1) * time_bin) * 1
        yj = (lifetime[j] >= (t-1) * time_bin) * 1
        a = Variable(torch.ones(1)).type(torch.FloatTensor)
        L2dist = torch.dist(score2[j, t-1] - score2[i, t-1], a, 2)
        total = total + L2dist* yi * (1-yj)
    return total

