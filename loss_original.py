import numpy as np
import torch
from torch.autograd import Variable
from functools import reduce

""" This file is copied from deepsurv's original code
We have revised our loss functions to this version, but there are some misleading part,
which is labeled both here and in loss.py
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
        # didn't consider equal time and 1 censored
        all_j = np.array(range(len(lifetime)))[lifetime > lifetime[i]]
        acc_pair.append([(i, j) for j in all_j])

    acc_pair = reduce(lambda x, y: x + y, acc_pair)
    return acc_pair


def c_index(censor, lifetime, score1):
    score1 = score1.data.cpu().numpy()
    acc_pair = acc_pairs(censor, lifetime)
    prob = sum([score1[i] >= score1[j] for (i, j) in acc_pair])[0]*1.0/len(acc_pair)
    return prob


def log_parlik(lifetime, censor, score1):
    t, H = unique_set(lifetime)
    keep_index = np.nonzero(censor.data.cpu().numpy())[0]  # censor = 1
    H = [list(set(h) & set(keep_index)) for h in H]
    n = [len(h) for h in H]
    # t is unique obs time, H: subject who has event at time in t

    score1 = score1.data.cpu().numpy()
    total = 0
    for j in range(len(t)):
        total_1 = np.sum(np.log(score1)[H[j]])
        m = n[j]
        total_2 = 0

        for i in range(m):
            # main part is np.sum(score1[sum(H[j:], [])]), which ignores censoring subjects
            subtotal = np.sum(score1[sum(H[j:], [])]) - (i * 1.0 / m) * (np.sum(score1[H[j]]))
            subtotal = np.log(subtotal)
            total_2 = total_2 + subtotal
        total = total + total_1 - total_2
        total = np.array([total])
    return Variable(torch.from_numpy(total).type(torch.FloatTensor)).view(-1, 1)

