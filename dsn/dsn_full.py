import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from copy import deepcopy
from common import *
from loss import auc_jm, dsn_loss, c_index
from models import DSNet
from preprocess import data_short_formatting
from utils import train_test_split, param_change, plot_loss
"""A scalable discrete-time survival model for neural networks
"""


# noinspection PyShadowingNames
def train(batch_size=100):
    model.train()
    train_loss = []
    idx = np.random.permutation(x_train.shape[0])

    count = 0
    while count < len(idx) / batch_size:
        batch_index = idx[count * batch_size: (count + 1) * batch_size]
        optimizer.zero_grad()
        loss = dsn_loss(
            model(x_train[batch_index, :]),
            label_train[batch_index],
            event_train[batch_index]
        )
        loss.backward()
        train_loss += [loss.data]
        optimizer.step()
        count += 1

    return train_loss[:-1]


# noinspection PyShadowingNames
def test(batch_size=200):
    model.eval()
    test_loss = []
    idx = np.random.permutation(x_test.shape[0])

    count = 0
    while count < x_test.shape[0] / batch_size:
        batch_index = idx[count * batch_size: (count + 1) * batch_size]
        loss = dsn_loss(
            model(x_test[batch_index, :]),
            label_test[batch_index],
            event_test[batch_index]
        )
        test_loss += [loss.data]
        count += 1
    return test_loss[:-1]


if __name__ == "__main__":
    TRUNCATE_TIME = 10  # preparing feature
    data = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'data', 'data.pkl'))
    data = data[(data.ttocvd >= 0)]
    data = data_short_formatting(
        data, ['cvd', 'ttocvd'] + BASE_COVS + INDICATORS, MARKERS, TRUNCATE_TIME
    )
    FEATURE_LIST = data.columns[3:-3]

    data['label'] = data['ttocvd'] - 14
    data.loc[data['ttocvd'] < 15, 'label'] = 0
    data.loc[data['ttocvd'] > 50, 'label'] = 37

    model = DSNet(32, 210, 37)
    param = deepcopy(model.state_dict())

    batch_size = 200
    n_epochs = 30
    learning_rate = .01
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 5, gamma=.1)
    train_set, test_set = train_test_split(data, .3)

    x_train, label_train, event_train = \
        torch.from_numpy(train_set.loc[:, FEATURE_LIST].values).type(torch.FloatTensor), \
        torch.from_numpy(train_set['label'].values).type(torch.LongTensor), \
        torch.from_numpy(train_set['cvd'].values).type(torch.LongTensor)
    x_test, label_test, event_test = \
        torch.from_numpy(test_set.loc[:, FEATURE_LIST].values).type(torch.FloatTensor), \
        torch.from_numpy(test_set['label'].values).type(torch.LongTensor), \
        torch.from_numpy(test_set['cvd'].values).type(torch.LongTensor)

    train_loss = []
    test_loss = []
    test_event = torch.from_numpy(test_set['cvd'].values).type(torch.IntTensor)
    test_time = torch.from_numpy(test_set['ttocvd'].values).type(torch.IntTensor)
    for epoch in range(n_epochs):
        print("*************** new epoch ******************")
        pred = 1 - torch.cumprod(1 - model(x_test), dim=1)
        # print(torch.mean(model(x_test), dim=0))
        print(torch.mean(pred, dim=0)[[6, 11, 16, 26]])

        auc_test = [
            auc_jm(test_event, test_time, pred[:, 6], 20),
            auc_jm(test_event, test_time, pred[:, 11], 25),
            auc_jm(test_event, test_time, pred[:, 16], 30),
            auc_jm(test_event, test_time, pred[:, 26], 40),
        ]
        print("ten year auc:", auc_test)
        cindex = [
            c_index(test_event, test_time, pred[:, 6]),
            c_index(test_event, test_time, pred[:, 11]),
            c_index(test_event, test_time, pred[:, 16]),
            c_index(test_event, test_time, pred[:, 26]),
        ]
        print("c index:", cindex)

        scheduler.step()
        for param_group in optimizer.param_groups:
            print("learning rate:", param_group['lr'])
        train_loss = train_loss + train(batch_size=batch_size)
        test_loss = test_loss + test(batch_size=batch_size)

        print("parameter change:", param_change(param, model))
        param = deepcopy(model.state_dict())

    plot_loss(train_loss, test_loss)




