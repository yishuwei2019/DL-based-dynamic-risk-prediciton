import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from copy import deepcopy
from common import *
from loss import auc_jm, dsn_loss
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
    TARGET_TIME = 40  # target time
    UPGRADE_TIME = 55

    data = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'data', 'data.pkl'))
    data = data[(data.ttocvd >= 0)]
    data = data_short_formatting(
        data, ['cvd', 'ttocvd'] + BASE_COVS + INDICATORS, MARKERS, TRUNCATE_TIME
    )
    FEATURE_LIST = data.columns[3:]

    ## discretize time into 5 buckets
    data['label'] = 0
    data.loc[data['ttocvd'] < 20, 'label'] = 0
    data.loc[(data['ttocvd'] >= 20) & (data['ttocvd'] < 25), 'label'] = 1
    data.loc[(data['ttocvd'] >= 25) & (data['ttocvd'] < 30), 'label'] = 2
    data.loc[(data['ttocvd'] >= 30) & (data['ttocvd'] < 40), 'label'] = 3
    data.loc[data['ttocvd'] >= 40, 'label'] = 4
    # correction for bias due to truncation
    data.loc[(data['ttocvd'] >= 15) & (data['ttocvd'] < 20) & (data['cvd'] == 0), 'label'] = 1
    data.loc[(data['ttocvd'] >= 23) & (data['ttocvd'] < 25) & (data['cvd'] == 0), 'label'] = 2
    data.loc[(data['ttocvd'] >= 28) & (data['ttocvd'] < 30) & (data['cvd'] == 0), 'label'] = 3
    data.loc[(data['ttocvd'] >= 36) & (data['cvd'] == 0), 'label'] = 4
    data.loc[(data['ttocvd'] > 50) & (data['cvd'] == 0), 'label'] = 5

    model = DSNet(35, 210, 5)
    param = deepcopy(model.state_dict())

    batch_size = 200
    n_epochs = 20
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
    for epoch in range(n_epochs):
        print("*************** new epoch ******************")
        auc_test = auc_jm(
            torch.from_numpy(test_set['cvd'].values).type(torch.IntTensor),
            torch.from_numpy(test_set['ttocvd'].values).type(torch.IntTensor),
            model(x_test)[:, 0],
            TARGET_TIME
        )
        print("ten year auc:", auc_test)

        scheduler.step()
        for param_group in optimizer.param_groups:
            print("learning rate:", param_group['lr'])
        train_loss = train_loss + train(batch_size=batch_size)
        test_loss = test_loss + test(batch_size=batch_size)

        print("parameter change:", param_change(param, model))
        param = deepcopy(model.state_dict())

    plot_loss(train_loss, test_loss)




