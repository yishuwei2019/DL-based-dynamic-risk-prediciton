import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from copy import deepcopy
from common import *
from models import SurvDl
from preprocess import survival_preprocess
from loss_original import (
    c_index,
    log_parlik,
    rank_loss,
)
from loss import coxph_logparlk
from utils import train_test_split

TRUNCATE_TIME = 10
TARGET_END = 30
data = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'data', 'data.pkl'))
data.event_time = data.ttocvd.round() - 1
data.event = data.cvd
data = survival_preprocess(data, ['event', 'event_time'] + BASE_COVS + INDICATORS, MARKERS,
                           TRUNCATE_TIME)
data = data[(data.event_time >= 0) & (data.event_time < TARGET_END)]
FEATURE_LIST = data.columns[3:]


# noinspection PyShadowingNames
def train(batch_size=100):
    model.train()
    train_loss = []
    idx = np.random.permutation(x_train.shape[0])

    count = 0
    while count < len(idx) / batch_size:
        x = x_train[idx[count * batch_size: (count + 1) * batch_size], :]
        lifetime = lifetime_train[idx[count * batch_size: (count + 1) * batch_size]]
        censor = censor_train[idx[count * batch_size: (count + 1) * batch_size]]
        optimizer.zero_grad()
        score1, score2 = model(x)
        # loss = log_parlik(lifetime, censor, score1)
        loss = coxph_logparlk(lifetime.numpy(), censor.numpy(), score1)
        # loss2 = sum(
        #     [rank_loss(lifetime, censor, score2, t + 1, time_bin) for t in range(num_time_units)])
        loss.backward()
        train_loss += [loss.data]
        optimizer.step()
        count += 1
        # if count % 10 == 0:
        #     print("train loss", sum(train_loss[:-9:-1]))
    return train_loss


# noinspection PyShadowingNames
def test(batch_size=200):
    model.eval()
    test_loss = []
    idx = np.random.permutation(x_test.shape[0])

    count = 0
    while count < x_test.shape[0] / batch_size:
        x = x_test[idx[count * batch_size: (count + 1) * batch_size], :]
        lifetime = lifetime_test[idx[count * batch_size: (count + 1) * batch_size]]
        censor = censor_test[idx[count * batch_size: (count + 1) * batch_size]]
        score1, score2 = model(x)
        # loss = log_parlik(lifetime, censor, score1)
        loss = coxph_logparlk(lifetime.numpy(), censor.numpy(), score1)
        # loss2 = sum(
        #     [rank_loss(lifetime, censor, score2, t + 1, time_bin) for t in range(num_time_units)])
        test_loss += [loss.data.data]
        count += 1
        # if count % 10 == 0:
        #     print("test_loss:", sum(test_loss[-9:-1]))
    return test_loss


if __name__ == '__main__':
    d_in, h, d_out = 21, 64, 16
    batch_size = 50
    num_time_units = 24  # 24 months
    time_bin = 30
    n_epochs = 20
    learning_rate = .1
    model = SurvDl(d_in, h, d_out, num_time_units)
    param = deepcopy(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 5, gamma=.1)
    train_set, test_set = train_test_split(data, .25)
    train_set = train_set

    x_train = torch.from_numpy(train_set.loc[:, FEATURE_LIST].values.astype('float')).type(
        torch.FloatTensor)
    lifetime_train = torch.from_numpy(train_set.event_time.values.astype('int32')).type(
        torch.IntTensor)
    censor_train = torch.from_numpy(train_set.event.values.astype('int32')).type(torch.IntTensor)

    x_test = torch.from_numpy(test_set.loc[:, FEATURE_LIST].values.astype('float')).type(
        torch.FloatTensor)
    lifetime_test = torch.from_numpy(test_set.event_time.values.astype('int32')).type(
        torch.IntTensor)
    censor_test = torch.from_numpy(test_set.event.values.astype('int32')).type(torch.IntTensor)

    for epoch in range(n_epochs):
        print("*************** new epoch ******************")
        score1_total, _ = model(x_test)
        # loss_total = log_parlik(lifetime_test, censor_test, score1_total)
        # print("total loss in test sample:", loss_total)
        cindex = c_index(censor_test, lifetime_test, score1_total)
        print("cindex in test sample:", cindex)

        if epoch < 15:
            scheduler.step()
        for param_group in optimizer.param_groups:
            print("learning rate:", param_group['lr'])
        train_loss = train(batch_size=batch_size)
        test_loss = test(batch_size=batch_size)

        print("train loss:", sum(train_loss) / len(train_loss))
        print("test loss:", sum(test_loss) / len(test_loss))
        cc = 0
        for key, value in param.items():
            cc = torch.norm(torch.add(value, torch.neg(model.state_dict()[key]))) + cc
        param = deepcopy(model.state_dict())
        print("parameter change:", cc)

        # f, axes = plt.subplots(2, 1)
        # axes[0].plot(train_loss, 'b')
        # axes[0].set_xlabel("iteration")
        # axes[0].set_ylabel("train loss")
        # axes[1].plot(test_loss, 'r')
        # axes[1].set_xlabel("iteration")
        # axes[1].set_ylabel("test loss")
        # plt.show()




