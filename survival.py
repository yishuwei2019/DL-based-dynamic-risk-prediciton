import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
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
    train_loss1 = []
    train_loss2 = []
    idx = np.random.permutation(x_train.shape[0])

    count = 0
    while count < len(idx) / batch_size:
        x = x_train[count * batch_size: (count + 1) * batch_size, :]
        lifetime = lifetime_train[count * batch_size: (count + 1) * batch_size]
        censor = censor_train[count * batch_size: (count + 1) * batch_size]
        optimizer.zero_grad()
        score1, score2 = model(x)
        loss1 = log_parlik(lifetime, censor, score1)
        # loss1 = coxph_logparlk(lifetime.numpy(), censor.numpy(), score1)
        loss2 = sum(
            [rank_loss(lifetime, censor, score2, t + 1, time_bin) for t in range(num_time_units)])

        loss = 1.0 * loss1
        loss.backward()
        train_loss1 += [loss1.data]
        train_loss2 += [loss2.data]
        optimizer.step()
        count += 1
        print("train loss 1", train_loss1[-1])
        print("train loss 2", train_loss2[-1])
    return train_loss1, train_loss2


# noinspection PyShadowingNames
def test(batch_size=200):
    model.eval()
    test_loss = []
    count = 0
    while count < x_test.shape[0] / batch_size:
        x = x_test[count * batch_size: (count + 1) * batch_size, :]
        lifetime = lifetime_test[count * batch_size: (count + 1) * batch_size]
        censor = censor_test[count * batch_size: (count + 1) * batch_size]
        score1, score2 = model(x)
        loss1 = log_parlik(lifetime, censor, score1)
        loss2 = sum(
            [rank_loss(lifetime, censor, score2, t + 1, time_bin) for t in range(num_time_units)])
        loss = 1.0 * loss1 + .5 * loss2
        test_loss += [loss.data[0]]
        count += 1

        print(test_loss[-1])
    return test_loss


if __name__ == '__main__':
    d_in, h, d_out = 21, 128, 32
    batch_size = 200
    num_time_units = 24  # 24 months
    time_bin = 30
    n_epochs = 20
    learning_rate = .001
    model = SurvDl(d_in, h, d_out, num_time_units)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 3, gamma=.1)
    train_set, test_set = train_test_split(data, .3)
    train_set = train_set

    x_train = torch.from_numpy(train_set.loc[:, FEATURE_LIST].values.astype('float')).type(
        torch.FloatTensor)
    lifetime_train = torch.from_numpy(train_set.event_time.values.astype('int32')).type(
        torch.FloatTensor)
    censor_train = torch.from_numpy(train_set.event.values.astype('int32')).type(torch.FloatTensor)

    x_test = torch.from_numpy(test_set.loc[:, FEATURE_LIST].values.astype('float')).type(
        torch.FloatTensor)
    lifetime_test = torch.from_numpy(test_set.event_time.values.astype('int32')).type(
        torch.FloatTensor)
    censor_test = torch.from_numpy(test_set.event.values.astype('int32')).type(torch.FloatTensor)

    for epoch in range(n_epochs):
        if epoch < 15:
            scheduler.step()
        print("*************** new epoch ******************")
        for param_group in optimizer.param_groups:
            print("learning rate:", param_group['lr'])
        train_loss = train(batch_size=batch_size)
        test_loss = test(batch_size=batch_size * 2)


