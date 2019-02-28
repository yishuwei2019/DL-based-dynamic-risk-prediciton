import os
import pandas as pd
import torch
import torch.optim as optim
from common import *
from models import CoxPH
from preprocess import survival_preprocess
from loss import (
    coxph_logparlk,
    sigmoid_concordance_loss,
    c_index
)
from utils import (
    train_test_split,
    id_loaders
)

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
def train(model, train_set, batch_size=100):
    train_loss = []
    train_ids = id_loaders(train_set.id, batch_size)
    print("train batch number:", len(train_ids))

    count = 0
    for ids in train_ids:
        data_b = data[data.id.isin(ids)]
        hazard_ratio, preds = model(
            torch.tensor(data_b.loc[:, FEATURE_LIST].values.astype(float), dtype=torch.float32))
        event = data_b.event.values.astype(int)
        event_time = data_b.event_time.values.astype(int)

        l1 = coxph_logparlk(event_time, event, hazard_ratio)
        l2 = sigmoid_concordance_loss(event_time, event, preds)
        loss = torch.add(l1, 10000 * l2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += [loss.tolist()]
        count += 1
        if count % 10 == 0:
            # print("10 batches trained:", sum(train_loss[-9:-1]))
            None
    return train_loss


# noinspection PyShadowingNames
def test(model, test_set, batch_size=200):
    loss_1 = 0
    loss_2 = 0
    cindex = 0
    test_ids = id_loaders(test_set.id, batch_size)

    for ids in test_ids:
        data_b = data[data.id.isin(ids)]
        hazard_ratio, preds = model(
            torch.tensor(data_b.loc[:, FEATURE_LIST].values.astype(float), dtype=torch.float32))
        event = data_b.event.values.astype(int)
        event_time = data_b.event_time.values.astype(int)

        loss_1 = loss_1 + coxph_logparlk(event_time, event, hazard_ratio)
        loss_2 = loss_2 + sigmoid_concordance_loss(event_time, event, preds)
        cindex = cindex + c_index(event_time, event, hazard_ratio)

    loss_1, loss_2, cindex = loss_1 / len(test_ids), loss_2 / len(test_ids), cindex / len(test_ids)
    print("loss_1:", loss_1)
    print("loss_2:", loss_2)
    print("cindex:", cindex)
    return loss_1, loss_2, cindex


if __name__ == '__main__':
    batch_size = 30
    n_epochs = 20
    learning_rate = .01
    model = CoxPH(
        input_size=data.shape[1] - 3,  # depends on dataset
        size_1=128,
        output_size=32,
        n_time_units=TARGET_END,
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 4, gamma=.1)
    train_set, test_set = train_test_split(data, .3)

    for epoch in range(n_epochs):
        if epoch < 15:
            # scheduler.step()
            None
        print("*************** new epoch ******************")
        for param_group in optimizer.param_groups:
            print("learning rate:", param_group['lr'])
            train_loss = train(model, train_set, batch_size=batch_size)
            test_loss = test(model, test_set, batch_size=batch_size * 2)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(os.path.dirname(__file__), 'longitudinal_model.pth'))

