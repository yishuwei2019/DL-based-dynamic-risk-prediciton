# noinspection SpellCheckingInspection
import torch
import os
import pickle
import torch.optim as optim
from models import (
    DynamicDeepHit,
    deephit_loss,
)
from common import MARKERS, FILE_DIR
from discrete_data import (
    data,
    target_time,
    CODE_POS,
)
from utils import (
    train_test_split,
    id_loaders,
    prepare_label,
    prepare_seq,
)


# noinspection PyShadowingNames
def train(model, train_set, batch_size=200, n_epochs=1, learning_rate=1e-3):
    train_short = train_set[['id', 'event', 'event_time']].groupby('id').head(1).set_index('id')
    train_loss = []

    for _epoch in range(n_epochs):
        print("*************** new epoch ******************")
        train_ids = id_loaders(train_set.id, batch_size)
        print("batch number:", len(train_ids))
        count = 0

        for ids in train_ids:
            ids_b, x = prepare_seq(train_set[MARKERS + ['id']], ids)
            train_short_b = train_short.loc[ids_b, :]
            label_b = prepare_label(train_short_b, CODE_POS, len(target_time))

            optimizer.zero_grad()
            marker_output, cs_output = model(x)
            loss_b = deephit_loss(cs_output, label_b)
            if torch.isnan(loss_b).any():
                with open("./debug/ids.txt", "wb") as fp:
                    pickle.dump(ids, fp)
                train_set.to_pickle(os.path.join(FILE_DIR, 'debug', 'train_set.pkl'))
                exit()

            loss_b.backward()
            optimizer.step()
            train_loss += loss_b.tolist()
            count += 1
            if count % 10 == 0:
                print("10 batch trained")
                print(train_loss[-10:-1])

    return train_loss


if __name__ == '__main__':
    batch_size = 20
    n_epochs = 1
    learning_rate = 1e-4
    model = DynamicDeepHit(
        num_event=3,
        rnn_param=[len(MARKERS), 1, 1, batch_size],
        cs_param=[3, 3],
        target_len=len(target_time)
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_set, test_set = train_test_split(data, .3)
    test_ids = id_loaders(test_set.id, batch_size)
    train_loss = train(model, train_set, batch_size=batch_size)
    print(train_loss)

