import os
import torch
import pandas as pd
from models import BiRNN, CsNet, DynamicDeepHit
from data import (
    FILE_DIR,
    MARKERS
)
from utils import (
    train_test_split,
    id_loaders,
    prepare_seq
)

data = pd.read_pickle(os.path.join(FILE_DIR, 'data', 'data.pkl'))
batch_size = 20

train_set, test_set = train_test_split(data, .3)
train_ids = id_loaders(train_set.id, batch_size)
test_ids = id_loaders(test_set.id, batch_size)

x = prepare_seq(data[MARKERS + ['id']], train_ids[0])
model = BiRNN(input_size=7, hidden_size=1, num_layers=1, batch_size=20)
out, seq_len = model(x)
cs_input = torch.stack(tuple([out[ii, seq_len[ii] - 1, :] for ii in range(len(seq_len))]))
cs_model = CsNet(input_size=cs_input.size()[1], layer1_size=3, layer2_size=3, output_size=3)
# cs_output = cs_model(cs_input)

d_model = DynamicDeepHit(num_event=3, rnn_param=[7, 1, 1, 20], cs_param=[3, 3],
                         target_len=15)
marker_output, cs_output = d_model(x)

print(data.head(50)[['id', 'SBP', 'SBP_y']])
