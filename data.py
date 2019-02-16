import pandas as pd
import numpy as np
import torch
from models import BiRNN, CsNet
from torch.nn.utils.rnn import pack_sequence, pad_sequence, pack_padded_sequence

from utils import (
    train_test_split,
    id_loaders,
    prepare_seq
)

# 'CVD_DTH', 'CHD_DTH' not included
base_covs = ['BIRTHYR', 'RACE', 'male', 'EDU_G', 'COHORT']
covs = ['age', 'RXCHL', 'RXHYP', 'SMOKER', 'HXDIAB']  # hxdiab is 0-1 indicator
markers = ['BMINOW', 'TOTCHL', 'LDLCHL', 'HDLCHL', 'SBP', 'DBP', 'GLUCOSE']
et_pairs = {'cvd': 'ttocvd', 'TOT_DTH': 'TTODTH', 'FNFSTRK': 'ttostrk', 'INCCHF': 'ttochf',
            'NFMI': 'ttomi'}  # event: time to event pair

data = pd.read_csv('./data/LRPP_updated.csv', delimiter=',').head(10000)
data = data.rename(columns={'ID_d': 'id'})
data['delta'] = data.groupby('id')['age'].diff().fillna(0)  # delta time
data['time'] = data.groupby('id')['delta'].cumsum()  # time since come to study
data.RACE = pd.to_numeric(data.RACE.replace({'White': 0, "Black": 1, 'HISPANIC': 2, 'ASIAN': 3}))
data.EDU_G = pd.to_numeric(
    data.EDU_G.replace({'high school/ged': 0, 'college or high': 1, 'less than high school': 2}))
data.COHORT = pd.to_numeric(data.COHORT.replace(
    {'CARIDA': 0, 'CHS': 1, 'FHS ORIGINAL': 2, 'FHS OFFSPRING': 3, 'ARIC': 4, 'MESA': 5, 'JHS': 6}))

for marker in markers:
    data[marker + '_y'] = data.groupby('id')[marker].shift()  # next marker in the sequence
data = data.fillna(method="ffill")
batch_size = 20

train_set, test_set = train_test_split(data, .3)
train_ids = id_loaders(train_set.id, batch_size)
test_ids = id_loaders(test_set.id, batch_size, shuffle=False)

x = prepare_seq(data[markers + ['id']], train_ids[0])
model = BiRNN(input_size=7, hidden_size=1, num_layers=1, batch_size=20)
out, seq_len = model(x)
cs_input = torch.stack(tuple([out[ii, seq_len[ii] - 1, :] for ii in range(len(seq_len))]))
cs_model = CsNet(input_size=cs_input.size()[1], layer1_size=3, layer2_size=3, output_size=3)
cs_output = cs_model(cs_input)



