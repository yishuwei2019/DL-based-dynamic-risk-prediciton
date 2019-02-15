import pandas as pd
import numpy as np
from utils import train_test_split

data = pd.read_csv('./data/LRPP_updated.csv', delimiter=',')
data = data.rename(columns={'ID_d': 'id'})
train_set, test_set = train_test_split(data, .3)
