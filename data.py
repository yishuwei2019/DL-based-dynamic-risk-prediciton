import pandas as pd
import os

FILE_DIR = os.path.dirname(__file__)

# 'CVD_DTH', 'CHD_DTH' not included
BASE_COVS = ['BIRTHYR', 'RACE', 'male', 'EDU_G', 'COHORT']
COVS = ['age', 'RXCHL', 'RXHYP', 'SMOKER', 'HXDIAB']  # hxdiab is 0-1 indicator
MARKERS = ['BMINOW', 'TOTCHL', 'LDLCHL', 'HDLCHL', 'SBP', 'DBP', 'GLUCOSE']
ET_PAIRS = {'cvd': 'ttocvd', 'TOT_DTH': 'TTODTH', 'FNFSTRK': 'ttostrk', 'INCCHF': 'ttochf',
            'NFMI': 'ttomi'}  # event: time to event pair


data = pd.read_csv(os.path.join(FILE_DIR, 'data', 'LRPP_updated.csv'), delimiter=',').head(10000)
data = data.rename(columns={'ID_d': 'id'})
data['delta'] = data.groupby('id')['age'].diff().fillna(0)  # delta time
data['time'] = data.groupby('id')['delta'].cumsum()  # time since come to study
data.RACE = pd.to_numeric(data.RACE.replace({'White': 0, "Black": 1, 'HISPANIC': 2, 'ASIAN': 3}))
data.EDU_G = pd.to_numeric(
    data.EDU_G.replace({'high school/ged': 0, 'college or high': 1, 'less than high school': 2}))
data.COHORT = pd.to_numeric(data.COHORT.replace(
    {'CARIDA': 0, 'CHS': 1, 'FHS ORIGINAL': 2, 'FHS OFFSPRING': 3, 'ARIC': 4, 'MESA': 5, 'JHS': 6}))
for marker in MARKERS:
    data[marker + '_y'] = data.groupby('id')[marker].shift(-1)  # next marker in the sequence
data = data.fillna(method="ffill")

data.to_pickle(os.path.join(FILE_DIR, 'data', 'data.pkl'))
