import pandas as pd
import os

FILE_DIR = os.path.dirname(__file__)

# 'CVD_DTH', 'CHD_DTH' not included
BASE_COVS = [
    'BIRTHYR',
    'RACE',
    'male',
    'EDU_G',
    'COHORT',
]

COVS = [
    'age',
    'RXCHL',
    'RXHYP',
    'SMOKER',
    'HXDIAB',
]  # hxdiab is 0-1 indicator

MARKERS = [
    'BMINOW',
    'TOTCHL',
    'LDLCHL',
    'HDLCHL',
    'SBP',
    'DBP',
    'GLUCOSE',
]

MULTI_EVENTS = [
    'FNFSTRK',
    'INCCHF',
    'NFMI',
]

EVENT_CODE = {
    'CENSOR': 0,
    'TTODTH': 1,
    'ttocvd': 2,
    'ttostrk': 3,
    'ttochf': 4,
    'ttomi': 5,
}

ET_PAIRS = {
    'cvd': 'ttocvd',
    'TOT_DTH': 'TTODTH',
    'FNFSTRK': 'ttostrk',
    'INCCHF': 'ttochf',
    'NFMI': 'ttomi',
}  # event: time to event pair

RACE_CODE = {
    'White': 0,
    "Black": 1,
    'HISPANIC': 2,
    'ASIAN': 3,
}
EDUCATION_CODE = {
    'high school/ged': 0,
    'college or high': 1,
    'less than high school': 2,
}
COHORT_CODE = {
    'CARIDA': 0,
    'CHS': 1,
    'FHS ORIGINAL': 2,
    'FHS OFFSPRING': 3,
    'ARIC': 4,
    'MESA': 5,
    'JHS': 6,
}

MULTI_TIMES = [ET_PAIRS[e] for e in MULTI_EVENTS]

data = pd.read_csv(os.path.join(FILE_DIR, 'data', 'LRPP_updated.csv'), delimiter=',').head(10000)
data = data.rename(columns={'ID_d': 'id'})

# code string variable to numeric
data.RACE = pd.to_numeric(data.RACE.replace(RACE_CODE))
data.EDU_G = pd.to_numeric(data.EDU_G.replace(EDUCATION_CODE))
data.COHORT = pd.to_numeric(data.COHORT.replace(COHORT_CODE))

# create time delta and time since study
data['delta'] = data.groupby('id')['age'].diff().fillna(0)  # delta time
data['time'] = data.groupby('id')['delta'].cumsum()  # time since come to study

# next marker in the sequence as response
for marker in MARKERS:
    data[marker + '_y'] = data.groupby('id')[marker].shift(-1)
data = data.fillna(method="ffill")

# frame into multiple event setting
# event: event code for the first incident / 0 for censoring
# event_time: event time for the first event / time of censoring
data_s = data[['id'] + MULTI_EVENTS + MULTI_TIMES].groupby('id').head(1)
data_s['event_time'] = data_s[MULTI_TIMES].min(axis='columns')
data_s['event'] = data_s[MULTI_EVENTS].any(axis='columns').astype('int64')
data_s.loc[data_s.event == 1, 'event'] = data_s.loc[
    data_s.event == 1, MULTI_TIMES].idxmin(
    axis='columns').map(EVENT_CODE)
data = data.merge(data_s[['id', 'event', 'event_time']], on='id', how='left')
data = data[data.time <= data.event_time]

data.to_pickle(os.path.join(FILE_DIR, 'data', 'data.pkl'))
