__all__ = [
    'BASE_COVS',
    'COVS',
    'MARKERS',
    'MULTI_TIMES',
    'MULTI_EVENTS',
    'EVENT_CODE',
    'EDUCATION_CODE',
    'RACE_CODE',
    'ET_PAIRS',
    'COHORT_CODE'
]

# 'CVD_DTH', 'CHD_DTH' not included
BASE_COVS = [
    # 'BIRTHYR',
    'RACE',
    'male',
    'EDU_G',
    # 'COHORT',
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
    # 'GLUCOSE',
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
