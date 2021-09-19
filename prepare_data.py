import warnings
warnings.simplefilter('ignore')

import gc

import numpy as np
import pandas as pd
pd.set_option('max_columns', None)

df_features = pd.read_pickle('/media/user01/wd1tb/yidian/data/feature.pkl')

df_features.drop(['keyword', 'history_docid'], axis=1, inplace=True)
gc.collect()

train = df_features[df_features['click'].notna()]
test  = df_features[df_features['id'].notna()]

train.to_pickle('/media/user01/wd1tb/yidian/data/train47.pkl')
test.to_pickle('/media/user01/wd1tb/yidian/data/test47.pkl')