import warnings
warnings.simplefilter('ignore')

import gc

import numpy as np
import pandas as pd
pd.set_option('max_columns', None)

from catboost import CatBoostClassifier

print('loading feature files')
train = pd.read_pickle('/media/user01/wd1tb/yidian/data/train47.pkl')
test  = pd.read_pickle('/media/user01/wd1tb/yidian/data/test47.pkl')
print('train:', train.shape, 'test:', test.shape)

params = {
    'boosting_type': 'Plain',
    'task_type': 'GPU',
    'learning_rate': 0.1,
    'eval_metric': 'AUC',
    'max_ctr_complexity': 1,
    'gpu_cat_features_storage': 'CpuPinnedMemory',
    'loss_function': 'Logloss',
    'iterations': 2000,
    'random_seed': 2021,
    'max_depth': 6,
    'reg_lambda': 0.2,
    'early_stopping_rounds': 50
}
model = CatBoostClassifier(**params)

not_use_cols = ['position', 'duration', 'date']
ycol = 'click'
feature_names = list(
    filter(lambda x: x not in [ycol, 'id'] + not_use_cols, train.columns))
df_train = train[(train['date'] >= '2021-07-01') & (train['date'] <= '2021-07-06')]
df_train['click'] = df_train['click'].astype(int)

print('training begin')
model.fit(df_train[feature_names], df_train[ycol], verbose=10)

print('prediction begin')
sub = test[['id']].copy()
sub['id'] = sub['id'].astype(int)
sub = sub.sort_values(by='id').reset_index(drop=True)
sub['click'] = pred_test[:, 1]

sub.to_csv('submission_catboost.csv', index=False)
print('submission file saved')