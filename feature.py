#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.simplefilter('ignore')
import pickle
import os
import re
import gc
from pandarallel import pandarallel
import numpy as np
import pandas as pd
pd.set_option('max_columns', None)
pd.set_option('max_rows', 500)
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler

from utils import reduce_mem

pandarallel.initialize()


# In[2]:


feat_path = '/media/user01/wd1tb/yidian/feat'
data_path = '/media/user01/wd1tb/yidian/data'


# In[3]:


load_feat = ['feat_basic', 'feat_basic_history_all', 'feat_global_statis']


# In[4]:


feat_list = []
for feat in tqdm(load_feat):
    df = pd.read_pickle(f'{feat_path}/{feat}.pkl')
    feat_list.append(df)


# In[5]:


feat_list[-1].head()


# In[6]:


feat_list[0].head()


# In[7]:


# 检查每个df的顺序是否一致
for i in range(len(load_feat)):
    if i == 0:
        continue

    for f in ['userid', 'docid', 'date']:
        if f not in feat_list[i]:
            continue
            
        if not (feat_list[0][f] == feat_list[i][f]).all():
            print(f'{load_feat[i]}的{f}顺序不一致')


# In[8]:


# 删除重复列
all_columns = feat_list[0].columns.tolist()

for i in range(len(load_feat)):
    if i == 0:
        continue

    raw_columns = feat_list[i].columns.tolist()
    drop_columns = list(set(all_columns) & set(raw_columns))

    feat_list[i].drop(columns=drop_columns, inplace=True)

    columns = feat_list[i].columns.tolist()
    all_columns += columns


# In[9]:


df_data = pd.concat(feat_list, axis=1)


# In[10]:


del feat_list
gc.collect()


# In[11]:


df_data.head()


# In[12]:


sparse_features = [
    'userid', 'docid', 'network', 'device', 'os', 'province', 'city', 'age',
    'gender', 'category1st', 'category2nd'
]
dense_features = [
    'refresh', 'userid_history_count', 'docid_history_count',
    'category1st_history_count', 'category2nd_history_count',
    'userid_category1st_history_count', 'userid_category2nd_history_count',
    'userid_ctr', 'docid_ctr', 'category1st_ctr', 'category2nd_ctr',
    'userid_category1st_ctr', 'userid_category2nd_ctr', 'userid_history_duration_mean',
 'docid_history_duration_mean',
 'category1st_history_duration_mean',
 'category2nd_history_duration_mean',
 'userid_category1st_history_duration_mean',
 'userid_category2nd_history_duration_mean']+['userid_history_duration_std',
 'docid_history_duration_std',
 'category1st_history_duration_std',
 'category2nd_history_duration_std',
 'userid_category1st_history_duration_std',
 'userid_category2nd_history_duration_std','userid_count',
 'docid_count',
 'category1st_count',
 'category2nd_count',
 'userid_category1st_count',
 'userid_category2nd_count']


# In[14]:


for col in tqdm(sparse_features):
    lbe = LabelEncoder()
    df_data[col] = lbe.fit_transform(df_data[col])
    df_data[col] = df_data[col] + 1
    df_data[col].fillna(0, inplace=True)


# In[15]:


for col in tqdm(dense_features):
    df_data[col] = (df_data[col] - df_data[col].min()) / (df_data[col].max() - df_data[col].min())
    df_data[col].fillna(0, inplace=True)


# In[16]:


df_data = reduce_mem(df_data, cols=[f for f in df_data.columns if f not in ['userid', 'docid', 'id', 'dt']])


# In[17]:


df_data.drop(columns=['timestamp', 'dt', 'pubtime', 'day'], inplace=True)


# In[18]:


df_data.head()


# In[19]:


os.makedirs(f'{data_path}', exist_ok=True)
df_data.to_pickle(f'{data_path}/feature.pkl')


# In[20]:


with open(os.path.join(data_path, 'dense_features.pkl'), 'wb') as f:
    pickle.dump(dense_features, f)


# In[ ]:




