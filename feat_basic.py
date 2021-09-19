#!/usr/bin/env python
# coding: utf-8

# In[15]:


import warnings
warnings.simplefilter('ignore')
import pickle
import os
import re
import gc

from sklearn.preprocessing import LabelEncoder

import numpy as np
import pandas as pd
pd.set_option('max_columns', None)
pd.set_option('max_rows', 500)
from tqdm import tqdm

from utils import reduce_mem


# In[2]:


data_path = '/media/user01/wd1tb/yidian/data'
feat_path = '/media/user01/wd1tb/yidian/feat'


# In[3]:


df_train = pd.read_pickle(f'{data_path}/train.pkl')
df_test = pd.read_pickle(f'{data_path}/test.pkl')


# In[4]:


df_test['date'].value_counts()


# In[5]:


df_data = pd.concat([df_train, df_test], sort=False)


# In[6]:


df_data.head()


# # user_info

# In[7]:


user_info = pd.read_pickle(f'{data_path}/user_info.pkl')
user_info.head()


# In[8]:


df_data = df_data.merge(user_info, how='left')


# # doc_info

# In[9]:


doc_info = pd.read_pickle(f'{data_path}/doc_info.pkl')
doc_info.head()


# In[10]:


all_keywords = set()

def get_all_keyword(x):
    global max_len, min_len
    if x == '':
        return

    splts = x.split(',')

    for sp in splts:
        keyword = sp.split(':')[0]
        all_keywords.add(keyword)


doc_info['keyword'].fillna('', inplace=True)
doc_info['keyword'].apply(get_all_keyword)


# In[11]:


keyword2id = dict(zip(all_keywords, range(1, len(all_keywords) + 1)))


def keyword_map(x):
    if x == '':
        return []

    keys = []
    for sp in x.split(','):
        keyword = sp.split(':')[0]
        keys.append(keyword)

    ret = [keyword2id[key] for key in keys]
    return ret


doc_info['keyword'] = doc_info['keyword'].apply(keyword_map)


# In[12]:


doc_info['docid'] = doc_info['docid'].astype('int')
df_data = df_data.merge(
    doc_info[['docid', 'category1st', 'category2nd', 'keyword', 'pubtime']], how='left', on='docid')


# In[13]:


df_data.tail()


# In[16]:


sparse_features = [
    'userid', 'docid', 'network', 'device', 'os', 'province', 'city', 'age',
    'gender', 'category1st', 'category2nd'
]

for col in tqdm(sparse_features):
    lbe = LabelEncoder()
    df_data[col] = lbe.fit_transform(df_data[col])
    df_data[col] = df_data[col] + 1
    df_data[col].fillna(0, inplace=True)


# In[17]:


df_data = reduce_mem(df_data, cols=[f for f in df_data.columns if f not in ['userid', 'docid', 'id', 'dt']])


# In[18]:


df_data.head()


# In[19]:


os.makedirs(f'{feat_path}', exist_ok=True)
df_data.to_pickle(f'{feat_path}/feat_basic.pkl')


# In[ ]:




