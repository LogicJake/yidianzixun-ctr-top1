#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.simplefilter('ignore')
import pickle
import os
import re
import gc

import numpy as np
import pandas as pd
pd.set_option('max_columns', None)
pd.set_option('max_rows', 500)
from tqdm import tqdm

from utils import reduce_mem


# In[2]:


feat_path = '/media/user01/wd1tb/yidian/feat'


# In[3]:


df = pd.read_pickle(f'{feat_path}/feat_basic.pkl')


# In[4]:


df.drop(columns=['keyword', 'dt'], inplace=True)


# In[5]:


df.head()


# In[6]:


dates = df['date'].unique()
dates.sort()
date_map = dict(zip(dates, range(len(dates))))
df['day'] = df['date'].map(date_map)


# In[7]:


for feat in tqdm([['userid'], ['docid'], ['category1st'], ['category2nd'],
                  ['userid', 'category1st'], ['userid', 'category2nd']]):
    res_arr = []
    for d in range(1, max(date_map.values()) + 1):
        df_temp = df[((df['day']) < d)]
        df_temp = df_temp.groupby(feat).size().reset_index()
        df_temp.columns = feat + [f'{"_".join(feat)}_history_count']
        df_temp['day'] = d
        res_arr.append(df_temp)
    stat_df = pd.concat(res_arr)

    df = df.merge(stat_df, how='left', on=feat + ['day'])


# In[8]:


# 目标转化率
target = 'click'
for gp in tqdm([['userid'], ['docid'], ['category1st'], ['category2nd'],
                ['userid', 'category1st'], ['userid', 'category2nd']]):
    res_arr = []
    name = f"{'_'.join(gp)}_ctr"
    
    for d in range(1, max(date_map.values()) + 1):
        temp = df[((df['day']) < d)]
        temp = temp.groupby(gp)[target].agg([(name, 'mean')]).reset_index()
        temp['day'] = d
        res_arr.append(temp)
    stat_df = pd.concat(res_arr)

    df = df.merge(stat_df, how='left', on=gp + ['day'])


# In[ ]:


target = 'duration'
for gp in tqdm([['userid'], ['docid'], ['category1st'], ['category2nd'],
                ['userid', 'category1st'], ['userid', 'category2nd']]):
    res_arr = []
    name_mean = f"{'_'.join(gp)}_history_duration_mean"
    name_std = f"{'_'.join(gp)}_history_duration_std"
    
    for d in range(1, max(date_map.values()) + 1):
        temp = df[((df['day']) < d)]
        temp = temp.groupby(gp)[target].agg([(name_mean, 'mean'), (name_std, 'std')]).reset_index()
        temp['day'] = d
        res_arr.append(temp)
    stat_df = pd.concat(res_arr)

    df = df.merge(stat_df, how='left', on=gp + ['day'])
    
new_columns


# In[ ]:


new_columns


# In[ ]:


df = reduce_mem(df, cols=[f for f in df.columns if f not in ['userid', 'docid', 'id', 'dt']])


# In[ ]:


df.head()


# In[ ]:


os.makedirs(f'{feat_path}', exist_ok=True)
df.to_pickle(f'{feat_path}/feat_basic_history_all.pkl')


# In[ ]:




