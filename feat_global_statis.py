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


df = pd.read_pickle(os.path.join(feat_path, 'feat_basic.pkl'))


# In[4]:


df.drop(columns=['keyword', 'dt'], inplace=True)


# In[5]:


df.head()


# In[6]:


for f in tqdm(['userid', 'docid', 'category1st', 'category2nd']):
    df[f + '_count'] = df[f].map(df[f].value_counts())


# In[7]:


for f in tqdm([['userid', 'category1st'], ['userid', 'category2nd']]):
    df_temp = df.groupby(f).size().reset_index().rename(
        columns={0: f'{"_".join(f)}_count'})
    df = df.merge(df_temp, how='left', on=f)


# In[8]:


# for f1, f2 in tqdm([['userid', 'category1st'], ['userid', 'category2nd']]):
#     df['{}_in_{}_nunique'.format(f2,
#                                  f1)] = df.groupby(f1)[f2].transform('nunique')


# In[9]:


# for f1, f2 in tqdm([['userid', 'duration'],
#                     ['category1st', 'duration'],
#                     ['category2nd', 'duration']]):
#     df['{}_in_{}_mean'.format(f2, f1)] = df.groupby(f1)[f2].transform('mean')


# In[10]:


df.head()


# In[11]:


df = reduce_mem(df, cols=[f for f in df.columns if f not in ['userid', 'docid', 'id', 'dt']])


# In[12]:


os.makedirs(f'{feat_path}', exist_ok=True)
df.to_pickle(f'{feat_path}/feat_global_statis.pkl')


# In[ ]:




