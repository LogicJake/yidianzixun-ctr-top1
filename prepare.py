#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


# In[2]:


raw_data_path = '/media/user01/wd1tb/yidian/data_for_ctr_predict'
data_path = '/media/user01/wd1tb/yidian/data'


# In[3]:


df_train = pd.read_csv(f'{raw_data_path}/train_data.txt',
                       sep='\t',
                       header=None)
df_train.columns = [
    'userid', 'docid', 'timestamp', 'network', 'refresh', 'position', 'click',
    'duration'
]


# In[4]:


df_train.head()


# In[5]:


df_train['dt'] = pd.to_datetime(df_train['timestamp'], utc=True,
                                unit='ms').dt.tz_convert('Asia/Shanghai')
df_train['date'] = df_train['dt'].dt.date
df_train['date'] = df_train['date'].astype('str')


# In[6]:


# # 采样
# df_train = df_train[df_train['date'] >= '2021-07-01']


# In[7]:


df_train['date'].value_counts()


# In[8]:


os.makedirs(f'{data_path}', exist_ok=True)


# In[9]:


df_train.to_pickle(f'{data_path}/train.pkl')


# # 测试集

# In[10]:


df_test = pd.read_csv(f'{raw_data_path}/test_data.txt', sep='\t', header=None)
df_test.columns = ['id', 'userid', 'docid', 'timestamp', 'network', 'refresh']


# In[11]:


df_test['dt'] = pd.to_datetime(df_test['timestamp'], utc=True,
                               unit='ms').dt.tz_convert('Asia/Shanghai')
df_test['date'] = df_test['dt'].dt.date
df_test['date'] = df_test['date'].astype('str')


# In[12]:


df_test['date'].value_counts()


# In[13]:


df_test.head()


# In[14]:


df_test.to_pickle(f'{data_path}/test.pkl')


# # user_info

# In[15]:


user_info = pd.read_csv(f'{raw_data_path}/user_info.txt',
                        sep='\t',
                        header=None)
user_info.columns = [
    'userid', 'device', 'os', 'province', 'city', 'age', 'gender'
]


# In[16]:


user_info.head()


# In[17]:


def get_cate(x):
    if type(x) == float:
        return x
    li = x.split(',')
    res = list()
    for i in li:
        lbl, prob = i.split(':')
        res.append([lbl, float(prob)])
    res = sorted(res, key=lambda x: x[1])
    return res[-1][0]


user_info['age'] = user_info['age'].apply(lambda x: get_cate(x))
user_info['gender'] = user_info['gender'].apply(lambda x: get_cate(x))


# In[18]:


# label encoding
for col in tqdm(
    ['device', 'os', 'province', 'city', 'age', 'gender']):
    lbe = LabelEncoder()
    user_info[col] = user_info[col].fillna('NAN')
    user_info[col] = lbe.fit_transform(user_info[col])

user_info.head()


# In[19]:


user_info.to_pickle(f'{data_path}/user_info.pkl')


# # doc_info

# In[29]:


with open(f'{raw_data_path}/doc_info.txt', 'r') as fd:
    doc_text = fd.read().split('\n')[:-1]
    
doc_data = list()
for text in doc_text:
    doc_data.append(text.split('\t'))
    
doc_info = pd.DataFrame(doc_data)
doc_info.columns = ['docid', 'title', 'pubtime', 'picnum', 'category1st', 'category2nd', 'keyword']
print(doc_info.shape)


# In[30]:


doc_info.head()


# In[31]:


# 脏数据
def clean_str(x):
    if x in ['上海', '云南', '山东', 'NoneType'] or x is None: return 0
    else: return int(x)

doc_info['picnum'] = doc_info['picnum'].apply(lambda x: clean_str(x))


# In[32]:


# 脏数据
def clean_str(x):
    if type(x) == str: return 0
    else: return x

doc_info['pubtime'] = doc_info['pubtime'].apply(lambda x: clean_str(x))


# In[33]:


for col in tqdm(['category1st', 'category2nd']):
    lbe = LabelEncoder()
    doc_info[col] = doc_info[col].fillna('NAN')
    doc_info[col] = lbe.fit_transform(doc_info[col])

doc_info.head()


# In[ ]:


doc_info.to_pickle(f'{data_path}/doc_info.pkl')


# In[ ]:




