import gc
import logging
import os
import pickle
import time
from operator import sub

import pandas as pd
import torch
from torch.utils.data import DataLoader

from mmoe import DocRec
from model_tools import BuildDataSet, model_evaluate, model_train
from utils import random_seed
import numpy as np

seed = 1996
random_seed(seed)

data_path = '/media/user01/wd1tb/yidian/data'
feat_path = '/media/user01/wd1tb/yidian/feat'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print(device)

os.makedirs('logging', exist_ok=True)
os.makedirs('sub', exist_ok=True)

file = 'train_model_' + time.strftime('%Y-%m-%d_%H-%M-%S') + '.log'
logging_filename = os.path.join('logging', file)
logging.basicConfig(filename=logging_filename,
                    format='%(levelname)s: %(message)s',
                    level=logging.DEBUG)

# 加载预训练权重
def load_weights():
    weights = {}
#     weights['docid'] = np.load(f'{feat_path}/docid_emb.npy')
    return weights

if __name__ == "__main__":
    mode = 'online'
    batch_size = 800

    df_data = pd.read_pickle(f'{data_path}/feature.pkl')

    # 连续特征
    with open(os.path.join(data_path, 'dense_features.pkl'), 'rb') as f:
        dense_features = pickle.load(f)

    sparse_features = {
        'userid': df_data['userid'].nunique() + 1,
        'docid': df_data['docid'].nunique() + 1,
        'network': df_data['network'].nunique() + 1,
        'device': df_data['device'].nunique() + 1,
        'os': df_data['os'].nunique() + 1,
        'province': df_data['province'].nunique() + 1,
        'city': df_data['city'].nunique() + 1,
        'age': df_data['age'].nunique() + 1,
        'gender': df_data['age'].nunique() + 1,
        'category1st': df_data['category1st'].nunique() + 1,
        'category2nd': df_data['category2nd'].nunique() + 1,
        'keyword': 1044103 + 1,
    }

    logging.debug(f'离散特征：{list(sparse_features.keys())}')
    logging.debug(f'连续特征：{dense_features}')

    if mode == 'offline':
#         train = df_data[(df_data['date'] <= '2021-07-05') & (df_data['date'] >= '2021-07-02')]
        train = df_data[(df_data['date'] <= '2021-07-05')]
    else:
#         train = df_data[(df_data['date'] <= '2021-07-06') & (df_data['date'] >= '2021-07-03')]
        train = df_data[(df_data['date'] <= '2021-07-06')]

    val = df_data[(df_data['date'] == '2021-07-06') & (df_data['id'].isnull())]
    test = df_data[df_data['id'].notnull()]
    submit = test[['id']].copy()
    submit['id'] = submit['id'].astype('int')

    del df_data
    gc.collect()
    
    logging.debug(
        f'训练集维度: {train.shape} 验证集维度: {val.shape} 测试集维度: {test.shape}')
    
    train_dataset = BuildDataSet(train, dense_features)
    train_loader = DataLoader(train_dataset,
                              num_workers=2,
                              batch_size=batch_size,
                              shuffle=True)
    del train
    gc.collect()
    print('train over')
    
    val_dataset = BuildDataSet(val, dense_features)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            num_workers=2,
                            shuffle=False)

    del val
    gc.collect()
    print('val over')

    test_dataset = BuildDataSet(test, dense_features, is_test=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             num_workers=2,
                             shuffle=False)
    
    del test
    gc.collect()
    print('test over')

    weights = load_weights()
    model = DocRec(sparse_features=sparse_features,
                   dense_features=dense_features,
                   embedding_dim=128,
                   device=device,
                   weights=weights)
    logging.info(model)
    best_model = model_train(model=model,
                             train_iter=train_loader,
                             valid_iter=val_loader,
                             epochs=1 if mode == 'offline' else 1,
                             device=device,
                             early_stopping=1 if mode == 'offline' else 0)

    # 模型评估
    val_auc, _ = model_evaluate(model=best_model,
                                data_iter=val_loader,
                                device=device)
    logging.info(val_auc)

    # 模型预测
    test_pred = model_evaluate(model=best_model,
                               data_iter=test_loader,
                               device=device,
                               test=True)

    # 保存提交文件
    submit['click'] = test_pred
    print(submit.head())

    os.makedirs('sub', exist_ok=True)
    submit.to_csv(os.path.join('sub', f'{mode}_{val_auc}.csv'), index=False)
