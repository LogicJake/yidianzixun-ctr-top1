import copy
import logging
import os
import time
import warnings

import numpy as np
import torch
import torch.utils.data as Data
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class BuildDataSet(Data.Dataset):
    def __init__(self, df, dense_features, is_test=False):
        # 离散 id
        self.userid_list = df['userid'].values
        self.docid_list = df['docid'].values
        self.network_list = df['network'].values
        self.device_list = df['device'].values
        self.os_list = df['os'].values
        self.province_list = df['province'].values
        self.city_list = df['city'].values
        self.age_list = df['age'].values
        self.gender_list = df['gender'].values
        self.category1st_list = df['category1st'].values
        self.category2nd_list = df['category2nd'].values

        df.drop(columns=['userid', 'docid', 'network', 'device', 'os', 'province', 'city', 'age', 'gender', 'category1st', 'category2nd'], inplace=True)
        
        # 连续特征
        self.dense_features_list = df[dense_features].values
        df.drop(columns=dense_features, inplace=True)
        
        # 变长 id
        self.keyword_list = df['keyword'].values

        df.drop(columns=['keyword', 'history_docid'], inplace=True)
        
        # 标签
        if not is_test:
            self.click_list = df['click'].values

        self.is_test = is_test

    def _pad_seq(self, seq, max_len, truncation='pre', dtype='int'):
        if type(seq) == float:
            seq = []

        if truncation == 'post':
            seq = seq[-max_len:]
        else:
            seq = seq[:max_len]
        pad_size = max_len - len(seq)
        if pad_size > 0:
            seq = [0] * pad_size + seq
        seq = np.array(seq, dtype=dtype)
        return seq

    def __getitem__(self, index):
        # 离散 id
        userid = self.userid_list[index]
        docid = self.docid_list[index]
        network = self.network_list[index]
        device_t = self.device_list[index]
        os = self.os_list[index]
        province = self.province_list[index]
        city = self.city_list[index]
        age = self.age_list[index]
        gender = self.gender_list[index]
        category1st = self.category1st_list[index]
        category2nd = self.category2nd_list[index]

        # 连续特征
        dense_features = self.dense_features_list[index]

        keywords = self._pad_seq(self.keyword_list[index], max_len=20)
                
        # 标签
        if not self.is_test:
            click = self.click_list[index]
        else:
            click = 0

        return userid, docid, network, device_t, os, province, city, age, gender, category1st, category2nd, dense_features, keywords, click

    def __len__(self):
        return len(self.userid_list)

def model_train(model,
                train_iter,
                valid_iter,
                epochs,
                device,
                early_stopping=0):
    start_time = time.time()

    # Train!
    logger.info("******************** Running training ********************")
    logger.info("  Num Epochs = %d", epochs)
    logger.info("  Train device:%s", device)

    valid_best_auc = 0
    last_improve = 0

    best_model = copy.deepcopy(model)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        logger.info('Epoch [{}/{}]'.format(epoch + 1, epochs))

        click_predict_all = []
        click_label_all = []

        for (userid, docid, network, device_t, os, province, city, age, gender, category1st, category2nd, dense_features, keywords, click) in tqdm(train_iter):
            model.train()

            # 离散特征
            userid = userid.to(device)
            docid = docid.to(device)
            network = network.to(device)
            device_t = device_t.to(device)
            os = os.to(device)
            province = province.to(device)
            city = city.to(device)
            age = age.to(device)
            gender = gender.to(device)
            category1st = category1st.to(device)
            category2nd = category2nd.to(device)

            # 连续特征
            dense_features = dense_features.float().to(device)

            # 变长id
            keywords = keywords.to(device)

            # 标签
            click = click.float().to(device)

            # 预测值
            predict, loss = model(userid, docid, network, device_t, os,
                                  province, city, age, gender, category1st,
                                  category2nd, dense_features, keywords, click)

            loss.backward()
            optimizer.step()
            model.zero_grad()

            click_predict_all.extend(list(predict.cpu().detach().numpy()))
            click_label_all.extend(list(click.cpu().detach().numpy()))

        train_auc = roc_auc_score(click_label_all, click_predict_all)
        valid_auc, _ = model_evaluate(model=model,
                                      data_iter=valid_iter,
                                      device=device)

        if valid_auc > valid_best_auc:
            valid_best_auc = valid_auc
            improve = '*'
            best_model = copy.deepcopy(model)
            last_improve = epoch
        else:
            improve = ''

        time_dif = time.time() - start_time

        msg = f'Train auc: {train_auc} Valid auc: {valid_auc} Time: {time_dif} {improve}'
        logger.info(msg)

        if early_stopping > 0 and epoch - last_improve >= early_stopping:
            logger.info('No optimization for a long time, auto-stopping...')
            break

    if early_stopping > 0:
        return best_model
    else:
        return model

def model_evaluate(model, data_iter, device, test=False):
    model.eval()

    click_predict_all = []
    click_label_all = []

    with torch.no_grad():
        for (userid, docid, network, device_t, os, province, city, age, gender, category1st, category2nd, dense_features, keywords, click) in data_iter:
            # 离散特征
            userid = userid.to(device)
            docid = docid.to(device)
            network = network.to(device)
            device_t = device_t.to(device)
            os = os.to(device)
            province = province.to(device)
            city = city.to(device)
            age = age.to(device)
            gender = gender.to(device)
            category1st = category1st.to(device)
            category2nd = category2nd.to(device)

            # 连续特征
            dense_features = dense_features.float().to(device)

            # 变长id
            keywords = keywords.to(device)

            click = click.float().to(device)

            predict, loss = model(userid, docid, network, device_t, os,
                                  province, city, age, gender, category1st,
                                  category2nd, dense_features, keywords, click)

            click_predict_all.extend(list(predict.cpu().detach().numpy()))

            if not test:
                click_label_all.extend(list(click.cpu().detach().numpy()))


    if test:
        return click_predict_all

    auc = roc_auc_score(click_label_all, click_predict_all)
    return auc, None
