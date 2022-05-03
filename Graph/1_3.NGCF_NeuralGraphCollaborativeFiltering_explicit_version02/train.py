from re import S

import os
import argparse
import collections
import numpy as np
import pandas as pd
import logging
import hashlib

from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sympy import N
from utils import read_json, rearrange_train_test_split

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import data_loader
import graph
import model

SEED = 123
torch.maual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(user_name, config_path=os.path.join(os.getcwd(), 'config.json')):
    config = read_json(config_path)

    base_dir = Path(config['train']['save_dir'])
    base_id = datetime.now().strftime(r'%m%d_%H%M%S')

    save_dir = base_dir / user_name / 'models' / base_id
    log_dir = base_dir / user_name / 'log' / base_id

    # make directory for saving check
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # init_logging and setLevel
    logger = logging.getLogger(user_name)
    logger.setLevel(config['train']['logging_berbosity'])

    # make train_valid data & preprocessing data
    file_name = config['data']['file_name']
    columns = config['data']['columns']
    origin_df = pd.read_csv(os.path.join(
        config['data']['data_dir'], file_name), encoding='cp949', names=columns)

    user_le = LabelEncoder().fit(origin_df[columns[0]])
    item_le = LabelEncoder().fit(origin_df[columns[1]])

    origin_df[columns[0]] = user_le.transform(origin_df[columns[0]])
    origin_df[columns[1]] = item_le.transform(origin_df[columns[1]])

    train_df, test_df = train_test_split(
        origin_df, test_size=config['preprocessing']['validation_split'], random_state=SEED)
    train_df, test_df = rearrange_train_test_split(train_df, test_df)

    train_dataset = data_loader.CustomDataset(train_df)
    test_dataset = data_loader.CustomDataset(test_df)
    train_dataloader = DataLoader(
        train_dataset, batch_size=config['data_loader']['batch_size'], shuffle=config['data_loader']['shuffle'])
    test_dataloader = DataLoader(
        test_dataset, batch_size=config['data_loader']['batch_size'], shuffle=config['data_loader']['shuffle'])

    # make graph
    data_generator = graph.Graph()
    plain_adj, norm_adj, mean_adj = data_generator.get.adj_mat()
    # make metric score

    # make hyper parameter dict
    hyper_parameter_dict = {}

    for lr in config['optimizer']['lr']:
        for regs in config['optimizer']['regs']:
            for gamma in config['lr_scheduler']['gamma']:
                for node_dropout in config['model']['node_drop']:
                    for embed_size in config['model']['embed_size']:

                        hyper_parameter_dict['node_drop'] = node_dropout
                        hyper_parameter_dict['gamma'] = gamma
                        hyper_parameter_dict['regs'] = regs
                        hyper_parameter_dict['lr'] = lr
                        hyper_parameter_dict['embed_size'] = embed_size

                        hash_key = hashlib.sha1(
                            str(hyper_parameter_dict).encode()).hexdigest()[:8]

                        model = model.NGCF(data_generator.n_users,
                                           data_generator.n_items,
                                           norm_adj,
                                           hyper_parameter_dict,
                                           config)

                        optimizer = optim.Adam(
                            model.parameters(), lr=hyper_parameter_dict['lr'])

                        for epoch in range(config['train']['epoch']):
                            train_loss = 0

                            model.train()
                            for users, items, labels in train_dataloader:
                                users_embedding, items_embedding = model(
                                    users, items)
                                batch_loss = model.loss(
                                    users_embedding, items_embedding, labels)
                                optimizer.zero_grad()
                                batch_loss.backward()
                                optimizer.step()
                                train_loss += batch_loss.item() / len(train_dataloader)

                            test_loss = 0

                            with torch.no_grad():
                                model.eval()
                                for users, items, labels in test_dataloader:
                                    users_embedding, items_embedding = model(
                                        users, items)

                                    batch_loss = model.loss(
                                        users_embedding, items_embedding, labels)
                                    test_loss += batch_loss.item() / len(test_dataloader)

                                # 여기에 logging하는 작업이 필요

    '''
    그런데 여기 아래의 과정에서 hyper parameter 관련 for문을 생성해야함
    그리고 logging도 신경써야함
    평가 metric의 성능도 확인해야함
    
    1. dataloader를 만들어야 함
    2. model을 생성하고 logging 해야함
    3. model의 gpu 관련 병렬처리를 해야함
    4. loss를 선언해야함
    5. metrics를 만들어야함
    6. optimzier를 선언해야함
    7. lr_scheduler를 선언해야함
    8. ...
    '''
