from re import S

import os
import argparse
import collections
import numpy as np
import pandas as pd
import logging
import hashlib
import logging

from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sympy import N

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import data_loader
import graph
import logger
from model import model_NGCF
from utils import read_json, rearrange_train_test_split, write_json, prepare_device, write_model

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(user_name, config_path=os.path.join(os.getcwd(), 'config.json'), logger_path=os.path.join(os.getcwd(), 'logger/logger_config.json')):
    config = read_json(config_path)

    device, list_ids = prepare_device(config['cuda']['n_gpu'])

    base_dir = Path(config['train']['save_dir'])
    base_id = datetime.now().strftime(r'%y%m%d_%H%M%S')

    save_dir = base_dir / user_name / 'models' / base_id
    log_dir = base_dir / user_name / 'log' / base_id

    # make directory for saving check
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

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
    data_generator = graph.Graph(train_df, config)
    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()

    # make metric score
    total_best_score = np.inf

    # make hyper parameter dict
    hyper_parameter_dict = {}
    for learning_rate in config['optimizer']['lr']:
        for regs in config['optimizer']['regs']:
            for scheduler_gamma in config['lr_scheduler']['gamma']:
                for node_dropout in config['model']['node_drop']:
                    for embed_size in config['model']['embed_size']:

                        patience = config['train']['early_stop']

                        best_score = np.inf

                        hyper_parameter_dict['model'] = {}
                        hyper_parameter_dict['optimizer'] = {}
                        hyper_parameter_dict['lr_scheduler'] = {}

                        hyper_parameter_dict['model']['node_drop'] = node_dropout
                        hyper_parameter_dict['model']['embed_size'] = embed_size

                        hyper_parameter_dict['lr_scheduler']['gamma'] = scheduler_gamma
                        hyper_parameter_dict['optimizer']['regs'] = regs
                        hyper_parameter_dict['optimizer']['lr'] = learning_rate

                        hash_key = hashlib.sha1(
                            str(hyper_parameter_dict).encode()).hexdigest()[:8]

                        logger.setup_logging(log_dir, hash_key)

                        logger_instance = logging.getLogger(user_name)
                        logger_instance.setLevel(
                            config['train']['logging_verbosity'])

                        model = model_NGCF.NGCF(data_generator.n_users,
                                                data_generator.n_items,
                                                norm_adj,
                                                hyper_parameter_dict,
                                                config,
                                                user_le,
                                                item_le,
                                                train_df,
                                                device).to(device)

                        # if (len(list_ids) > 1) & (device != "cpu"):
                        if (len(list_ids) > 1):
                            model = torch.nn.DataParallel(
                                model, device_ids=config['cuda']['device_ids'])

                        optimizer = optim.Adam(
                            model.parameters(), lr=learning_rate, weight_decay=regs)
                        scheduler = optim.lr_scheduler.StepLR(
                            optimizer, step_size=config['lr_scheduler']['step_size'], gamma=scheduler_gamma)

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
                            scheduler.step()

                            with torch.no_grad():
                                model.eval()
                                for users, items, labels in test_dataloader:
                                    users_embedding, items_embedding = model(
                                        users, items)

                                    batch_loss = model.loss(
                                        users_embedding, items_embedding, labels)
                                    test_loss += batch_loss.item() / len(test_dataloader)

                            logger_instance.info('epoch : {}, train_loss : {}, test_loss : {}'.format(
                                epoch, round(train_loss, 4), round(test_loss, 4)))

                            if test_loss < best_score:
                                best_score = test_loss
                                patience = config['train']['early_stop']

                            else:
                                patience -= 1
                                if patience == 0:
                                    break

                        if best_score < total_best_score:
                            total_best_score = best_score

                            save_config = config.copy()
                            save_config.update(hyper_parameter_dict)

                            write_json(save_config, save_dir / 'model.json')
                            write_model(model, save_dir / 'model.pickle')


if __name__ == '__main__':
    main('test03')
