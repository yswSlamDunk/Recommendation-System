import os
import numpy as np
import pandas as pd
import logging
import hashlib
import argparse

from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import data_loader
import logger

from graph import getSparseGraph
from model import model_Light_GCN
from utils import *

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(user_name, make_graph, config_path = os.path.join(os.getcwd(), 'config.json'), logger_path = os.path.join(os.getcwd(), 'logger/logger_config.json')):
    config = read_json(config_path)

    device = prepare_device()

    base_dir = Path(config['train']['save_dir'])
    base_id = datetime.now().strftime(r'%y%m%d_%H%M%S')

    save_dir = base_dir / user_name / 'models' / base_id
    log_dir = base_dir / user_name / 'log' / base_id

    # make direcotry for saving check
    save_dir.mkdir(parents = True, exist_ok = True)
    log_dir.mkdir(parents = True, exist_ok = True)

    # make train_valid data & preprocessing data
    file_name = config['data']['file_name']
    columns = config['data']['columns']
    origin_df = pd.read_csv(os.path.join(config['data']['data_dir'], file_name), encoding = 'latin1', on_bad_lines = 'skip', sep = ';', header = 0)

    user_le = LabelEncoder().fit(origin_df[columns[0]])
    item_le = LabelEncoder().fit(origin_df[columns[1]])

    origin_df[columns[0]] = user_le.transform(origin_df[columns[0]])
    origin_df[columns[1]] = item_le.transform(origin_df[columns[1]])

    train_df, test_df = train_test_split(origin_df, test_size = config['preprocessing']['validation_split'], random_state = SEED)
    train_df, test_df = rearrange_train_test_split(train_df, test_df, config)


    train_dataset = data_loader.CustomDataset(train_df, device)
    test_dataset = data_loader.CustomDataset(test_df, device)
    train_dataloader = DataLoader(train_dataset, batch_size = config['data_loader']['batch_size'], shuffle = config['data_loader']['shuffle'])
    test_dataloader = DataLoader(test_dataset, batch_size = config['data_loader']['batch_size'], shuffle = config['data_loader']['shuffle'])

    # make metric score
    total_best_score = np.inf

    # make graph or load graph
    if (make_graph == True) or ((make_graph == False) and (~check_graph(config))):
        graph = getSparseGraph(train_df, args, device)
        write_graph(config, graph)

    else:
        graph = load_graph(config)

    # make hyper parameter dict
    hyper_parameter_dict = {}
    for learning_rate in config['optimizer']['lr']:
        for regs in config['optimizer']['regs']:
            for scheduler_gamma in config['lr_scheduler']['gamma']:
                for embed_size in config['model']['embed_size']:
                    for num_layer in config['model']['num_layers']:
                        for node_dropout in config['model']['node_dropout']:

                            patience = config['train']['early_stop']

                            best_score = np.inf

                            hyper_parameter_dict['model'] = {}
                            hyper_parameter_dict['optimizer'] = {}
                            hyper_parameter_dict['lr_scheduler'] = {}

                            hyper_parameter_dict['model']['node_dropout'] = node_dropout
                            hyper_parameter_dict['model']['num_layers'] = num_layer
                            hyper_parameter_dict['model']['embed_size'] = embed_size
                            hyper_parameter_dict['lr_scheduler']['gamma'] = scheduler_gamma
                            hyper_parameter_dict['optimizer']['regs'] = regs
                            hyper_parameter_dict['optimizer']['lr'] = learning_rate

                            hash_key = hashlib.sha1(str(hyper_parameter_dict).encode()).hexdigest()[:8]

                            logger.setup_logging(log_dir, hash_key)

                            logger_instance = logging.getLogger(user_name)
                            logger_instance.setLevel(config['train']['logging_verbosity'])

                            model = model_Light_GCN.Light_GCN(hyper_parameter_dict, config, user_le, item_le, train_df, device, graph)

                            optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = regs)
                            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = config['lr_scheduler']['step_size'], gamma = scheduler_gamma)

                            criterion = nn.MSEloss()

                            for epoch in range(config['train']['epoch']):
                                train_loss = 0

                                model.train()
                                for users, items, labels in train_dataloader:
                                    hat_labels = model(users, items)
                                    batch_loss = criterion(hat_labels, labels)
                                    optimizer.zero_grad()
                                    batch_loss.backward()
                                    optimizer.step()
                                    train_loss += batch_loss.item() / len(train_dataloader)

                                test_loss = 0
                                scheduler.step()

                                with torch.no_grad():
                                    model.eval()
                                    for users, items, labels in test_dataloader:
                                        hat_labels = model(users, items)
                                        batch_loss = criterion(hat_labels, labels)
                                        test_loss += batch_loss.item() / len(test_dataloader)

                                logger_instance.info('epoch : {}, train_loss : {}, test_loss : {}'.format(epoch, round(train_loss, 4), round(test_loss, 4)))

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
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--user", dest = "user", action = "store")
    parser.add_argument("-g", "--make graph", dest = "graph", action = "store", default = True)
    args = parser.parse_args()

    main(args.user, args.graph)

                                    