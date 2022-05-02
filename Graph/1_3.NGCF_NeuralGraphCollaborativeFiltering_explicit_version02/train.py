from re import S

import os
import argparse
import collections
import torch
import numpy as np
import pandas as pd
import logging

from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils import read_json, rearrange_train_test_split

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
