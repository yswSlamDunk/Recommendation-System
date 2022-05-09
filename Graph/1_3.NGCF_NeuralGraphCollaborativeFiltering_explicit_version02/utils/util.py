import json
import torch
import pandas as pd
import pickle

from pathlib import Path
from itertools import repeat
from collections import OrderedDict


def write_model(model, fname='model.pickle'):
    # 지금 여기에 모델 저장하는 부분 작성 필요.
    with fname.open('wb') as handle:
        pickle.dump(model, handle)


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname='model.conf'):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def prepare_device(n_gpu_use):
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print('Warning: There\'s no GPU available on this machine, training will be performed on CPU.')
        n_gpu_use = n_gpu
    device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")

    list_ids = list(range(n_gpu_use))
    return device, list_ids


def rearrange_train_test_split(train, test):
    test_only_item = list(set(test['movie_id'].unique(
    ).flatten()) - set(train['movie_id'].unique().flatten()))
    test_only_user = list(set(test['user_id'].unique(
    ).flatten()) - set(train['user_id'].unique().flatten()))

    if len(test_only_user) != 0:
        test_only = test[test['user_id'].isin(test_only_user)]
        train = pd.concat([train, test_only], axis=0)
        test = test[~test['user_id'].isin(test_only_user)]

    if len(test_only_item) != 0:
        test_only = test[test['movie_id'].isin(test_only_item)]
        train = pd.concat([train, test_only], axis=0)
        test = test[~test['movie_id'].isin(test_only_item)]

    return train, test
