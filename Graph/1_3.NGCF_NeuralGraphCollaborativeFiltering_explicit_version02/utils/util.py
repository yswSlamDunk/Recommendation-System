import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def preprare_device(n_gpu_use):
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print('Warning: There\'s no GPU available on this machine, training will be performed on CPU.')
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
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


# class MetricTracker:
#     def __init__(self, *keys, writer=None):
#         self.writer = writer
#         self._data = pd.DataFrame(
#             index=keys, columns=['total', 'counts', 'average'])
#         self.reset()

#     def reset(self):
#         for col in self._data.columns:
#             self._data[col].values[:] = 0

#     def update(self, key, value, n=1):
#         if self.writer is not None:
#             self.writer.add_scalar(key, value)
#         self._data.total[key] += value * n
#         self._data.counts[key] += n
#         self._data.everate[key] = self._data.total[key] / \
#             self._data.counts[key]

#     def avg(self, key):
#         return self._data.average[key]

#     def result(self):
#         return dict(self._data.average)
