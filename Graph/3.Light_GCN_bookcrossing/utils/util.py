import os
import json
import torch
import pandas as pd
import pickle

from pathlib import Path
from collections import OrderedDict


def check_graph(args):
    check = os.path.isfile(os.path.join(
        args['data']['data_dir'], args['data']['graph_name']))
    return check


def write_graph(args, graph):
    with open(os.path.join(args['data']['data_dir'], args['data']['graph_name']), 'wb') as f:
        pickle.dump(graph, f)


def load_graph(args):
    with open(os.path.join(args['data']['data_dir'], args['data']['grapn_name']), 'rb') as f:
        graph = pickle.load(f)
        return graph


def write_model(model, fname='model.pickle'):
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


def prepare_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def rearrange_train_test_split(train, test, args):
    test_only_user = list(set(test[args['data']['columns'][0]].unique(
    ).flatten()) - set(train[args['data']['columns'][0]].unique().flatten()))
    test_only_item = list(set(test[args['data']['columns'][1]].unique(
    ).flatten()) - set(train[args['data']['columns'][1]].unique().flatten()))

    if len(test_only_user) != 0:
        test_only = test[test[args['data']['columns'][0]].isin(test_only_user)]
        train = pd.concat([train, test_only], axis=0)
        test = test[~test[args['data']['columns'][0]].isin(test_only_user)]

    if len(test_only_item) != 0:
        test_only = test[test[args['data']['columns'][1]].isin(test_only_item)]
        train = pd.concat([train, test_only], axis=0)
        test = test[~test[args['data']['columns'][1]].isin(test_only_item)]

    return train, test
