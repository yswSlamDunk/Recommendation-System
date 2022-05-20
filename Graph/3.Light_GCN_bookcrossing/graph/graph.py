import numpy as np
import scipy.sparse as sp

import torch


def make_train_matrix(train_df, args):
    rows, cols = train_df[args['data']['columns']
                          [0]], train_df[args['data']['columns'][1]]
    values = train_df[args['data']['columns'][2]]

    n_user = train_df[args['data']['columns'][0]].nunique()
    n_item = train_df[args['data']['columns'][1]].nunique()

    train_matrix = sp.csr_matrix(
        (values, (rows, cols)), dtype='float64', shape=(n_user, n_item))

    return train_matrix


def convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))


def getSparseGraph(train_df, args, device):
    train_matrix = make_train_matrix(train_df, args)

    n_users, n_items = train_matrix.shape

    adj_mat = sp.dok_matrix(
        (n_users + n_items, n_users + n_items), dtype=np.float32)
    adj_mat = adj_mat.tolil()

    R = train_matrix.tolil()

    adj_mat[:n_users, n_users:] = R
    adj_mat[n_users:, :n_users] = R.T
    adj_mat = adj_mat.todok()

    rowsum = np.array(adj_mat.sum(axis=1))
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)

    norm_adj = d_mat.dot(adj_mat)
    norm_adj = norm_adj.dot(d_mat)
    norm_adj = norm_adj.tocsr()

    Graph = convert_sp_mat_to_sp_tensor(norm_adj)
    Graph = Graph.coalesce().to(device)

    return Graph
