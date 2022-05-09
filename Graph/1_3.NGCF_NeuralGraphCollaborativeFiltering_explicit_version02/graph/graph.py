import numpy as np
import scipy.sparse as sp


class Graph(object):
    def __init__(self, train_df, config):
        self.n_users = len(train_df[config['data']['columns'][0]].unique())
        self.n_items = len(train_df[config['data']['columns'][1]].unique())

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.exist_users = list(
            train_df[config['data']['columns'][0]].unique())

        for i, row in train_df.iterrows():
            self.R[row[config['data']['columns'][0]],
                   row[config['data']['columns'][1]]] = 1

    def get_adj_mat(self):
        adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
        return adj_mat, norm_adj_mat, mean_adj_mat

    def create_adj_mat(self):
        adj_mat = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def mean_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isninf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return bi_lap.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            return temp

        norm_adj_mat = mean_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = mean_adj_single(adj_mat)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()
