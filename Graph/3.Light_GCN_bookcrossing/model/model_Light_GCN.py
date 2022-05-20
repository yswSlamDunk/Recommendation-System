import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

import utils


class Light_GCN(nn.Module):
    def __init__(self, hyper_parameter_dict, args, user_le, item_le, train_df, device, graph):
        super(Light_GCN, self).__init__()
        self.args = args

        self.n_user = len(user_le.classes_)
        self.n_item = len(item_le.classes_)

        self.user_le = user_le
        self.item_le = item_le
        self.train_df = train_df

        self.device = device

        self.embed_size = hyper_parameter_dict['model']['embed_size']
        self.num_layers = hyper_parameter_dict['model']['num_layers']
        self.num_folds = hyper_parameter_dict['model']['num_folds']
        self.node_dropout = hyper_parameter_dict['model']['node_dropout']

        self.split = args['model']['split']

        self.Graph = graph

        self.build_graph()

    def build_graph(self):
        self.user_embedding = nn.Embedding(self.n_user, self.embed_size)
        self.item_embedding = nn.Embedding(self.n_item, self.embed_size)

        nn.init.normal_(self.user_embedding.weight, 0, 0.01)
        nn.init.normal_(self.item_embedding.weight, 0, 0.01)

        self.to(self.device)

    def lightgcn_embedding(self, graph):
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight
        all_emb = torch.cat([users_emb, items_emb], dim=0)

        embs = [all_emb]

        g_droped = graph

        ego_emb = all_emb
        for k in range(self.num_layers):
            if self.split:
                tmp_emb = []
                for f in range(len(g_droped)):
                    tmp_emb.append(torch.sparse.mm(g_droped[f], ego_emb))
                side_emb = torch.cat(tmp_emb, dim=0)
                all_emb = side_emb

            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        lightgcn_out = torch.mean(embs, dim=1)
        users, items = torch.split(lightgcn_out, [self.n_user, self.n_item])

        return users, items

    # def make_train_matrix(self, train_df, args):
    #     rows, cols = train_df[args['data']['columns']
    #                           [0]], train_df[args['data']['columns'][1]]
    #     values = train_df[args['data']['columns'][2]]

    #     sp_data = sp.csr_matrix(
    #         (values, (rows, cols)), dtype='float64', shape=(self.n_user, self.n_item))
    #     self.train_matrix = sp_data

    # def _split_A_hat(self, A):
    #     A_fold = []
    #     fold_len = (self.n_user + self.n_item) // self.num_folds

    #     for i_fold in range(self.num_folds):
    #         start = i_fold * fold_len
    #         if i_fold == self.num_folds - 1:
    #             end = self.n_user + self.n_item
    #         else:
    #             end = (i_fold + 1) * fold_len
    #         A_fold.append(self._convert_sp_mat_to_sp_tensor(
    #             A[start:end]).coalesce().to(self.device))

    #     return A_fold

    # def _convert_sp_mat_to_sp_tensor(self, X):
    #     coo = X.tocoo().astype(np.float32)
    #     row = torch.Tensor(coo.row).long()
    #     col = torch.Tensor(coo.col).long()
    #     index = torch.stack([row, col])
    #     data = torch.FloatTensor(coo.data)
    #     return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    # def getSparseGraph(self):
    #     n_users, n_items = self.train_matrix.shape

    #     adj_mat = sp.dok_matrix(
    #         (n_users + n_items, n_users + n_items), dtype=np.float32)
    #     adj_mat = adj_mat.tolil()

    #     R = self.train_matrix.tolil()

    #     adj_mat[:n_users, n_users:] = R
    #     adj_mat[n_users:, :n_users] = R.T
    #     adj_mat = adj_mat.todok()

    #     rowsum = np.array(adj_mat.sum(axis=1))
    #     d_inv = np.power(rowsum, -0.5).flatten()
    #     d_inv[np.isinf(d_inv)] = 0.
    #     d_mat = sp.diags(d_inv)

    #     norm_adj = d_mat.dot(adj_mat)
    #     norm_adj = norm_adj.dot(d_mat)
    #     norm_adj = norm_adj.tocsr()

    #     if self.split == True:
    #         Graph = self._split_A_hat(norm_adj)

    #     else:
    #         Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
    #         Graph = Graph.coalesce().to(self.device)

    #     return Graph

    def predict_batch_users(self, user_ids):
        user_embeddings = F.embedding(user_ids, self.user_embedding_pred)
        item_embeddings = self.item_Embedding_pred
        return np.matmul(user_embeddings, item_embeddings.T)

    def foward(self, user, item):
        u_embedding, i_embedding = self.lightgcn_embedding(self.Graph)

        user_latent = F.embedding(user, u_embedding)
        item_latent = F.embedding(item, i_embedding)

        score = torch.mul(user_latent, item_latent).sum(1)

        return score
