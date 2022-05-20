import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class NGCF(nn.Module):
    def __init__(self, norm_adj, hyper_parameter_dict, args, user_le, item_le, train_df, device):
        super(NGCF, self).__init__()
        self.args = args

        self.n_user = len(user_le.classes_)
        self.n_item = len(item_le.classes_)
        self.device = device
        self.train_df = train_df
        self.user_le = user_le
        self.item_le = item_le

        self.emb_size = hyper_parameter_dict['model']['embed_size']
        self.node_dropout = hyper_parameter_dict['model']['node_drop']
        self.mess_dropout = args['model']['mess_dropout']

        self.norm_adj = norm_adj

        self.layers = args['model']['layer_size']

        self.embedding_dict, self.weight_dict = self.init_weight()

        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(
            self.norm_adj).to(self.device)

    def init_weight(self):
        initializer = nn.init.xavier_uniform_

        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_user, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_item, self.emb_size)))
        })

        weight_dict = nn.ParameterDict()
        layers = [self.emb_size] + self.layers

        for k in range(len(self.layers)):
            weight_dict.update({'W_gc_%d' % k: nn.Parameter(
                initializer(torch.empty(layers[k], layers[k+1])))})
            weight_dict.update({'b_gc_%d' % k: nn.Parameter(
                initializer(torch.empty(1, layers[k+1])))})
            weight_dict.update({'W_bi_%d' % k: nn.Parameter(
                initializer(torch.empty(layers[k], layers[k+1])))})
            weight_dict.update({'b_bi_%d' % k: nn.Parameter(
                initializer(torch.empty(1, layers[k+1])))})

        return embedding_dict, weight_dict

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor(np.array([coo.row, coo.col]))
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, users, items, drop_flag=True):
        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    self.node_dropout,
                                    self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj

        ego_embeddings = torch.cat([self.embedding_dict['user_emb'],
                                    self.embedding_dict['item_emb']], 0)

        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)

            sum_embeddings = torch.matmul(
                side_embeddings, self.weight_dict['W_gc_%d' % k]) + self.weight_dict['b_gc_%d' % k]

            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = torch.matmul(
                bi_embeddings, self.weight_dict['W_bi_%d' % k]) + self.weight_dict['b_bi_%d' % k]

            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(
                sum_embeddings + bi_embeddings)
            ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)

            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, 1)
        u_g_embeddings = all_embeddings[:self.n_user, :]
        i_g_embeddings = all_embeddings[self.n_user:, :]

        self.u_g_embeddings = u_g_embeddings
        self.i_g_embeddings = i_g_embeddings

        u_g_embeddings = u_g_embeddings[users, :]
        i_g_embeddings = i_g_embeddings[items, :]

        return u_g_embeddings, i_g_embeddings

    def predict(self, u_g_embeddings, i_g_embeddings):
        return torch.sum(u_g_embeddings * i_g_embeddings, dim=1)

    def loss(self, u_g_embeddings, i_g_embeddings, labels):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(self.predict(
            u_g_embeddings, i_g_embeddings), labels))
        return loss

    def recommendation(self, before_labelencoder_user, K=10, ignore_before=True):
        user_tr = self.user_le.transform([before_labelencoder_user])[0]

        u_g_embedding = self.u_g_embeddings[user_tr, :]
        total = torch.sum(u_g_embedding * self.i_g_embeddings, axis=1)

        trans = np.arange(self.n_item)
        argsort = total.argsort(descending=True)
        recommendation_item = self.item_le.inverse_transform(trans[argsort])

        list_item = self.train_df[self.train_df[self.args['data']['columns']
                                                [0]] == user_tr][self.args['data']['columns'][1]].unique()

        if len(list_item) == self.train_df[self.args['data']['columns'][1]].nunique():
            ignore_before == False

        if ignore_before == True:
            recommendation_item = list(
                filter(lambda x: (x not in list_item) == True, recommendation_item))[:K]
        else:
            recommendation_item = recommendation_item[:K]

        return recommendation_item
