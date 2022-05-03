import torch
import torch.nn as nn
import torch.nn.functional as F


class NGCF(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, hyper_parameter_dict, args):
        super(NGCF, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.device = args.device

        self.emb_size = hyper_parameter_dict['embed_size']
        self.node_dropout = hyper_parameter_dict['node_dropout']
        self.mess_dropout = args.mess_dropout

        self.norm_adj = norm_adj

        self.layers = args.layer_size
        # self.decay = args.regs

        self.embedding_dict, self.weight_dict = self.init_weight()

        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(
            self.norm_adj).to(self.device)

    def init_weight(self):
        initializer = nn.init.xavier_uniform_

        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_user, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_itme, self.emb_size)))
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
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FlooatTensor(i, v, coo.shape)

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
