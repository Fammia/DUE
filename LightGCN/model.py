import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_


class LightGCN(nn.Module):
    def __init__(self, users_num, items_num, embedding_dim=64, layers=2, norm_adj=None):
        super(LightGCN, self).__init__()
        self.users_num = users_num
        self.items_num = items_num
        self.embedding_dim = embedding_dim
        self.layers = layers

        self.user_embedding = nn.Embedding(self.users_num, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.items_num, self.embedding_dim)
        self._init_weights()

        self.norm_adj = norm_adj

    def _init_weights(self):
        xavier_normal_(self.user_embedding.weight.data)
        xavier_normal_(self.item_embedding.weight.data)

    def set_norm_adj(self, norm_adj):
        self.norm_adj = norm_adj

    def get_all_embeddings(self):
        assert self.norm_adj is not None, "Normalized adjacency is not initialized."

        all_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embs = [all_emb]

        # LightGCN propagation: no feature transform and no activation.
        for _ in range(self.layers):
            all_emb = torch.sparse.mm(self.norm_adj, all_emb)
            embs.append(all_emb)

        all_emb = torch.stack(embs, dim=1).mean(dim=1)
        user_emb, item_emb = torch.split(all_emb, [self.users_num, self.items_num], dim=0)
        return user_emb, item_emb

    def getUsersRating(self, users):
        user_emb, item_emb = self.get_all_embeddings()
        users_emb = user_emb[users.long()]
        rating = torch.matmul(users_emb, item_emb.t())
        return rating

    def bpr_score(self, users, items):
        user_emb, item_emb = self.get_all_embeddings()
        users_emb = user_emb[users.long()]
        items_emb = item_emb[items.long()]
        return torch.sum(users_emb * items_emb, dim=1)

    def forward(self, users, items):
        return self.bpr_score(users, items)


def build_norm_adj(train_mat, users_num, items_num):
    if not sp.isspmatrix_coo(train_mat):
        train_mat = train_mat.tocoo()

    user_np = train_mat.row
    item_np = train_mat.col + users_num

    row = np.concatenate([user_np, item_np])
    col = np.concatenate([item_np, user_np])
    data = np.ones(len(row), dtype=np.float32)

    adj = sp.coo_matrix(
        (data, (row, col)),
        shape=(users_num + items_num, users_num + items_num),
        dtype=np.float32,
    )

    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(deg + 1e-7, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0

    norm_data = deg_inv_sqrt[row] * data * deg_inv_sqrt[col]
    norm_adj = sp.coo_matrix(
        (norm_data, (row, col)),
        shape=(users_num + items_num, users_num + items_num),
        dtype=np.float32,
    )

    indices = torch.LongTensor(np.vstack((norm_adj.row, norm_adj.col)))
    values = torch.FloatTensor(norm_adj.data)
    shape = torch.Size(norm_adj.shape)
    return torch.sparse.FloatTensor(indices, values, shape).coalesce()
