import math

import torch
from torch import nn, optim
from dgnn2 import DGNN
from film import Scale_4, Shift_4
from Emlp import EMLP
from node_relu import Node_edge


class MYModel3(nn.Module):
    def __init__(self, args):
        super(MYModel3, self).__init__()
        self.args = args
        self.gnn = DGNN(args)
        self.EMLP = EMLP(args)
        self.optim = optim.Adam([{'params': self.gnn.parameters()},
                                 ], lr=args.lr)

    def forward(self, s_self_feat, s_one_hop_feat, s_two_hop_feat,
                t_self_feat, t_one_hop_feat, t_two_hop_feat,
                neg_self_feat, neg_one_hop_feat, neg_two_hop_feat,
                s_his_time, s_his_his_time,
                t_his_time, t_his_his_time,
                neg_his_time, neg_his_his_time,
                s_edge_rate, s_t_rate, s_n_rate,
                training=True):
        s_emb = self.gnn(s_self_feat, s_one_hop_feat, s_two_hop_feat, s_his_time, s_his_his_time)  # 源节点节点的emb
        t_emb = self.gnn(t_self_feat, t_one_hop_feat, t_two_hop_feat, t_his_time, t_his_his_time)  # 目标节点的emb
        neg_embs = self.gnn(neg_self_feat, neg_one_hop_feat, neg_two_hop_feat, neg_his_time, neg_his_his_time) # 负采样
        emb_loss = nn.CosineEmbeddingLoss()
        pos = 1 - torch.cosine_similarity(s_emb, t_emb)
        pos = pos * s_t_rate
        L = torch.mean(pos) + emb_loss(s_emb, neg_embs, s_n_rate) * torch.mean(pos)
        # L = emb_loss(s_emb, neg_embs, s_n_rate) +emb_loss(s_emb, t_emb, torch.ones(len(s_n_rate)))
        # 反向传播
        if training:
            self.optim.zero_grad()
            L.backward()
            self.optim.step()

        return round((L.detach().clone()).cpu().item(),
                     4), s_emb.detach().clone().cpu(), t_emb.detach().clone().cpu(), s_emb.detach().clone().cpu(), neg_embs.detach().clone().cpu()
