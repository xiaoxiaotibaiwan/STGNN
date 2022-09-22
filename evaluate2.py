# 使用采样的方法来计算AUC、 precision,针对网络数据集过大的情况 the number of nodes > 1000
import numpy as np
import pandas as pd
import networkx as nx
from init_net import init
from sklearn.metrics import roc_auc_score, average_precision_score


class Evaluation:
    def __init__(self, test_file, nodes_dict):
        self.train_nodes = nodes_dict.keys()
        fur_g = init(test_file)
        death_node = set(self.train_nodes) - set(fur_g.nodes)
        fur_g.add_nodes_from(list(death_node))
        self.fur_g = fur_g
        self.test_event = np.loadtxt(fname=test_file, skiprows=0, dtype=int)
        self.nodes_dict = nodes_dict

    def sample(self):  # 是否在T时刻后产生联系
        s = []
        t = []
        neg = []

        for edge in self.fur_g.edges:
            u, v = edge
            neg1 = np.random.choice(list(set(self.fur_g.nodes) - set(self.fur_g[u])), 1)[0]
            neg2 = np.random.choice(list(set(self.fur_g.nodes) - set(self.fur_g[v])), 1)[0]
            s.append(self.nodes_dict[u])
            t.append(self.nodes_dict[v])
            neg.append(self.nodes_dict[neg1])
            s.append(self.nodes_dict[v])
            t.append(self.nodes_dict[u])
            neg.append(self.nodes_dict[neg2])
        return s, t, neg

    def contiousSamping(self):
        return
