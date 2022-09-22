from __future__ import division

import random
import sys
import os
import networkx
from torch.utils.data import Dataset
import numpy as np
import sys
import torch
from sklearn import preprocessing
from init_net import init
import powerlaw
import networkx as nx


class DataHelper(Dataset):
    def __init__(self, file_path, ratio, neg_size, hist_len, directed=False, transform=None,
                 tlp_flag=False):
        self.node2hist = dict()  # 节点历史记录
        self.neg_size = neg_size  # 负采样大小
        self.hist_len = hist_len  # 历史长度
        self.directed = directed  # 是否为有向图
        self.transform = transform  # ？

        # self.max_d_time = -sys.maxint  # Time interval [0, T]
        self.max_d_time = -sys.maxsize  # 1.0
        self.all_edge_index = []  # 所有连边
        self.node_time_nodes = dict()  # 节点交互历史
        self.node_set = set()  # 所有节点集合
        self.degrees = dict()  # 节点的度值
        self.edge_list = []  # 连边列表
        self.node_rate = {}  # 节点速率
        self.edge_rate = {}  # 边速率
        self.node_sum = {}  # 节点计数
        self.edge_sum = {}  # 连边计数
        self.time_stamp = []  # 时间戳列表
        self.time_edges_dict = {}  # 时序边字典
        self.time_nodes_dict = {}  # 时序节点字典
        print('loading data...')  # 开始加载数据

        g = init(file_path)  # 存放连边信息的时序图
        dif_list = []  #
        for node in g:
            for nbr in g[node]:
                time_stamp = g[node][nbr]["time_stamp"]
                dif_list = np.hstack((dif_list, np.diff(time_stamp)))
        sys.stdout = open(os.devnull, 'w')
        results = powerlaw.Fit(dif_list, verbose=False)  # 屏蔽一下输出
        sys.stdout = sys.__stdout__
        b = results.alpha - 1
        aim = results.xmin ** b / (1 - ratio)
        hot_span = aim ** (1 / b)
        data = np.loadtxt(fname=file_path, skiprows=0)[:, [0, 1, 2]]  # 时序网络原始文件（i,j,t）
        nodes = np.unique(np.vstack((data[:, 0], data[:, 1])))  # 读取所有的节点
        nodes_dict = {}  # relabel  # 将节点编号归一到0到max
        for i in range(len(nodes)):
            nodes_dict[nodes[i]] = i
        g = networkx.relabel_nodes(g, nodes_dict)  # 对节点重现编号使其一致
        self.node_dict = nodes_dict  # 保存节点字典
        self.g = g
        begin = data[:, 2][0]
        span = data[:, 2][-1] - data[:, 2][0]
        self.half_life = hot_span
        print('half life:', self.half_life)
        data[:, 2] = data[:, 2] - begin
        data[:, 2] = data[:, 2] / span  # 将时间统一归一化到1
        infile = torch.tensor(data)  # 将处理好的数据转化了tensor
        for i in range(infile.size(0)):  # 遍历每个event生成data
            s_node = int(infile[i][0].item())  # source node    # 源节点
            t_node = int(infile[i][1].item())  # target node    # 目标节点
            s_node = nodes_dict[s_node]  # 替换为顺序编号
            t_node = nodes_dict[t_node]
            d_time = float(infile[i][2].item())  # time slot, delta t

            self.node_set.update([s_node, t_node])  # 更新节点集合

            if s_node not in self.degrees:  # 节点度
                self.degrees[s_node] = 0
            if t_node not in self.degrees:
                self.degrees[t_node] = 0

            self.all_edge_index.append([s_node, t_node])  # 静态连边

            if s_node not in self.node2hist:  # node2hist  {node: [(historical neighbor, time)], ……}    # 节点的历史交互数据
                self.node2hist[s_node] = list()
            if not directed:  # undirected
                if t_node not in self.node2hist:
                    self.node2hist[t_node] = list()

            if tlp_flag:
                if d_time >= 1.0:
                    continue
            # the new added node's degree is 0

            self.edge_list.append((s_node, t_node, d_time))  # edge list
            if not directed:
                self.edge_list.append((t_node, s_node, d_time))

            self.node2hist[s_node].append((t_node, d_time))
            if not directed:
                self.node2hist[t_node].append((s_node, d_time))  # because undirected, so add the inverse version

            if s_node not in self.node_time_nodes:
                self.node_time_nodes[s_node] = dict()  # for the new added s_node, create a dict for it
            if d_time not in self.node_time_nodes[s_node]:
                self.node_time_nodes[s_node][d_time] = list()  # for the new time,
            self.node_time_nodes[s_node][d_time].append(t_node)
            if not directed:  # undirected
                if t_node not in self.node_time_nodes:
                    self.node_time_nodes[t_node] = dict()
                if d_time not in self.node_time_nodes[t_node]:
                    self.node_time_nodes[t_node][d_time] = list()
                self.node_time_nodes[t_node][d_time].append(s_node)

            if d_time > self.max_d_time:  # 实时更新最大时间戳
                self.max_d_time = d_time  # record the max time

            self.degrees[s_node] += 1  # node degree
            self.degrees[t_node] += 1

            self.time_stamp.append(d_time)
            if not self.time_edges_dict.__contains__(d_time):
                self.time_edges_dict[d_time] = []
            self.time_edges_dict[d_time].append((s_node, t_node))
            if not self.time_nodes_dict.__contains__(d_time):
                self.time_nodes_dict[d_time] = []
            self.time_nodes_dict[d_time].append(s_node)
            self.time_nodes_dict[d_time].append(t_node)
        # 手动添加随机特征
        ramdom_feacture = np.empty((len(self.node_set) + 1, 128))
        for i in range(len(self.node_set)):
            vector = (torch.rand(128) - 0.5) * 2  # (-1,1)
            ramdom_feacture[i, :] = vector
        ramdom_feacture[len(nodes_dict), :] = torch.zeros(128)  # 匿名节点
        node_feature = torch.tensor(ramdom_feacture)

        # hks_para = pd.read_csv(node_feature_path, index_col=[0])
        # replace_para = hks_para.median(axis=0) / 2
        # hks_feature = np.empty((len(self.node_set)+1, 3))
        # for key in nodes_dict.keys():
        #     vector = hks_para.loc[key][:]
        #     if not vector[0] > 0:
        #         vector = replace_para
        #     loc = nodes_dict[key]
        #     hks_feature[loc, :] = vector
        # hks_feature[len(nodes_dict), :] = torch.zeros(3)
        # hks_feature = torch.tensor(hks_feature)
        # self.node_features = hks_feature
        self.node_features = node_feature
        self.node_features = preprocessing.StandardScaler().fit_transform(self.node_features)

        # print("degree_features", degree_features[0:5])
        self.node_list = sorted(list(self.node_set))
        self.time_stamp = sorted(list(set(self.time_stamp)))  # !!! time from 0 to 1
        # print('time minimum:', min(self.time_stamp))
        # print('time maxmum:', max(self.time_stamp))

        self.node_dim = len(self.node_set)  # 节点的数量
        self.data_size = 0
        for s in self.node2hist:
            hist = self.node2hist[s]
            hist = sorted(hist,
                          key=lambda x: x[1])  # from past(0) to now(1). This supports the events ranked in time order.
            self.node2hist[s] = hist
            self.data_size += len(self.node2hist[s])

        self.max_nei_len = max(map(lambda x: len(x), self.node2hist.values()))  #
        self.idx2source_id = np.zeros((self.data_size,), dtype=np.int32)
        self.idx2target_id = np.zeros((self.data_size,), dtype=np.int32)
        idx = 0
        for s_node in self.node2hist:
            for t_idx in range(len(self.node2hist[s_node])):
                if self.half_life < self.node2hist[s_node][t_idx][1] :
                    self.idx2source_id[idx] = s_node
                    self.idx2target_id[idx] = t_idx
                    idx += 1

    def get_node_dim(self):
        return self.node_dim

    def get_max_d_time(self):
        return self.max_d_time

    def binary_search(self, node_his, time):  # 返回最小的
        low = 0
        mid = 0
        hi = len(node_his) - 1

        if node_his[-1][1] < time:
            return -1
        # 利用二分查找，找到第一个大于等于begin的位置
        while low < hi:
            mid = (low + hi) // 2
            if node_his[mid][1] >= time:
                hi = mid
            else:
                low = mid + 1
        return low

    def get_time_nbrs(self, node, e_time):  # 时序最短节点
        if node == len(self.node_dict):  # 匿名节点
            his_node = np.array(len(self.node_dict)).repeat(self.hist_len)
            his_node = np.array(his_node, dtype=int)
            his_weight = np.array(0).repeat(self.hist_len)
            his_weight = np.array(his_weight, dtype=float)
            his_time = np.array(e_time).repeat(self.hist_len)
            his_time = np.array(his_time, dtype=float)
            return his_node, his_weight, his_time
        else:
            high = self.binary_search(self.node2hist[node], e_time)
            low = self.binary_search(self.node2hist[node], e_time - 2*self.half_life)  # 给定相同长度是历史窗口，这里我们设置较大的一个范围，如果不设置，其实历史大小是变化的，引入了噪声项
            node_event_dict = {}
            node_event_time = {}
            for his in self.node2hist[node][low:high]:
                s_node = his[0]
                s_time = his[1]
                if s_node not in node_event_dict.keys():
                    node_event_dict[s_node] = np.exp(s_time - e_time)
                    node_event_time[s_node] = e_time
                else:
                    node_event_dict[s_node] += np.exp(s_time - e_time)
                    node_event_time[s_node] = e_time
            times = []
            node_event_dict = np.array(sorted(node_event_dict.items(), key=lambda x: x[1], reverse=True))  # 选取权重最大的top
            # np.random.shuffle(node_event_dict)
            if len(node_event_dict) > 0:
                his_node = np.array(node_event_dict[:, 0], dtype=int)
                his_weight = np.array(node_event_dict[:, 1], dtype=float)  # 相对权重
                for key in his_node:
                    times.append(node_event_time[key])
                times = np.array(times, dtype=float)
                if len(his_node) >= self.hist_len:
                    return his_node[:self.hist_len], his_weight[:self.hist_len], times[:self.hist_len]
                else:
                    add = self.hist_len - len(his_node)
                    return np.hstack((his_node, np.array(len(self.node_dict)).repeat(add))), np.hstack(
                        (his_weight, np.array(0).repeat(add))), np.hstack(
                        (times, np.array(e_time).repeat(add)))
            else:
                his_node = np.array(len(self.node_dict)).repeat(self.hist_len)
                his_node = np.array(his_node, dtype=int)
                his_weight = np.array(0).repeat(self.hist_len)
                his_weight = np.array(his_weight, dtype=float)
                his_time = np.array(e_time).repeat(self.hist_len)
                his_time = np.array(his_time, dtype=float)
                return his_node, his_weight, his_time

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        # sampling via htne
        s_node = self.idx2source_id[idx]
        t_idx = self.idx2target_id[idx]
        neg_node = self.negative_sampling(s_node).astype(int)[0]
        t_node = self.node2hist[s_node][t_idx][0]
        e_time = self.node2hist[s_node][t_idx][1]
        begin_time_one = e_time - self.half_life
        if begin_time_one < 0:
            begin_time_one = 0
        s_his_nodes, s_his_weight, s_his_times = self.get_time_nbrs(s_node, e_time)  # 一阶时序邻居
        t_his_nodes, t_his_weight, t_his_times = self.get_time_nbrs(t_node, e_time)  #
        neg_his_nodes, neg_his_weight, neg_his_times = self.get_time_nbrs(neg_node, e_time)
        s_his_his_nodes_list = []
        s_his_his_weight_list = []
        t_his_his_nodes_list = []
        t_his_his_weight_list = []
        neg_his_his_nodes_list = []
        neg_his_his_weight_list = []
        for i in range(self.hist_len):
            n1 = s_his_nodes[i]
            t1 = s_his_times[i]
            n2 = t_his_nodes[i]
            t2 = t_his_times[i]
            n3 = neg_his_nodes[i]
            t3 = neg_his_times[i]
            s_his_his_nodes, s_his_his_weight, s_his_his_times = self.get_time_nbrs(n1, t1)  # 2德尔塔历史信息
            t_his_his_nodes, t_his_his_weight, t_his_his_times = self.get_time_nbrs(n2, t2)
            neg_his_his_nodes, neg_his_his_weight, neg_his_his_times = self.get_time_nbrs(n3, t3)
            s_his_his_nodes_list.append(s_his_his_nodes)
            s_his_his_weight_list.append(s_his_his_weight)
            t_his_his_nodes_list.append(t_his_his_nodes)
            t_his_his_weight_list.append(t_his_his_weight)
            neg_his_his_nodes_list.append(neg_his_his_nodes)
            neg_his_his_weight_list.append(neg_his_his_weight)

        s_his_nodes = np.array(s_his_nodes).astype(int)
        t_his_nodes = np.array(t_his_nodes).astype(int)
        neg_his_nodes = np.array(neg_his_nodes).astype(int)
        s_his_his_nodes_list = np.array(s_his_his_nodes_list).astype(int)
        t_his_his_nodes_list = np.array(t_his_his_nodes_list).astype(int)
        neg_his_his_nodes_list = np.array(neg_his_his_nodes_list).astype(int)

        s_his_times = np.array(s_his_times)
        t_his_times = np.array(t_his_times)
        neg_his_times = np.array(neg_his_times)
        s_his_his_times_list = np.array(s_his_his_weight_list)
        t_his_his_times_list = np.array(t_his_his_weight_list)
        neg_his_his_times_list = np.array(neg_his_his_weight_list)

        s_self_feat = self.node_features[s_node]  # 自身属性
        s_one_hop_feat = self.node_features[s_his_nodes]  # 一阶时序节点
        s_two_hop_feat = []
        for i in range(self.hist_len):
            s_two_feat = self.node_features[s_his_his_nodes_list[i]]
            s_two_hop_feat.append(s_two_feat)
        s_two_hop_feat = np.array(s_two_hop_feat)

        t_self_feat = self.node_features[t_node]
        t_one_hop_feat = self.node_features[t_his_nodes]
        t_two_hop_feat = []
        for i in range(self.hist_len):
            t_two_feat = self.node_features[t_his_his_nodes_list[i]]
            t_two_hop_feat.append(t_two_feat)
        t_two_hop_feat = np.array(t_two_hop_feat)

        neg_self_feat = self.node_features[neg_node]
        neg_one_hop_feat = self.node_features[neg_his_nodes]
        neg_two_hop_feat = []
        for i in range(self.hist_len):
            neg_two_feat = self.node_features[neg_his_his_nodes_list[i]]
            neg_two_hop_feat.append(neg_two_feat)
        neg_two_hop_feat = np.array(neg_two_hop_feat)

        end_time = e_time + self.half_life
        s_edge_rate = 0  # 节点的速率
        for key in self.node_time_nodes[s_node].keys():
            if end_time >= key >= e_time:
                s_edge_rate += len(self.node_time_nodes[s_node][key])
        s_t_rate = 1  # 连边在半衰期内的速率
        st_time_stamp = self.g[s_node][t_node]['time_stamp']
        st_time_stamp = np.array(st_time_stamp)
        s_t_rate = len(np.where(e_time <= st_time_stamp)[0]) + len(np.where(end_time > st_time_stamp)[0])
        s_t_rate = (s_t_rate - len(st_time_stamp))
        s_n_rate = -1  # 负连边在半衰期内出现的速率

        sample = {
            # 's_node': s_node,  # e.g., 5424
            # 't_node': t_node,  # e.g., 5427
            'event_time': e_time,

            's_history_times': s_his_times,
            't_history_times': t_his_times,
            's_his_his_times_list': s_his_his_times_list,
            't_his_his_nodes_list': t_his_his_nodes_list,
            't_his_his_times_list': t_his_his_times_list,

            's_self_feat': s_self_feat,
            's_one_hop_feat': s_one_hop_feat,
            's_two_hop_feat': s_two_hop_feat,

            't_self_feat': t_self_feat,
            't_one_hop_feat': t_one_hop_feat,
            't_two_hop_feat': t_two_hop_feat,

            'neg_his_times_list': neg_his_times,
            'neg_his_his_times_list': neg_his_his_times_list,

            'neg_self_feat': neg_self_feat,
            'neg_one_hop_feat': neg_one_hop_feat,
            'neg_two_hop_feat': neg_two_hop_feat,

            's_edge_rate': s_edge_rate,
            's_t_rate': s_t_rate,
            's_n_rate': s_n_rate,

        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def negative_sampling(self, node):
        neg = np.random.choice(list(set(self.g.nodes) - set(self.g[node])), 1)
        return neg

    def getitem(self, node, time):
        s_node = node
        # e_time = self.node2hist[node][-1][1]
        e_time = time
        begin_time_one = e_time - self.half_life
        if begin_time_one < 0:
            begin_time_one = 0
        s_his_nodes, s_his_weight, s_his_times = self.get_time_nbrs(s_node, e_time)  # 一阶时序邻居
        s_his_his_nodes_list = []
        s_his_his_weight_list = []
        for i in range(self.hist_len):
            n1 = s_his_nodes[i]
            t1 = s_his_times[i]
            s_his_his_nodes, s_his_his_weight, s_his_his_times = self.get_time_nbrs(n1, t1)  # 2德尔塔历史信息
            s_his_his_nodes_list.append(s_his_his_nodes)
            s_his_his_weight_list.append(s_his_his_weight)

        s_his_nodes = np.array(s_his_nodes).astype(int)
        s_his_his_nodes_list = np.array(s_his_his_nodes_list).astype(int)
        s_his_times = np.array(s_his_times)
        s_his_his_times_list = np.array(s_his_his_weight_list)
        s_self_feat = self.node_features[s_node]  # 自身属性
        s_one_hop_feat = self.node_features[s_his_nodes]  # 一阶时序节点
        s_two_hop_feat = []
        for i in range(self.hist_len):
            s_two_feat = self.node_features[s_his_his_nodes_list[i]]
            s_two_hop_feat.append(s_two_feat)
        s_two_hop_feat = np.array(s_two_hop_feat)

        sample = {
            # 's_node': s_node,  # e.g., 5424
            # 't_node': t_node,  # e.g., 5427
            'event_time': e_time,
            's_history_times': s_his_times,
            's_his_his_times_list': s_his_his_times_list,
            's_self_feat': s_self_feat,
            's_one_hop_feat': s_one_hop_feat,
            's_two_hop_feat': s_two_hop_feat,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
