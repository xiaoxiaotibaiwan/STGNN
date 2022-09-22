import pandas as pd
import torch
import numpy as np
import random
import argparse
import threading
from data_dyn_cite3 import DataHelper
from torch.utils.data import DataLoader, RandomSampler
from evaluate2 import Evaluation
from my_model3 import MYModel3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FType = torch.FloatTensor
LType = torch.LongTensor


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_test_split(file_path, name, pos):
    data = np.loadtxt(fname=file_path, skiprows=0, dtype=int)  # 时序网络原始文件（i,j,t）
    total_time_interval = data[-1][2] - data[0][2]
    split_time = data[0][2] + pos * total_time_interval
    times = data[:, 2]
    train_his = data[np.argwhere(times <= split_time)[:, 0]]
    nodes = np.unique(np.vstack((train_his[:, 0], train_his[:, 1])))
    test_his_ori = data[np.argwhere(times > split_time)[:, 0]]
    removed_test = []
    for i in range(len(test_his_ori)):
        event = test_his_ori[i]
        u = event[0]
        v = event[1]
        if u in nodes and v in nodes:
            removed_test.append(event)
    train_his = np.array(train_his, dtype=int)
    removed_test = np.array(removed_test, dtype=int)
    train_path = f"train/train_{name}_{pos}.csv"
    test_path = f"test/test_{name}_{pos}.csv"
    np.savetxt(train_path, train_his, fmt="%d")
    np.savetxt(test_path, removed_test, fmt="%d")
    return train_path, test_path, nodes


def main(args):
    setup_seed(args.seed)
    print("Network name:" + args.net_name)
    train_path, test_path, nodes = train_test_split(args.file_path, args.net_name, args.train_rate)
    Data = DataHelper(train_path, args.ratio, args.neg_size, args.hist_len, args.directed,
                      tlp_flag=args.tlp_flag)
    Eva = Evaluation(test_path, Data.node_dict)
    model = MYModel3(args).to(device)
    model.train()
    for j in range(args.epoch_num):
        # r_sample = RandomSampler(Data)
        loader = DataLoader(Data, batch_size=args.batch_size, num_workers=1, shuffle=True)
        # 训练
        for i_batch, sample_batched in enumerate(loader):
            loss, _, _, _, _, = model.forward(
                sample_batched['s_self_feat'].type(FType).reshape(-1, args.feat_dim).to_sparse().to(device),
                sample_batched['s_one_hop_feat'].type(FType).reshape(-1, args.feat_dim).to_sparse().to(device),
                sample_batched['s_two_hop_feat'].type(FType).reshape(-1, args.feat_dim).to_sparse().to(device),
                sample_batched['t_self_feat'].type(FType).reshape(-1, args.feat_dim).to_sparse().to(device),
                sample_batched['t_one_hop_feat'].type(FType).reshape(-1, args.feat_dim).to_sparse().to(device),
                sample_batched['t_two_hop_feat'].type(FType).reshape(-1, args.feat_dim).to_sparse().to(device),
                sample_batched['neg_self_feat'].type(FType).reshape(-1, args.feat_dim).to_sparse().to(device),
                sample_batched['neg_one_hop_feat'].type(FType).reshape(-1, args.feat_dim).to_sparse().to(device),
                sample_batched['neg_two_hop_feat'].type(FType).reshape(-1, args.feat_dim).to_sparse().to(device),
                # sample_batched['event_time'].type(FType).to(device),
                sample_batched['s_history_times'].type(FType).to(device),
                sample_batched['s_his_his_times_list'].type(FType).to(device),
                sample_batched['t_history_times'].type(FType).to(device),
                sample_batched['t_his_his_times_list'].type(FType).to(device),
                sample_batched['neg_his_times_list'].type(FType).to(device),
                sample_batched['neg_his_his_times_list'].type(FType).to(device),
                sample_batched['s_edge_rate'].type(FType).to(device),
                sample_batched['s_t_rate'].type(FType).to(device),
                sample_batched['s_n_rate'].type(FType).to(device),
            )
            if j == 0:
                print('batch_{} sig_loss:'.format(i_batch), loss)
            if i_batch == 10:
                break
        print('ep_{}_event_loss:'.format(j + 1), loss)
        # 存储
        embs = {}
        for i, node in enumerate(Data.node_dict.keys()):
            his = Data.getitem(i, 1)
            s_self_feat = torch.tensor(his['s_self_feat'], dtype=torch.float)
            s_one_hop_feat = torch.tensor(his['s_one_hop_feat'], dtype=torch.float)
            s_two_hop_feat = torch.tensor(his['s_two_hop_feat'], dtype=torch.float)
            # event_time = torch.tensor(his['event_time'], dtype=torch.float)
            s_history_times = torch.tensor([his['s_history_times']], dtype=torch.float)
            s_his_his_times_list = torch.tensor([his['s_his_his_times_list']], dtype=torch.float)
            node_emb = model.gnn(
                s_self_feat.reshape(-1, args.feat_dim).to_sparse().to(device),
                s_one_hop_feat.reshape(-1, args.feat_dim).to_sparse().to(device),
                s_two_hop_feat.reshape(-1, args.feat_dim).to_sparse().to(device),
                s_history_times.to(device),
                s_his_his_times_list.to(device),
            )
            node_emb = node_emb.cpu().detach().numpy()[0]
            embs[node] = node_emb
        embs = pd.Series(embs)
        t1 = threading.Thread(name='t1', target=embs.to_pickle(
            f"result/{args.net_name}_{args.ratio}_{args.hist_len}-{j}.pkl"))  # 加快io
        t1.start()
        # embs.to_pickle(f"result/{args.net_name}_{args.ratio}_{args.hist_len}-{j}.pkl")
    # torch.save(model.state_dict(), '../res/cite/model.pkl')


if __name__ == '__main__':
    networks = ["radoslaw-email"]
    # ,"ia-contact",  "wiki-elec",  "radoslaw-email"
    for net in networks:
        parser = argparse.ArgumentParser()
        parser.add_argument('--file_path', type=str, default=f'original/{net}.csv')
        parser.add_argument('--train_rate', type=str, default=0.75)
        parser.add_argument('--net_name', type=str, default=net)
        parser.add_argument('--ratio', type=str, default=0.5)
        parser.add_argument('--neg_size', type=int, default=1)  # 1为正负平衡采样，不修改
        parser.add_argument('--hist_len', type=int, default=25)  # 历史记录的长度
        parser.add_argument('--directed', type=bool, default=False)  # 是否为有向图
        parser.add_argument('--epoch_num', type=int, default=10, help='epoch number')  # 训练
        parser.add_argument('--tlp_flag', type=bool, default=True)
        parser.add_argument('--batch_size', type=int, default=500)  # 批处理大小
        parser.add_argument('--lr', type=float, default=0.01)  # 学习率
        parser.add_argument('--hid_dim', type=int, default=16)  # 影藏层维度
        parser.add_argument('--feat_dim', type=int, default=128)  # 节点自身特征维度
        parser.add_argument('--out_dim', type=int, default=16)  # 最后的嵌入维度
        parser.add_argument('--seed', type=int, default=1)  # 随机种子
        parser.add_argument('--ncoef', type=float, default=0.01)  #
        parser.add_argument('--l2_reg', type=float, default=0.001)
        args = parser.parse_args()
        main(args)
