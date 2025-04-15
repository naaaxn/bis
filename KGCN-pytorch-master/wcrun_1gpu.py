# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scikit-learn
import pandas as pd
import numpy as np
import argparse
import random
# from model import KGCN
from data_loader import DataLoader
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os
from torch.distributed import init_process_group, destroy_process_group, barrier
from utils_lcy import *
import gc
import dgl
from KGCN_train_wc import KGCN
import torch.multiprocessing as mp
import time

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # 初始化进程组, rank 表示当前进程的唯一标识符，而 world_size 表示参与分布式训练的总进程数
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


# Dataset class
class KGCNDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        user_id = np.array(self.df.iloc[idx]['userID'])
        item_id = np.array(self.df.iloc[idx]['itemID'])
        label = np.array(self.df.iloc[idx]['label'], dtype=np.float32)
        # print(user_id, 'a', item_id, 'b',label, 'c')
        user_id = np.reshape(user_id,(1,))
        item_id = np.reshape(item_id,(1,))
        label = np.reshape(label,(1,))
        # print(user_id.shape, 'a', item_id.shape, 'b',label.shape, 'c')
        return user_id, item_id, label

def train(rank: int,      #  当前进程的编号
         world_size: int,    # 总的进程数量
         config: RunConfig,
         data_loader: DataLoader,
         train_loader: KGCNDataset,
         test_loader: KGCNDataset,
         kg,
         args,):
    print("开始了！")
    ddp_setup(rank, world_size)
    print("跑通了！")
    # kg = kg.to(rank)   # 将数据，如知识图谱等移动到对应的GPU上
    config.rank = rank
    config.world_size = world_size

    # ######################################### 训练 #########################################
    # prepare network, loss function, optimizer
    num_user, num_entity, num_relation = data_loader.get_num()
    user_encoder, entity_encoder, relation_encoder = data_loader.get_encoders()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    net = KGCN(num_user, num_entity, num_relation, kg, args, device).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.l2_weight)
    # print('device: ', device)

    # train
    loss_list = []
    test_loss_list = []
    auc_score_list = []
    for epoch in range(args.n_epochs):
        running_loss = 0.0
        #print(train_loader)
        for i, (user_ids, item_ids, labels) in enumerate(train_loader):
            # print(user_ids," ",item_ids," ",labels)
            user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = net(user_ids, item_ids).to(device)    #传入的是一个批次的数据量，是batch*1维张量
            
            outputs = outputs.reshape(-1,1)
            # print(outputs, labels)
            loss = criterion(outputs, labels)
            torch.cuda.synchronize()  # 自己写的？同步？
            loss.backward()
            
            optimizer.step()

            running_loss += loss.item()
        
        # print train loss per every epoch
        print('[Epoch {}]train_loss: '.format(epoch+1), running_loss / len(train_loader))
        loss_list.append(running_loss / len(train_loader))
            
        # evaluate per every epoch
        with torch.no_grad():
            test_loss = 0
            total_roc = 0
            for user_ids, item_ids, labels in test_loader:
                user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
                outputs = net(user_ids, item_ids)
                outputs = outputs.reshape(-1,1)
                test_loss += criterion(outputs, labels).item()
                total_roc += roc_auc_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
            print('[Epoch {}]test_loss: '.format(epoch+1), test_loss / len(test_loader))
            test_loss_list.append(test_loss / len(test_loader))
            auc_score_list.append(total_roc / len(test_loader))

    # trainer = KGCNTrainer(config, net, train_loader, test_loader, loc_feat, optimizer, nid_dtype=torch.int32)
    # trainer.train()
    # ########################################## end #########################################
    destroy_process_group()

if __name__=="__main__":
    # prepare arguments (hyperparameters)
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
    parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
    parser.add_argument('--n_epochs', type=int, default=20, help='the number of epochs')
    parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
    parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
    parser.add_argument('--n_iter', type=int, default=2, help='number of iterations when computing entity representation')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--ratio', type=float, default=0.8, help='size of training dataset')
    parser.add_argument('--nprocs', default=4, type=int, help='Number of GPUs / processes')
    
    args = parser.parse_args(['--l2_weight', '1e-4'])
    
    config = RunConfig()
    rank = 0
    world_size = min(args.nprocs, torch.cuda.device_count())
    print(f"using {world_size} GPUs")

    # 做一些运行配置
    config.total_epoch = args.n_epochs

    # ######################################对数据进行处理#####################################
    # build dataset and knowledge graph
    # 其中kg三元组头尾都是编码ID，用户项目交互矩阵dataset的user和item都是编码ID，且三元组中的头ID与dataset中item ID对齐
    # 知识图谱三元组（item，关系，其他属性实体），用户项目交互矩阵（用户ID，itemID，label标签）
    data_loader = DataLoader(args.dataset)
    kg = data_loader.load_kg()
    df_dataset = data_loader.load_dataset()
    # print(kg)
    # print(data_loader.df_kg)
    # print(df_dataset)
    # print(df_dataset[df_dataset['userID']==1217])
    # print(df_dataset[df_dataset['itemID']==4])

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(df_dataset, df_dataset['label'], test_size=1 - args.ratio, shuffle=False, random_state=999)
    train_dataset = KGCNDataset(x_train)
    test_dataset = KGCNDataset(x_test)
    # print(train_dataset.__getitem__(10))
    # print(train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

    # prepare network, loss function, optimizer
    num_user, num_entity, num_relation = data_loader.get_num()
    user_encoder, entity_encoder, relation_encoder = data_loader.get_encoders()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = KGCN(num_user, num_entity, num_relation, kg, args, device).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.l2_weight)
    # print('device: ', device)

    # train
    loss_list = []
    test_loss_list = []
    auc_score_list = []

    
    torch.cuda.synchronize()  # 等gpu操作完
    start = time.time()
    for epoch in range(args.n_epochs):
        # torch.cuda.synchronize()  # 等gpu操作完
        # start1 = time.time()
        running_loss = 0.0
        #print(train_loader)
        for i, (user_ids, item_ids, labels) in enumerate(train_loader):
            # print(user_ids," ",item_ids," ",labels)
            user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(user_ids, item_ids)    #传入的是一个批次的数据量，是batch*1维张量
            #print(outputs)
            #print(labels)
            outputs = outputs.reshape(-1,1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # print train loss per every epoch
        print('[Epoch {}]train_loss: '.format(epoch+1), running_loss / len(train_loader))
        loss_list.append(running_loss / len(train_loader))
        
        # torch.cuda.synchronize()  # 等gpu操作完
        # end1 = time.time()
        # print('[Epoch {}]Time: {:.4f}s'.format(epoch+1, end1 - start1))
            
        # evaluate per every epoch
        with torch.no_grad():
            test_loss = 0
            total_roc = 0
            for user_ids, item_ids, labels in test_loader:
                user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
                outputs = net(user_ids, item_ids)
                outputs = outputs.reshape(-1,1)
                test_loss += criterion(outputs, labels).item()
                total_roc += roc_auc_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
            print('[Epoch {}]test_loss: '.format(epoch+1), test_loss / len(test_loader))
            test_loss_list.append(test_loss / len(test_loader))
            auc_score_list.append(total_roc / len(test_loader))
    
    torch.cuda.synchronize()  # 等gpu操作完
    end = time.time()
    print('Total Time: {:.4f}s'.format(end - start))
    # ########################################## end ########################################

    # 单机多卡，spawn要放入 if __name__=="__main__": 中，不然会引发错误
    # mp.spawn(train, args=(world_size, config, ), nprocs=world_size, daemon=True) 
    # mp.spawn(train, args=(world_size, config, data_loader, train_loader, test_loader, kg, args, ), nprocs=world_size, daemon=True)           




