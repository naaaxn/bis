import os
import socket
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

import pandas as pd
import numpy as np
import argparse
import random
from kgcn_model import KGCN
from data_loader_yxb import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
import time

import os
import socket
import torch.distributed as dist
import dgl
import scipy.sparse
from dgl import DGLGraph

from PaGraph.model.gcn_nssc import GCNSampling
import PaGraph.data as data
import PaGraph.storage as storage
from PaGraph.parallel import SampleLoader

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = "29592" #要指定端口，动态生成会导致init_proess_group卡死
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    torch.manual_seed(rank)
    # print('rank [{}] process successfully launches'.format(rank))

def reduce_value(value, average=True):
    world_size = dist.get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value
 
    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size
 
        return value 

def cleanup():
    dist.destroy_process_group()

def construct_kg(df_kg):
    '''
    Construct knowledge graph
    Knowledge graph is dictionary form
    'head': [(relation, tail), ...]
    '''
    print('Construct knowledge graph ...', end=' ')
    kg = dict()
    for i in range(len(df_kg)):
        head = df_kg.iloc[i]['head']
        relation = df_kg.iloc[i]['relation']
        tail = df_kg.iloc[i]['tail']
        if head in kg:
            kg[head].append((relation, tail))
        else:
            kg[head] = [(relation, tail)]
        if tail in kg:
            kg[tail].append((relation, head))
        else:
            kg[tail] = [(relation, head)]
    print('Done')
    return kg

def demo_basic(rank, world_size, args):

    # print(f"Running basic DDP example on rank {rank}.")
    
    ddp_setup(rank, world_size)
    
    #读入该分区的训练集
    dataname = os.path.join(args.sub_graph_dataset, '{}naive'.format(world_size))   
    sub_train_file = os.path.join(dataname, 'sub_user_{}.csv'.format(rank))
    df_dataset = pd.read_csv(sub_train_file, sep='\t')
    # print("哇哇哇哇哇哇哇哇哇哇哇哇哇哇哇：",df_dataset.size)

    #读入该分区的kg（KGCN可处理的格式）
    dataname = os.path.join(args.sub_graph_dataset, '{}naive'.format(world_size))
    sub_train_file = os.path.join(dataname, 'sub_kg_{}.csv'.format(rank))
    df_kg = pd.read_csv(sub_train_file, sep='\t')
    kg = construct_kg(df_kg)
    # print(  "kg:", kg)

    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(df_dataset, df_dataset['label'], test_size=1 - args.ratio, shuffle=False, random_state=999)
    train_dataset = KGCNDataset(x_train)
    test_dataset = KGCNDataset(x_test)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)  
    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    # # print("train_sampler:",train_sampler)
    # assert args.batch_size % world_size == 0
    # batch_size_per_GPU = args.batch_size // world_size
    # # 这里的batch_size指的是每个进程下的batch_size。也就是说，总batch_size是这里的batch_size再乘以并行数(world_size)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_per_GPU, sampler=train_sampler)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_per_GPU, sampler=test_sampler)

    
    #写死了，先跑通，这里需要多个分区的模型参数大小相同，暂时不知道怎么同步，取多个分区最大值
    # num_relation = df_kg['relation'].nunique()
    # num_user = df_dataset['userID'].nunique()
    # num_entity = (pd.concat([df_kg['head'], df_kg['tail'], df_dataset['itemID']])).nunique()
    # print("数量：", num_relation, num_user, num_entity)
    num_user = 1872
    num_entity = 9366
    num_relation = 60

    # create model and move it to GPU with id rank
    device = torch.device(f"cuda:{rank}")
    kgcn = KGCN(num_user, num_entity, num_relation, kg, args, device).to(device)
    net = DDP(kgcn, device_ids=[rank], output_device=rank)
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.l2_weight)
    # print('device: ', device)

    loss_list = []
    test_loss_list = []
    auc_score_list = []


    torch.cuda.synchronize()  # 等gpu操作完
    start = time.time()
    for epoch in range(args.n_epochs):
        running_loss = 0.0
        step=0
        # train_sampler.set_epoch(args.n_epochs)
        for i, (user_ids, item_ids, labels) in enumerate(train_loader):
            user_ids, item_ids, labels = user_ids.to(rank), item_ids.to(rank), labels.to(rank)
            optimizer.zero_grad() 
            outputs = net(user_ids, item_ids)    
            outputs = outputs.reshape(-1,1) 
            loss = criterion(outputs, labels)
            loss.backward()            
            optimizer.step() 
            running_loss += loss.item() 
            
            step += 1
            if rank == 0 and step % 400 == 0:
                print('epoch [{}]. TrainLoss: '.format(epoch + 1), running_loss / len(train_loader))
        
        # print(f"rank {rank} : ")
        # print('[Epoch {}]train_loss: '.format(epoch+1), running_loss / len(train_loader))
        loss_list.append(running_loss / len(train_loader))
            
        # evaluate per every epoch
        with torch.no_grad():
            test_loss = 0
            total_roc = 0
            step=0
            # train_sampler.set_epoch(args.n_epochs)
            for user_ids, item_ids, labels in test_loader:
                user_ids, item_ids, labels = user_ids.to(rank), item_ids.to(rank), labels.to(rank)
                outputs = net(user_ids, item_ids)
                outputs = outputs.reshape(-1,1)
                test_loss += criterion(outputs, labels).item()
                total_roc += roc_auc_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
                
                step += 1
                if rank == 0 and step % 100 == 0:
                    print('epoch [{}]. TestLoss: '.format(epoch + 1), test_loss / len(test_loader))
        
            # print(f"rank {rank} : ")
            # print('[Epoch {}]test_loss: '.format(epoch+1), test_loss / len(test_loader))
            test_loss_list.append(test_loss / len(test_loader))
            auc_score_list.append(total_roc / len(test_loader))

    # ########################################## end #########################################
    torch.cuda.synchronize()  # 等gpu操作完
    end = time.time()
    print('Total Time: {:.4f}s'.format(end - start))
    dist.barrier()
    dist.destroy_process_group()


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
        # print('1', user_id, 'a', item_id, 'b',label, 'c')
        user_id = np.reshape(user_id,(1,)) # 从一维数组变为一维矩阵，可能是满足模型输入要求
        item_id = np.reshape(item_id,(1,))
        label = np.reshape(label,(1,))
        # print('2', user_id, 'a', item_id.shape, 'b',label.shape, 'c')
        return user_id, item_id, label   #(array([1409]), array([3027]), array([1.], dtype=float32))

if __name__ == "__main__":

    parser = argparse.ArgumentParser() 
    parser.add_argument('--sub_graph_dataset', type=str, default='PaGraph/data/wc_data_pagraph', help='which sub_graph_dataset to use')
    parser.add_argument("--gpu", type=str, default='cpu',
                      help="gpu ids. such as 0 or 0,1,2")
    parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
    parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
    parser.add_argument('--n_epochs', type=int, default=20, help='the number of epochs')
    parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
    parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
    parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size') # ？
    parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization') # ？
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate') # ?
    parser.add_argument('--ratio', type=float, default=0.8, help='size of training dataset') # ?

    args = parser.parse_args(['--l2_weight', '1e-4']) 


    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    # print('world_size: ', world_size)


    mp.spawn(demo_basic, args=(world_size, args), nprocs=world_size, daemon=True)