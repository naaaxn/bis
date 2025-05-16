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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
import time

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
    os.environ['MASTER_PORT'] = "29593" #瑕佹寚瀹氱鍙ｏ紝鍔ㄦ€佺敓鎴愪細瀵艰嚧init_proess_group鍗℃
    # 鍒濆鍖栬繘绋嬬粍
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    torch.manual_seed(rank)
    # print('rank [{}] process successfully launches'.format(rank))

def reduce_value(value, average=True):
    world_size = dist.get_world_size()
    if world_size < 2:  # 鍗旼PU鐨勬儏鍐�
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

def demo_basic(rank, world_size, usr_emb, rel_emb, subgraph_info, args):

    # print(f"Running basic DDP example on rank {rank}.")
    
    ddp_setup(rank, world_size)
    
    # 鍒涘缓缂撳瓨鍥�
    dataname = os.path.basename(args.sub_graph_dataset)
    remote_g = dgl.contrib.graph_store.create_graph_from_store(dataname, "shared_mem")
    
    #璇诲叆璇ュ垎鍖虹殑璁粌闆�
    dataname = os.path.join(args.sub_graph_dataset, args.data+'_{}naive'.format(world_size))   
    sub_train_file = os.path.join(dataname, 'sub_user_{}.csv'.format(rank))
    df_dataset = pd.read_csv(sub_train_file, sep='\t')

    #璇诲叆璇ュ垎鍖虹殑kg锛圞GCN鍙鐞嗙殑鏍煎紡锛�
    sub_train_file = os.path.join(dataname, 'sub_kg_{}.csv'.format(rank))
    df_kg = pd.read_csv(sub_train_file, sep='\t')
    # print("df_kg: ", df_kg)

    # 閲嶆柊鏄犲皠瀛愬浘, 鏋勯€犲垱寤篊OO鐭╅樀
    unique_nodes = pd.concat([df_kg['head'], df_kg['tail']]).drop_duplicates().reset_index(drop=True)
    num_nodes = len(unique_nodes)
    continuous_ids = np.arange(num_nodes)
    full_to_sub = dict(zip(unique_nodes, continuous_ids))
    sub_to_full = torch.LongTensor(unique_nodes)

    df_kg['head'] = df_kg['head'].map(full_to_sub)
    df_kg['tail'] = df_kg['tail'].map(full_to_sub)
    df_dataset['itemID'] = df_dataset['itemID'].map(full_to_sub)

    rows = df_kg['head'].values  
    cols = df_kg['tail'].values  
    data = df_kg['relation'].values
    src_node, dst_node = rows, cols  # 涓嶅啀杩涜瀵圭О鎵╁睍
    shape = (num_nodes, num_nodes)  
    sub_adj = scipy.sparse.coo_matrix((data, (src_node, dst_node)), shape=shape)
    
    embed_names = ['vfeatures', 'norm']
    cacher = storage.GraphCacheServer(remote_g, sub_adj.shape[0], sub_to_full, rank)
    cacher.init_field(embed_names)
    cacher.log = False

    # 鏁版嵁闆嗗垝鍒�
    if int(subgraph_info.iloc[0]["num_user1"]) < int(subgraph_info.iloc[0]["num_user2"]):
        if rank==0:
           sampled_rows = df_dataset.sample(n=int(subgraph_info.iloc[0]["num_user2"])-int(subgraph_info.iloc[0]["num_user1"]))
           df_dataset = pd.concat([df_dataset, sampled_rows], ignore_index=True) 
    elif int(subgraph_info.iloc[0]["num_user1"]) > int(subgraph_info.iloc[0]["num_user2"]):
        if rank==1:
           sampled_rows = df_dataset.sample(n=int(subgraph_info.iloc[0]["num_user1"])-int(subgraph_info.iloc[0]["num_user2"]))
           df_dataset = pd.concat([df_dataset, sampled_rows], ignore_index=True)  
      
    train_dataset, test_dataset = train_test_split(df_dataset, test_size=1 - args.ratio, shuffle=False, random_state=999)
    # print("train_dataset: ", train_dataset)

    kg = construct_kg(df_kg)
    g = DGLGraph(sub_adj, readonly=True)

    # 妯″瀷鍒涘缓
    device = torch.device(f"cuda:{rank}")

    kgcn = KGCN(usr_emb, rel_emb, kg, num_nodes, args, device).to(device)
    net = DDP(kgcn, device_ids=[rank], output_device=rank)

    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.l2_weight)

    # print("train_dataset['itemID'].values: ",len(train_dataset['itemID'].values))
    train_sampler = dgl.contrib.sampling.NeighborSampler(g, batch_size=32,
      expand_factor = 100, neighbor_type='in', shuffle=False, num_hops=args.n_iter, 
      seed_nodes=train_dataset['itemID'].values, prefetch=True
    )
    test_sampler = dgl.contrib.sampling.NeighborSampler(g, batch_size=32,
      expand_factor = 100, neighbor_type='in', shuffle=False, num_hops=args.n_iter, 
      seed_nodes=test_dataset['itemID'].values, prefetch=True
    )
    print(f"宸ヤ綔鑰呮暟閲忎负{train_sampler._num_workers}")
    # print(train_sampler.batch_size)
    
    # 璁粌   
    torch.cuda.synchronize()  # 绛塯pu鎿嶄綔瀹�
    start = time.time()
    for epoch in range(args.n_epochs):
        train_nodes_count = dict()  # 璁板綍鍚屼竴涓缁冭妭鐐归噸澶嶅嚭鐜板灏戞
        step = 0
        running_loss = 0.0
        for nf in train_sampler:           
            cacher.fetch_data(nf)
            batch_nids = nf.layer_parent_nid(-1).numpy()
            # print("batch_nids", batch_nids)
            labels=[]
            user_ids=[]
            for i in range(len(batch_nids)):
                if batch_nids[i] in train_nodes_count:
                    train_nodes_count[batch_nids[i]] += 1
                else:
                    train_nodes_count[batch_nids[i]] = 1
                label = train_dataset.loc[train_dataset['itemID'] == batch_nids[i], 'label'].iloc[train_nodes_count[batch_nids[i]]-1]
                user_id = train_dataset.loc[train_dataset['itemID'] == batch_nids[i], 'userID'].iloc[train_nodes_count[batch_nids[i]]-1]
                
                labels.insert(i, [label])
                user_ids.insert(i, user_id)       
            # print("鍒拌繖閲屼簡")
            user_ids, batch_nids, labels = torch.LongTensor(user_ids), torch.LongTensor(batch_nids), torch.Tensor(labels)
            user_ids, batch_nids, labels = user_ids.to(device), batch_nids.to(device), labels.to(device)

            optimizer.zero_grad()
            
            outputs = net(nf,user_ids, batch_nids).to(device)   

            outputs = outputs.reshape(-1,1)
            loss = criterion(outputs, labels)                               
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            step += 1
            if epoch == 0 and step == 1:
                cacher.auto_cache(g, embed_names)
        if rank == 0:
            print('[Epoch {}]train_loss: '.format(epoch+1), running_loss / step)
        if cacher.log:
            miss_rate = cacher.get_miss_rate()
            print('Epoch average miss rate: {:.4f}'.format(miss_rate))

        with torch.no_grad():
            test_nodes_count = dict()
            step = 0
            test_loss = 0
            for nf in test_sampler:                   
                cacher.fetch_data(nf)
                batch_nids = nf.layer_parent_nid(-1).numpy()
                labels=[]
                user_ids=[]
                for i in range(len(batch_nids)):
                    if batch_nids[i] in test_nodes_count:
                        test_nodes_count[batch_nids[i]] += 1
                    else:
                        test_nodes_count[batch_nids[i]] = 1
                    label = test_dataset.loc[test_dataset['itemID'] == batch_nids[i], 'label'].iloc[test_nodes_count[batch_nids[i]]-1]
                    user_id = test_dataset.loc[test_dataset['itemID'] == batch_nids[i], 'userID'].iloc[test_nodes_count[batch_nids[i]]-1]
                    labels.insert(i, [label])
                    user_ids.insert(i, user_id)
                
                user_ids, batch_nids, labels = torch.LongTensor(user_ids), torch.LongTensor(batch_nids), torch.Tensor(labels)
                user_ids, batch_nids, labels = user_ids.to(device), batch_nids.to(device), labels.to(device)
                
                outputs = net(nf,user_ids, batch_nids).to(device)
                outputs = outputs.reshape(-1,1)
                test_loss += criterion(outputs, labels).item()
                step += 1

            if rank == 0:
                print('[Epoch {}]test_loss: '.format(epoch+1), test_loss / step)
            
        torch.cuda.synchronize()  # 绛塯pu鎿嶄綔瀹�
        end = time.time()
        print('[Epoch {}]time: {:.4f}s'.format(epoch+1, end - start))

    torch.cuda.synchronize()  # 绛塯pu鎿嶄綔瀹�
    end = time.time()
    print('Total Time: {:.4f}s'.format(end - start))
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
        user_id = np.reshape(user_id,(1,)) # 浠庝竴缁存暟缁勫彉涓轰竴缁寸煩闃碉紝鍙兘鏄弧瓒虫ā鍨嬭緭鍏ヨ姹�
        item_id = np.reshape(item_id,(1,))
        label = np.reshape(label,(1,))
        # print('2', user_id, 'a', item_id.shape, 'b',label.shape, 'c')
        return user_id, item_id, label   #(array([1409]), array([3027]), array([1.], dtype=float32))

if __name__ == "__main__":

    parser = argparse.ArgumentParser() 
    parser.add_argument('--data', type=str, default='music', help='which dataset to use')
    
    parser.add_argument('--sub_graph_dataset', type=str, default='/data/xj/zj/data/wc_data_pagraph', help='which sub_graph_dataset to use')
    parser.add_argument("--gpu", type=str, default='cpu', help="gpu ids. such as 0 or 0,1,2")
    parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
    parser.add_argument('--n_epochs', type=int, default=20, help='the number of epochs')
    parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
    parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
    parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size') # 锛�
    parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization') # 锛�
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate') # ?
    parser.add_argument('--ratio', type=float, default=0.8, help='size of training dataset') # ?

    args = parser.parse_args(['--l2_weight', '1e-4']) 

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    print("args.data: ", args.data)
    data_info = pd.read_csv(os.path.join(args.sub_graph_dataset, args.data+'_info.txt'), sep="\t")
    # num_user = 1872
    # num_relation = 60
    num_user = int(data_info.iloc[0]['num_user'])
    num_relation = int(data_info.iloc[0]['num_relation'])
    print("num_user: ", num_user, "num_relation: ", num_relation)
    usr_emb = torch.nn.Embedding(num_user, args.dim)
    rel_emb = torch.nn.Embedding(num_relation, args.dim)
    
    subgraph_info = pd.read_csv(os.path.join(args.sub_graph_dataset, args.data+'_subgraph_info.txt'), sep="\t")

    mp.spawn(demo_basic, args=(world_size, usr_emb, rel_emb, subgraph_info, args), nprocs=world_size, daemon=True)