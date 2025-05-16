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
import PaGraph.storage as storage
from PaGraph.parallel import SampleLoader
import dgl
import scipy.sparse
from dgl import DGLGraph
import PaGraph.data as data
import avx_sampler_bag
import my_sampler_th_1



#读入该分区的训练集
dataname = os.path.join('PaGraph/data/wc_data_pagraph')   
sub_train_file = os.path.join(dataname, 'ratings_data.txt')
df_dataset = pd.read_csv(sub_train_file, sep=' ')

#读入该分区的kg（KGCN可处理的格式）
sub_train_file = os.path.join(dataname, 'ratings_data.txt')
df_kg = pd.read_csv(sub_train_file, sep=' ')
# print("df_kg: ", df_kg)

# 重新映射子图, 构造创建COO矩阵
unique_nodes = pd.concat([df_kg['head'], df_kg['tail']]).drop_duplicates().reset_index(drop=True)
num_nodes = len(unique_nodes)
continuous_ids = np.arange(num_nodes)
full_to_sub = dict(zip(unique_nodes, continuous_ids))
sub_to_full = torch.LongTensor(unique_nodes)

df_kg['head'] = df_kg['head'].map(full_to_sub)
df_kg['tail'] = df_kg['tail'].map(full_to_sub)


rows = df_kg['head'].values  
cols = df_kg['tail'].values  
data = df_kg['relation'].values
print(data)
src_node, dst_node = np.concatenate((rows, cols)), np.concatenate((cols, rows))
data_copy = data.copy()
data = np.concatenate((data, data_copy))
shape = (num_nodes, num_nodes)  
sub_adj = scipy.sparse.coo_matrix((data, (src_node, dst_node)), shape=shape)



g = DGLGraph(sub_adj, readonly=True)

#seed = random.sample(list(set(rows)), 16)
seed =list(set(rows))
print(f"seed的长度为{len(seed)}")

with open('nei2.txt', 'a') as file:
    for item in seed:
        file.write(f"{item} ")  # 每个元素换行写入
    file.write("over\n")

start=time.time()
#dgl.contrib.sampling.NeighborSampler
#my_sampler_th.NeighborSampler
train_sampler = my_sampler_th_1.NeighborSampler(g, batch_size=32,
      expand_factor =50, neighbor_type='in', shuffle=False, num_hops=3,
      seed_nodes=seed, prefetch=True
    )
# test_sampler = dgl.contrib.sampling.NeighborSampler(g, batch_size=32,
#       expand_factor =50, neighbor_type='in', shuffle=False, num_hops=3, 
#       seed_nodes=seed, prefetch=True
#     )
print(f"\n采样时间为：{time.time()-start} ")

# start=time.time()
# for nf in test_sampler:
#     flag=0
#     temp_index=1
#     for ii in range(4):
#             # print(self.device)
#             # if flag:
#             #     test1_array=nf.block_parent_eid(temp_index).numpy()
#             #     temp_index-=1
#             #     with open('nei2.txt', 'a') as file:
#             #         file.write(f"第{3-1-temp_index}层边 ")
#             #         for item in sorted(test1_array):
#             #             file.write(f"{item} ")  # 每个元素换行写入
#             #         file.write(f"\n该层边数量为：{len(test1_array)} ")
#             #         file.write("over\n")
#             batch_nids = nf.layer_parent_nid(3-ii).numpy()
#             # 打开文件并以追加模式写入
#             with open('nei0.txt', 'a') as file:
#                 file.write(f"第{ii}层 ")
#                 file.write(f"\n该层节点数量为：{len(batch_nids)} ")
#                 file.write("over\n")
#             flag=1

# print(f"\n采样时间为：{time.time()-start} ")

# start=time.time()
# batch_nids=[]
# for nf in train_sampler:
#     flag=0
#     temp_index=1
#     for ii in range(4):
#             # print(self.device)
#             # if flag:
#             #     test1_array=nf.block_parent_eid(temp_index).numpy()
#             #     temp_index-=1
#             #     with open('nei1.txt', 'a') as file:
#             #         file.write(f"第{3-1-temp_index}层边 ")
#             #         for item in sorted(test1_array):
#             #             file.write(f"{item} ")  # 每个元素换行写入
#             #         file.write(f"\n该层边数量为：{len(test1_array)} ")
#             #         file.write("over\n")
#             batch_nids=(nf.layer_parent_nid(3-ii).numpy())
#             with open('nei.txt', 'a') as file:
#                 file.write(f"第{ii}层 ")
#                 file.write(f"\n该层节点数量为：{len(batch_nids)} ")
#                 file.write("over\n")
#             flag=1
# print(f"\n采样时间为：{time.time()-start} ")

start=time.time()
sampler = avx_sampler_bag.avx_edges_sampler.MultiLayerNeighborSampler([50, 50, 50,50])  # 三层采样
# 创建数据加载器
dataloader = dgl.dataloading.DataLoader(
    g,
    seed,
    sampler,
    batch_size=32,
    shuffle=False,
    drop_last=False,
    prefetch_factor=None,
    num_workers=0
)

# 迭代 dataloader
with open("nei3.txt", "a") as file:
    for input_nodes, output_nodes, blocks in dataloader:
        for i, block in enumerate(blocks):
            # 目标节点 (dst_nodes) 是当前层的采样节点
            dst_nodes = block.dstdata[dgl.NID]
            file.write(f"第0层 ")
            file.write(f"\n该层节点数量为：{len(dst_nodes)} ")
            file.write("over\n")
        src_nodes = block.srcdata[dgl.NID]
        file.write(f"第0层 ")
        file.write(f"\n该层节点数量为：{len(dst_nodes)} ")
        file.write("over\n")

print(f"\n采样时间为：{time.time()-start} ")
