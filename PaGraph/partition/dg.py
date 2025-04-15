import os
import sys
import dgl
from dgl import DGLGraph
import torch
import numpy as np
import scipy.sparse as spsp
import argparse
import PaGraph.data as data
import pandas as pd

import ordering
from utils_yxb import get_sub_graph
from data_loader import DataLoader

def in_neighbors(csc_adj, nid): # 是不是要改成出边？
  return csc_adj.indices[csc_adj.indptr[nid]: csc_adj.indptr[nid+1]] 
#得到节点nid那一列节点（节点nid 的入边邻居节点）的行索引

def in_neighbors_hop(csc_adj, nid, hops):
  if hops == 1:
    return in_neighbors(csc_adj, nid)
  else:
    nids = []
    for depth in range(hops):
      neighs = nids[-1] if len(nids) != 0 else [nid] # 需不需要改？只取最后一个邻居节点的邻居？
      # print("depth:", depth, "  neighs:", neighs)
      for n in neighs:
        nids.append(in_neighbors(csc_adj, n))
        # print("n:", n, "  nids:", nids)

    return np.unique(np.hstack(nids))


def dg_max_score(score, p_vnum):
  ids = np.argsort(score)[-2:]
  if score[ids[0]] != score[ids[1]]:
    return ids[1]
  else:
    return ids[0] if p_vnum[ids[0]] < p_vnum[ids[1]] else ids[1]


def dg_ind(adj, neighbors, belongs, p_vnum, r_vnum, pnum):
  """
  Params:
    neighbor: in-neighbor vertex set
    belongs: np array, each vertex belongings to which partition
    p_vnum: np array, each partition total vertex w/o. redundancy
    r_vnum: np array, each partition total vertex w/. redundancy
    pnum: partition number
  """
  com_neighbor = np.ones(pnum, dtype=np.int64)
  score = np.zeros(pnum, dtype=np.float32)
  # count belonged vertex
  neighbor_belong = belongs[neighbors] # belongs不是被初始化为-1吗？
  belonged = neighbor_belong[np.where(neighbor_belong != -1)]
  pid, freq = np.unique(belonged, return_counts=True)
  com_neighbor[pid] += freq
  avg_num = adj.shape[0] * 0.65 / pnum # need modify to match the train vertex num 
  score = com_neighbor * (-p_vnum + avg_num) / (r_vnum + 1)
  return score


def dg(partition_num, adj, train_nids, hops, item_df): # 执行图分区的主要函数，将图数据划分多个子图
  csc_adj = adj.tocsc()
  vnum = adj.shape[0]
  vtrain_num = train_nids.shape[0]
  belongs = -np.ones(vnum, dtype=np.int8)
  r_belongs = [-np.ones(vnum, dtype=np.int8) for _ in range(partition_num)]
  p_vnum = np.zeros(partition_num, dtype=np.int64)
  r_vnum = np.zeros(partition_num, dtype=np.int64)

  progress = 0
  #for nid in range(0, train_nids):
  print('total vertices: {} | train vertices: {}'.format(vnum, vtrain_num))
  for step, nid in enumerate(train_nids):  
    #neighbors = in_neighbors(csc_adj, nid)
    neighbors = in_neighbors_hop(csc_adj, nid, hops)  # 获取给定跳数内的所有邻居节点
    score = dg_ind(csc_adj, neighbors, belongs, p_vnum, r_vnum, partition_num)
    ind = dg_max_score(score, p_vnum)
    if belongs[nid] == -1:
      belongs[nid] = ind
      p_vnum[ind] += 1
      neighbors = np.append(neighbors, nid)
      for neigh_nid in neighbors:
        if r_belongs[ind][neigh_nid] == -1:
          r_belongs[ind][neigh_nid] = 1
          r_vnum[ind] += 1
    # progress
    if int(vtrain_num * progress / 100) <= step:
      sys.stdout.write('=>{}%\r'.format(progress))
      sys.stdout.flush()
      progress += 1
  print('')
  
  sub_v = []
  sub_trainv = []
  sub_user = [] # user
  for pid in range(partition_num):
    p_trainids = np.where(belongs == pid)[0] # belongs存储每个节点的分区号
    
    mask = item_df['itemID'].isin(p_trainids)
    filtered_df = item_df[mask]
    print("分区", pid, "  训练集划分：")
    print(filtered_df)
    sub_user.append(filtered_df)
    # print("sub_user:")
    # print(sub_user)

    sub_trainv.append(p_trainids) # 得到区号pid的节点编号
    p_v = np.where(r_belongs[pid] != -1)[0] #r_belongs [partition_num, vnum],不同分区是否有这个节点，
    sub_v.append(p_v)
    assert p_v.shape[0] == r_vnum[pid]
    print('vertex# with self-reliance: ', r_vnum[pid])
    print('vertex# w/o  self-reliance: ', p_vnum[pid])
    #print('orginal vertex: ', np.where(belongs == pid)[0])
    #print('redundancy vertex: ', np.where(r_belongs[pid] != -1)[0])
  print("sub_trainv:")
  print(sub_trainv)
  print("sub_v:")
  print(sub_v)

  return sub_v, sub_trainv, sub_user

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Partition')
  parser.add_argument("--dataset", type=str, default=None,
                      help="dataset dir")
  parser.add_argument("--partition", type=int, default=2,
                      help="num of partitions")
  parser.add_argument("--num_hop", type=int, default=1,
                      help="num of hop neighbors required for a batch")
  parser.add_argument("--ordering", dest='ordering', action='store_true')
  parser.add_argument('--dataset_name', type=str, default='music', help='which dataset to use')
  parser.set_defaults(ordering=False)
  args = parser.parse_args()

  # get data
  adj = spsp.load_npz(os.path.join(args.dataset, '{}/adj.npz'.format(args.dataset_name)))
  # for row, col, value in zip(adj.row, adj.col, adj.data):
  #     print(f"({row}, {col}): {value}")

  train_mask, val_mask, test_mask = data.get_masks(args.dataset)
  train_nids = np.nonzero(train_mask)[0].astype(np.int64)
  labels = data.get_labels(args.dataset)
  # user_item_rating数据 
  data_loader = DataLoader(args.dataset_name)
  item_df = data_loader.load_dataset()
  # 去除括号和逗号
  item_df.userID = item_df['userID'].astype(str).str.replace(',)', '')
  item_df.userID = item_df['userID'].str.replace('(', '').astype(int)
  # print("item_df:")
  # print(item_df)

  # 导入kg图
  # kg_df = data_loader.load_kg()
  kg_df = data_loader.df_kg
  # kg_df = pd.read_csv(os.path.join(args.dataset, 'data/{}/kg.txt'.format(args.dataset_name)), sep="\t", names=["head", "relation", "tail"])
  # print("kg_df:")
  # print(kg_df) 

  # ordering
  if args.ordering:
    print('re-ordering graphs...')
    adj = adj.tocsc()
    adj, vmap = ordering.reordering(adj, depth=args.num_hop) # vmap: orig -> new
    # save to files
    mapv = np.zeros(vmap.shape, dtype=np.int64)
    mapv[vmap] = np.arange(vmap.shape[0]) # mapv: new -> orig
    train_nids = np.sort(vmap[train_nids])
    spsp.save_npz(os.path.join(args.dataset, 'adj.npz'), adj)
    np.save(os.path.join(args.dataset, 'labels.npy'), labels[mapv])
    np.save(os.path.join(args.dataset, 'train.npy'), train_mask[mapv])
    np.save(os.path.join(args.dataset, 'val.npy'), val_mask[mapv])
    np.save(os.path.join(args.dataset, 'test.npy'), test_mask[mapv])
  
  # partition
  # 把train_nids（训练集的去重节点ID）直接替换成评分表里面的items
  unique_items = item_df['itemID'].drop_duplicates()
  train_nids = unique_items.to_numpy()

  p_v, p_trainv, p_user = dg(args.partition, adj, train_nids, args.num_hop, item_df)
  # print("p_v:")
  # print(p_v) # 多，不仅包括分区节点，还包括训练节点用到的邻居节点，取kg要用p_v才够用
  # print("p_trainv:")
  # print(p_trainv) # 包含每个分区的训练节点集合，少
  print("p_user:")
  print(p_user) # rating数据划分
  
  # save to file
  partition_dataset = os.path.join(args.dataset, '{}naive_{}'.format(args.partition, args.dataset_name))
  try:
    os.mkdir(partition_dataset)
  except FileExistsError:
    pass
  dgl_g = dgl.DGLGraph(adj, readonly=True)

  for pid, (pv, ptrainv, puser) in enumerate(zip(p_v, p_trainv, p_user)):
    print('generating subgraph# {}...'.format(pid))
    #subadj, sub2fullid, subtrainid = node2graph(adj, pv, ptrainv)
    subadj, sub2fullid, subtrainid = get_sub_graph(dgl_g, ptrainv, args.num_hop)

    #直接先转换成kgcn能接受的格式，字典的形式更容易转化，使用DGraph很多东西不能确定，也不适用于KGCN处理的格式
    kg_df_filtered = kg_df[(kg_df['head'].isin(p_v)) | (kg_df['tail'].isin(pv))]
    print("分区后的kg：")
    print(kg_df_filtered)

    # sublabel = labels[sub2fullid[subtrainid]] # !!!IndexError: index 12 is out of bounds for axis 0 with size 10
    # files
    subadj_file = os.path.join(
      partition_dataset,
      'subadj_{}.npz'.format(str(pid)))
    sub_trainid_file = os.path.join(
      partition_dataset,
      'sub_trainid_{}.npy'.format(str(pid)))
    sub_train2full_file = os.path.join(
      partition_dataset,
      'sub_train2fullid_{}.npy'.format(str(pid)))
    sub_label_file = os.path.join(
      partition_dataset,
      'sub_label_{}.npy'.format(str(pid)))
  
    sub_kg_file = os.path.join(
      partition_dataset,
      'sub_kg_{}.csv'.format(str(pid)))
    kg_df_filtered.to_csv(sub_kg_file, sep='\t', index = False) 

    # 保存训练集节点
    sub_user_file = os.path.join(
      partition_dataset,
      'sub_user_{}.csv'.format(str(pid)))
    puser.to_csv(sub_user_file, sep='\t', index = False)

    spsp.save_npz(subadj_file, subadj)
    np.save(sub_trainid_file, subtrainid)
    np.save(sub_train2full_file, sub2fullid)
    # np.save(sub_label_file, sublabel)
