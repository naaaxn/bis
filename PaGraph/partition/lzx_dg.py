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
from utils import get_sub_graph

#对图数据进行分区，并将分区后的子图数据保存到文件中。
# 获取节点的入边邻居节点
def in_neighbors(csc_adj, nid):
  return csc_adj.indices[csc_adj.indptr[nid]: csc_adj.indptr[nid+1]]

# 获取节点的指定跳数的入边邻居节点
def in_neighbors_hop(csc_adj, nid, hops):
  if hops == 1:
    return in_neighbors(csc_adj, nid)
  else:
    nids = []
    for depth in range(hops):
      neighs = nids[-1] if len(nids) != 0 else [nid]
      for n in neighs:
        nids.append(in_neighbors(csc_adj, n))
    return np.unique(np.hstack(nids))

# 计算最大得分的节点
def dg_max_score(score, p_vnum):
  ids = np.argsort(score)[-2:]
  if score[ids[0]] != score[ids[1]]:
    return ids[1]
  else:
    return ids[0] if p_vnum[ids[0]] < p_vnum[ids[1]] else ids[1]

# 计算节点的指标
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
  neighbor_belong = belongs[neighbors]
  belonged = neighbor_belong[np.where(neighbor_belong != -1)]
  pid, freq = np.unique(belonged, return_counts=True)
  com_neighbor[pid] += freq
  avg_num = adj.shape[0] * 0.65 / pnum # need modify to match the train vertex num
  score = com_neighbor * (-p_vnum + avg_num) / (r_vnum + 1)
  return score

# 分区处理函数，只取了部分进行训练
#adj为kg三元组数据，train_nids()为训练节点，hops为跳数
#partition_num=2
def dg(partition_num, adj, train_nids, hops,input_dataframe):

  csc_adj = adj.tocsc()
  vnum = adj.shape[0]  # 获取图中节点的数量
  vtrain_num = train_nids.shape[0]  # 获取训练节点的数量
  belongs = -np.ones(vnum, dtype=np.int8)  # 初始化节点所属分区数组，全为-1
  r_belongs = [-np.ones(vnum, dtype=np.int8) for _ in range(partition_num)]  # 初始化每个分区的节点所属数组，同样均为-1
  p_vnum = np.zeros(partition_num, dtype=np.int64)  # 初始化每个分区的节点数量数组 [0 0]
  r_vnum = np.zeros(partition_num, dtype=np.int64)  # 初始化每个分区的具有冗余的节点数量数组 [0 0]

  progress = 0  # 进度计数器
  #for nid in range(0, train_nids):
  print('total vertices: {} | train vertices: {}'.format(vnum, vtrain_num))  # 打印总节点数和训练节点数
  # 遍历训练节点
  for step, nid in enumerate(train_nids):  
    #neighbors = in_neighbors(csc_adj, nid)
    neighbors = in_neighbors_hop(csc_adj, nid, hops)  # 获取节点的指定跳数的入边邻居节点
    score = dg_ind(csc_adj, neighbors, belongs, p_vnum, r_vnum, partition_num)  # 计算节点的指标
    ind = dg_max_score(score, p_vnum)  # 获取最大得分的节点
    if belongs[nid] == -1:  # 如果节点未分配分区
      belongs[nid] = ind  # 分配节点到分区
      p_vnum[ind] += 1  # 分区节点数量加一
      neighbors = np.append(neighbors, nid)  # 将当前节点添加到邻居节点中
      for neigh_nid in neighbors:  # 遍历邻居节点
        if r_belongs[ind][neigh_nid] == -1:  # 如果邻居节点未分配分区
          r_belongs[ind][neigh_nid] = 1  # 分配邻居节点到分区
          r_vnum[ind] += 1  # 分区具有冗余的节点数量加一
    # 进度显示
    if int(vtrain_num * progress / 100) <= step:
      sys.stdout.write('=>{}%\r'.format(progress))  # 显示进度百分比
      sys.stdout.flush()
      progress += 1
  print('')  # 换行

  #pid代表分区编号，sub_v代表每个分区的节点，sub_trainv代表每个分区的训练节点
  sub_v = []  
  sub_trainv = []  
  sub_user=[]
  # 遍历每个分区
  for pid in range(partition_num):  
    print("----------当前分区："+str(pid)+"-------------")
    p_trainids = np.where(belongs == pid)[0]  # 获取属于当前分区的节点

    print("-------------print(p_trainids)----------")
    print(p_trainids)

    #将user一起分区了
    mask = input_dataframe['itemID'].isin(p_trainids)
    partition_data = input_dataframe[mask]
    print("----------user----------")
    print(partition_data)
    sub_user.append(partition_data)

    sub_trainv.append(p_trainids)  # 添加到训练节点列表中
    p_v = np.where(r_belongs[pid] != -1)[0]  # 获取具有冗余的节点
    sub_v.append(p_v)  # 添加到节点列表中
    assert p_v.shape[0] == r_vnum[pid]  # 断言确保节点数量正确
    print('vertex# with self-reliance: ', r_vnum[pid])  # 打印具有冗余的节点数量
    print('vertex# w/o  self-reliance: ', p_vnum[pid])  # 打印不具有冗余的节点数量
    #print('orginal vertex: ', np.where(belongs == pid)[0])
    #print('redundancy vertex: ', np.where(r_belongs[pid] != -1)[0])
  return sub_v, sub_trainv,sub_user  # 返回分区后的节点和训练节点列表

# #提取user和label
# def user_partition(item_ids_to_partition, input_dataframe):
#     """
#     根据给定的itemIDs，将数据分割成相应的部分，并整合到一个新的DataFrame中返回。
    
#     :param item_ids_to_partition: 一个包含itemID的集合或列表。
#     :param input_dataframe: 待处理的DataFrame。
#     :return: 分割并整合后的DataFrame。
#     """
#     # 预先创建一个空的DataFrame，用于存储结果，避免在循环中使用append
#     new_df = pd.DataFrame()
#     # 使用列表来收集各个分区的数据
#     partition_data_list = []
#     for item_id in item_ids_to_partition:
#         # 直接筛选满足条件的数据，并将其添加到列表中
#         partition_data = input_dataframe.loc[input_dataframe['itemID'] == item_id]
#         partition_data_list.append(partition_data)
#     # 使用pd.concat来整合所有分区的数据
#     new_df = pd.concat(partition_data_list, ignore_index=False)
#     return new_df


# 主函数
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Partition')
  parser.add_argument("--dataset", type=str, default=None,
                      help="dataset dir")
  parser.add_argument("--partition", type=int, default=2,
                      help="num of partitions")
  parser.add_argument("--num_hop", type=int, default=1,
                      help="num of hop neighbors required for a batch")
  parser.add_argument("--ordering", dest='ordering', action='store_true')
  parser.set_defaults(ordering=False)
  args = parser.parse_args()


  #获取数据
  adj = spsp.load_npz(os.path.join(args.dataset, 'adj.npz'))
  train_mask, val_mask, test_mask = data.get_masks(args.dataset)
  train_nids = np.nonzero(train_mask)[0].astype(np.int64)
  #找到train_mask中值为True的位置，并返回这些位置的整数索引。
  labels = data.get_labels(args.dataset)

  #获取user_item_label.txt数据
  df_uil = pd.read_csv(os.path.join(args.dataset,'user_item_label.txt'), sep="\s+", header=None, names=["userID", "itemID", "label"])
  # print(df_uil)
  
  #排序，不重要
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
  # 去重
  train_nids = df_uil['itemID'].drop_duplicates()

  p_v, p_trainv,p_user = dg(args.partition, adj, train_nids, args.num_hop,df_uil)
  # print("---------p_trainv----------")
  # print(p_trainv)

  # #创建partition_results用来存储分区后的user数据
  # partition_results = {}
  # for i in range(args.partition):
  #     print(f"---------------------------------分区{i}的user内容:")
  #     # 应用user_partition函数，并将结果存储到字典中，键为分区编号字符串
  #     partition_results[f"partition_{i}"] = user_partition(p_trainv[i], df_uil)
  #     # partition_results[f"partition_{i}"].to_csv(f'output{i}.csv', index=False, encoding='utf-8')
  #     print(partition_results[f"partition_{i}"])

  # save to file
  partition_dataset = os.path.join(args.dataset, '{}naive'.format(args.partition))
  try:
    #用于存储分区后的数据
    os.mkdir(partition_dataset)
  except FileExistsError:
    pass
  #基于adj文件构建一个只读的图对象dgl_g
  dgl_g = dgl.DGLGraph(adj, readonly=True)
  #对每个分区进行循环处理
  #在循环中生成子图数据，并将邻接矩阵、训练节点ID、全局节点ID和标签分别保存
  for pid, (pv, ptrainv) in enumerate(zip(p_v, p_trainv)):
    print('generating subgraph# {}...'.format(pid))
    #subadj, sub2fullid, subtrainid = node2graph(adj, pv, ptrainv)
    #获取子图
    subadj, sub2fullid, subtrainid = get_sub_graph(dgl_g, ptrainv, args.num_hop)
    sublabel = labels[sub2fullid[subtrainid]]
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
    spsp.save_npz(subadj_file, subadj)
    np.save(sub_trainid_file, subtrainid)
    np.save(sub_train2full_file, sub2fullid)
    np.save(sub_label_file, sublabel)