import os
import sys
import dgl
from dgl import DGLGraph
import torch
import numpy as np
import scipy.sparse as spsp

def get_sub_graph(dgl_g, train_nid, num_hops):
  nfs = []
  for nf in dgl.contrib.sampling.NeighborSampler(dgl_g, len(train_nid),
                                                 dgl_g.number_of_nodes(),
                                                 neighbor_type='in',
                                                 shuffle=False,
                                                 num_workers=16,
                                                 num_hops=num_hops,
                                                 seed_nodes=train_nid,
                                                 prefetch=False):
    nfs.append(nf)
  
  assert(len(nfs) == 1)
  nf = nfs[0]
  full_edge_src = []
  full_edge_dst = []
  full_edge_weights = []
  for i in range(nf.num_blocks):
    # nf_src_nids, nf_dst_nids, _ = nf.block_edges(i, remap_local=False)
    nf_src_nids, nf_dst_nids, _ = nf.block_edges(i, remap_local=False)
    full_edge_src.append(nf.map_to_parent_nid(nf_src_nids))
    full_edge_dst.append(nf.map_to_parent_nid(nf_dst_nids))

  full_srcs = torch.cat(tuple(full_edge_src)).numpy()
  full_dsts = torch.cat(tuple(full_edge_dst)).numpy()

  # 直接把子图导出，转换为kgcn处理的dataframe格式
  print("full_scrs:")
  print(full_srcs.shape)
  print(full_srcs)

  # has_nodes = dgl_g.has_nodes(full_srcs)
  # print("has nodes head:", has_nodes)
  # is1 = all(x==1 for x in has_nodes)
  # print("is1:", is1)

  print("full_dsts:")
  print(full_dsts.shape)
  print(full_dsts)

 
  # set up mappings
  sub2full = np.unique(np.concatenate((full_srcs, full_dsts)))
  full2sub = np.zeros(np.max(sub2full) + 1, dtype=np.int64)
  full2sub[sub2full] = np.arange(len(sub2full), dtype=np.int64)
  # map to sub graph space
  sub_srcs = full2sub[full_srcs]
  sub_dsts = full2sub[full_dsts]
  vnum = len(sub2full)
  enum = len(sub_srcs)
  # data = np.ones(sub_srcs.shape[0], dtype=np.uint8) # 边的权重
  # 步骤1：按行堆叠元素对
  paired_array = np.stack((full_srcs, full_dsts), axis=-1)
  # 步骤2：去除重复的配对 但为什么会有重复配对？？存在的，两点多边，因为两个东西的关系不一定是只有一个，所以是重复图
  unique_paired_array, indices = np.unique(paired_array, axis=0, return_index=True, return_inverse=False)
  # 步骤3：重新分成两个Numpy数组
  new_array1 = unique_paired_array[:, 0]
  new_array2 = unique_paired_array[:, 1] 
  # new_array1 = np.append(new_array1, -1)
  # new_array2 = np.append(new_array2, -1)

  # data = (dgl_g.edge_ids([0,1,3, 3, 1, 0], [4454, 7498,9347, 9347, 7498, 4454], return_uv=False)).numpy()
  
  data = torch.cat(tuple(dgl_g.edge_ids(new_array1, new_array2, return_uv=True))).numpy()
  # data = torch.cat(tuple(dgl_g.edge_ids(full_srcs, full_dsts, return_uv=False))).numpy()
  
  
  data = data[-full_srcs.shape[0]:]
  print("边的数量：")
  print(data.shape)
  print(data)
  coo_adj = spsp.coo_matrix((data, (sub_srcs, sub_dsts)), shape=(vnum, vnum)) # 坏了，这里存储的是子图 哦没事，既然有映射那就可以直接转成全局图
  # coo_adj = spsp.coo_matrix((data, (full_srcs, full_dsts)), shape=(vnum, vnum)) # 只能改成全局图itemID，不然训练集没法用,也不需要子图到全局图的映射
  csr_adj = coo_adj.tocsr() # remove redundant edges
  enum = csr_adj.data.shape[0]
  # csr_adj.data = np.ones(enum, dtype=np.uint8) #这里又设为1？为什么
  print('vertex#: {} edge#: {}'.format(vnum, enum))
  # train nid
  tnid = nf.layer_parent_nid(-1).numpy()
  valid_t_max = np.max(sub2full)
  valid_t_min = np.min(tnid)
  tnid = np.where(tnid <= valid_t_max, tnid, valid_t_min)
  subtrainid = full2sub[np.unique(tnid)]
  return csr_adj, sub2full, subtrainid


def node2graph(fulladj, nodelist, train_nids):
  g = dgl.DGLGraph(fulladj)
  subg = g.subgraph(nodelist)
  sub2full = subg.parent_nid.numpy()
  subadj = subg.adjacency_matrix_scipy(transpose=True, return_edge_ids=False)
  # get train vertices under subgraph scope
  subtrain = subg.map_to_subgraph_nid(train_nids).numpy()
  return subadj, sub2full, subtrain