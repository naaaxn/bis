# -*- coding: utf-8 -*-
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

def in_neighbors(csc_adj, nid): # 閺勵垯绗夐弰顖濐洣閺€瑙勫灇閸戦缚绔熼敍?
  return csc_adj.indices[csc_adj.indptr[nid]: csc_adj.indptr[nid+1]] 
#瀵版鍩岄懞鍌滃仯nid闁絼绔撮崚妤勫Ν閻愮櫢绱欓懞鍌滃仯nid 閻ㄥ嫬鍙嗘潏褰掑仸鐏炲懓濡悙鐧哥礆閻ㄥ嫯顢戠槐銏犵穿

def in_neighbors_hop(csc_adj, nid, hops):
  if hops == 1:
    return in_neighbors(csc_adj, nid)
  else:
    nids = []
    for depth in range(hops):
      neighs = nids[-1] if len(nids) != 0 else [nid] # 闂団偓娑撳秹娓剁憰浣规暭閿涚喎褰ч崣鏍ㄦ付閸氬簼绔存稉顏堝仸鐏炲懓濡悙鍦畱闁鐪抽敍?
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
  neighbor_belong = belongs[neighbors] # belongs娑撳秵妲哥悮顐㈠灥婵瀵叉稉?-1閸氭绱�
  belonged = neighbor_belong[np.where(neighbor_belong != -1)]
  pid, freq = np.unique(belonged, return_counts=True)
  com_neighbor[pid] += freq
  avg_num = adj.shape[0] * 0.65 / pnum # need modify to match the train vertex num 
  score = com_neighbor * (-p_vnum + avg_num) / (r_vnum + 1)
  return score


def dg(partition_num, adj, train_nids, hops, item_df): # 閹笛嗩攽閸ユ儳鍨庨崠铏规畱娑撴槒顩﹂崙鑺ユ殶閿涘苯鐨㈤崶鐐殶閹诡喖鍨濋崚鍡楊樋娑擃亜鐡欓崶?
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
    neighbors = in_neighbors_hop(csc_adj, nid, hops)  # 閼惧嘲褰囩紒娆忕暰鐠鸿櫕鏆熼崘鍛畱閹碘偓閺堝鍋︾仦鍛板Ν閻�?
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
    p_trainids = np.where(belongs == pid)[0] # belongs鐎涙ê鍋嶅В蹇庨嚋閼哄倻鍋ｉ惃鍕瀻閸栧搫褰�
    
    mask = item_df['itemID'].isin(p_trainids)
    filtered_df = item_df[mask]
    print("閸掑棗灏�", pid, "  鐠侇厾绮岄梿鍡楀灊閸掑棴绱�")
    print(filtered_df)
    sub_user.append(filtered_df)
    # print("sub_user:")
    # print(sub_user)

    sub_trainv.append(p_trainids) # 瀵版鍩岄崠鍝勫娇pid閻ㄥ嫯濡悙鍦椽閸�?
    p_v = np.where(r_belongs[pid] != -1)[0] #r_belongs [partition_num, vnum],娑撳秴鎮撻崚鍡楀隘閺勵垰鎯侀張澶庣箹娑擃亣濡悙鐧哥礉
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
  # user_item_rating閺佺増宓� 
  data_loader = DataLoader(args.dataset_name)
  item_df = data_loader.load_dataset()
  # 閸樺娅庨幏顒€褰块崪宀勨偓妤€褰�
  item_df.userID = item_df['userID'].astype(str).str.replace(',)', '')
  item_df.userID = item_df['userID'].str.replace('(', '').astype(int)
  # print("item_df:")
  # print(item_df)

  # 鐎电厧鍙唊g閸�?
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
  # 閹跺rain_nids閿涘牐顔勭紒鍐肠閻ㄥ嫬骞撻柌宥堝Ν閻愮D閿涘娲块幒銉︽禌閹广垺鍨氱拠鍕瀻鐞涖劑鍣烽棃銏㈡畱items
  unique_items = item_df['itemID'].drop_duplicates()
  train_nids = unique_items.to_numpy()

  p_v, p_trainv, p_user = dg(args.partition, adj, train_nids, args.num_hop, item_df)
  # print("p_v:")
  # print(p_v) # 婢舵熬绱濇稉宥勭矌閸栧懏瀚崚鍡楀隘閼哄倻鍋ｉ敍宀冪箷閸栧懏瀚拋顓犵矊閼哄倻鍋ｉ悽銊ュ煂閻ㄥ嫰鍋︾仦鍛板Ν閻愮櫢绱濋崣鏉抔鐟曚胶鏁_v閹靛秴顧勯悽?
  # print("p_trainv:")
  # print(p_trainv) # 閸栧懎鎯堝В蹇庨嚋閸掑棗灏惃鍕唲缂佸啳濡悙褰掓肠閸氬牞绱濈亸?
  print("p_user:")
  print(p_user) # rating閺佺増宓侀崚鎺戝瀻
  
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

    #閻╁瓨甯撮崗鍫ｆ祮閹广垺鍨歬gcn閼宠姤甯撮崣妤冩畱閺嶇厧绱￠敍灞界摟閸忓摜娈戣ぐ銏犵础閺囨潙顔愰弰鎾规祮閸栨牭绱濇担璺ㄦ暏DGraph瀵板牆顦挎稉婊嗐偪娑撳秷鍏樼涵顔肩暰閿涘奔绡冩稉宥夆偓鍌滄暏娴滃冬GCN婢跺嫮鎮婇惃鍕壐瀵�?
    kg_df_filtered = kg_df[(kg_df['head'].isin(p_v)) | (kg_df['tail'].isin(pv))]
    print("閸掑棗灏崥搴ｆ畱kg")
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

    # 娣囨繂鐡ㄧ拋顓犵矊闂嗗棜濡悙?
    sub_user_file = os.path.join(
      partition_dataset,
      'sub_user_{}.csv'.format(str(pid)))
    puser.to_csv(sub_user_file, sep='\t', index = False)

    spsp.save_npz(subadj_file, subadj)
    np.save(sub_trainid_file, subtrainid)
    np.save(sub_train2full_file, sub2fullid)
    # np.save(sub_label_file, sublabel)
