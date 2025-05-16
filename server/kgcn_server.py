import sys
import os
import argparse
import numpy as np
import torch
import dgl
from dgl import DGLGraph
from dgl.contrib.sampling import SamplerPool
import dgl.function as fn
import multiprocessing
import sys
import os
import argparse
import numpy as np
import torch
import dgl
from dgl import DGLGraph
from dgl.contrib.sampling import SamplerPool
import dgl.function as fn
import multiprocessing

import PaGraph.data.get_data_kgcn as data
from PaGraph.parallel import SampleDeliver

def main(args):
  coo_adj, vfeat, efeat = data.get_graph_data(args.dataset,args.data)

  graph = dgl.DGLGraph(coo_adj, readonly=True)
  vfeatures = torch.FloatTensor(vfeat)
  efeatures = torch.FloatTensor(efeat)

  graph_name = os.path.basename(args.dataset)
  vnum = graph.number_of_nodes()
  enum = graph.number_of_edges()
  vfeat_size = vfeat.shape[1]
  efeat_size = efeat.shape[1]

  print('=' * 30)
  print("Graph Name: {}\nNodes Num: {}\tEdges Num: {}\nvFeature Size: {}\neFeature Size: {}"
        .format(graph_name, vnum, enum, vfeat_size, efeat_size)
  )
  print('=' * 30)

  # create server
  g = dgl.contrib.graph_store.create_graph_store_server(
        graph, graph_name,
        'shared_mem', args.num_workers, 
        False)
  
  # calculate norm for gcn  鐠侊紕鐣籊CN閻ㄥ嫬缍婃稉鈧崠鏍ф礈鐎涳拷
  dgl_g = DGLGraph(graph, readonly=True)

  if args.model == 'gcn':
    dgl_g = DGLGraph(graph, readonly=True)
    # 鐠侊紕鐣绘禍鍡樼槨娑擃亣濡悙鍦畱閸忋儱瀹抽敍灞借嫙閸欐牕鍙鹃崐鎺撴殶娴ｆ粈璐熻ぐ鎺嶇閸栨牕娲滅€涙劑鈧倸娲滄稉鍝勬躬GCN娑擃叏绱濇稉杞扮啊闁灝鍘ゆ惔锔芥殶婢堆呮畱閼哄倻鍋ｇ€靛湱绮ㄩ弸婊€楠囬悽鐔荤箖婢堆呮畱瑜板崬鎼烽敍宀勨偓姘埗娴兼矮濞囬悽銊ㄥΝ閻愬湱娈戞惔锔芥殶閻ㄥ嫬鈧帗鏆熸潻娑滎攽瑜版帊绔撮崠锟�
    norm = 1. / dgl_g.in_degrees().float().unsqueeze(1)
    # preprocess 
    if args.preprocess:
      print('Preprocessing features...')
      dgl_g.ndata['norm'] = norm
      dgl_g.ndata['features'] = features
      dgl_g.update_all(fn.copy_src(src='features', out='m'),
                       fn.sum(msg='m', out='preprocess'),
                       lambda node : {'preprocess': node.data['preprocess'] * node.data['norm']})
      features = dgl_g.ndata['preprocess']
    g.ndata['norm'] = norm
    g.ndata['vfeatures'] = vfeatures
    g.edata['efeatures'] = efeatures
    del dgl_g

  elif args.model == 'graphsage':
    if args.preprocess: # for simple preprocessing
      print('preprocessing: warning: jusy copy')
      g.ndata['neigh'] = features
    g.ndata['features'] = features

  # remote sampler 
  if args.sample:
    subgraph = []
    sub_trainnid = []
    for rank in range(args.num_workers):
      subadj, _ = data.get_sub_train_graph(args.dataset, rank, args.num_workers)
      train_nid = data.get_sub_train_nid(args.dataset, rank, args.num_workers)
      subgraph.append(dgl.DGLGraph(subadj, readonly=True))
      sub_trainnid.append(train_nid)
    hops = args.gnn_layers - 1 if args.preprocess else args.gnn_layers
    print('Expected trainer#: {}. Start sampling at server end...'.format(args.num_workers))
    deliver = SampleDeliver(subgraph, sub_trainnid, args.num_neighbors, hops, args.num_workers)
    deliver.async_sample(args.n_epochs, args.batch_size, one2all=args.one2all)
    
  print('start running graph server on dataset: {}'.format(graph_name))
  g.run()




if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='GraphServer')
 
  parser.add_argument('--data', type=str, default='music', help='which data to use')

  parser.add_argument("--dataset", type=str, default=None,
                      help="dataset folder path")
  
  parser.add_argument("--num-workers", type=int, default=1,
                      help="the number of workers")
  
  parser.add_argument("--model", type=str, default="gcn",
                      help="model type for preprocessing")

  # sample options
  parser.add_argument("--sample", dest='sample', action='store_true')
  parser.set_defaults(sample=False)
  parser.add_argument("--num-neighbors", type=int, default=2)
  parser.add_argument("--gnn-layers", type=int, default=2)
  parser.add_argument("--batch-size", type=int, default=6000)
  #parser.add_argument("--num-workers", type=int, default=8)
  parser.add_argument("--n-epochs", type=int, default=10)
  parser.add_argument("--one2all", dest='one2all', action='store_true')
  parser.set_defaults(one2all=False)

  parser.add_argument("--preprocess", dest='preprocess', action='store_true')
  parser.set_defaults(preprocess=False)
  
  args = parser.parse_args()
  main(args)
import PaGraph.data as data
from PaGraph.parallel import SampleDeliver

def main(args):
  coo_adj, vfeat, efeat , ufeat= data.get_graph_data(args.dataset)

  graph = dgl.DGLGraph(coo_adj, readonly=True)
  vfeatures = torch.FloatTensor(vfeat)
  efeatures = torch.FloatTensor(efeat)

  graph_name = os.path.basename(args.dataset)
  vnum = graph.number_of_nodes()
  enum = graph.number_of_edges()
  vfeat_size = vfeat.shape[1]
  efeat_size = efeat.shape[1]

  print('=' * 30)
  print("Graph Name: {}\nNodes Num: {}\tEdges Num: {}\nvFeature Size: {}\neFeature Size: {}"
        .format(graph_name, vnum, enum, vfeat_size, efeat_size)
  )
  print('=' * 30)

  # create server
  g = dgl.contrib.graph_store.create_graph_store_server(
        graph, graph_name,
        'shared_mem', args.num_workers, 
        False)
  
  # calculate norm for gcn  计算GCN的归一化因子
  dgl_g = DGLGraph(graph, readonly=True)

  if args.model == 'gcn':
    dgl_g = DGLGraph(graph, readonly=True)
    # 计算了每个节点的入度，并取其倒数作为归一化因子。因为在GCN中，为了避免度数大的节点对结果产生过大的影响，通常会使用节点的度数的倒数进行归一化
    norm = 1. / dgl_g.in_degrees().float().unsqueeze(1)
    # preprocess 
    if args.preprocess:
      print('Preprocessing features...')
      dgl_g.ndata['norm'] = norm
      dgl_g.ndata['features'] = features
      dgl_g.update_all(fn.copy_src(src='features', out='m'),
                       fn.sum(msg='m', out='preprocess'),
                       lambda node : {'preprocess': node.data['preprocess'] * node.data['norm']})
      features = dgl_g.ndata['preprocess']
    g.ndata['norm'] = norm
    g.ndata['vfeatures'] = vfeatures
    g.edata['efeatures'] = efeatures
    del dgl_g

  elif args.model == 'graphsage':
    if args.preprocess: # for simple preprocessing
      print('preprocessing: warning: jusy copy')
      g.ndata['neigh'] = features
    g.ndata['features'] = features

  # remote sampler 
  if args.sample:
    subgraph = []
    sub_trainnid = []
    for rank in range(args.num_workers):
      subadj, _ = data.get_sub_train_graph(args.dataset, rank, args.num_workers)
      train_nid = data.get_sub_train_nid(args.dataset, rank, args.num_workers)
      subgraph.append(dgl.DGLGraph(subadj, readonly=True))
      sub_trainnid.append(train_nid)
    hops = args.gnn_layers - 1 if args.preprocess else args.gnn_layers
    print('Expected trainer#: {}. Start sampling at server end...'.format(args.num_workers))
    deliver = SampleDeliver(subgraph, sub_trainnid, args.num_neighbors, hops, args.num_workers)
    deliver.async_sample(args.n_epochs, args.batch_size, one2all=args.one2all)
    
  print('start running graph server on dataset: {}'.format(graph_name))
  g.run()




if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='GraphServer')

  parser.add_argument("--dataset", type=str, default=None,
                      help="dataset folder path")
  
  parser.add_argument("--num-workers", type=int, default=1,
                      help="the number of workers")
  
  parser.add_argument("--model", type=str, default="gcn",
                      help="model type for preprocessing")

  # sample options
  parser.add_argument("--sample", dest='sample', action='store_true')
  parser.set_defaults(sample=False)
  parser.add_argument("--num-neighbors", type=int, default=2)
  parser.add_argument("--gnn-layers", type=int, default=2)
  parser.add_argument("--batch-size", type=int, default=6000)
  #parser.add_argument("--num-workers", type=int, default=8)
  parser.add_argument("--n-epochs", type=int, default=10)
  parser.add_argument("--one2all", dest='one2all', action='store_true')
  parser.set_defaults(one2all=False)

  parser.add_argument("--preprocess", dest='preprocess', action='store_true')
  parser.set_defaults(preprocess=False)
  
  args = parser.parse_args()
  main(args)
