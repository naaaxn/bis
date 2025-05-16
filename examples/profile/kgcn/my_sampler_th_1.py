import torch
import dgl
from dgl import NodeFlow
import ctypes
import numpy as np
import random
import sys
import threading
from numbers import Integral
import traceback
from dgl import utils
from dgl import NodeFlow
from dgl import backend as F
import queue
import time
import scipy.sparse as sp
from dgl import DGLGraph


indptr_array = None
indices_array = None
edge_data_array = None
max_node_id = 0


class OFOBJ():
    def __init__(self, graph ,node_mapping ,edge_mapping,layer_offsets,block_offsets):
        self.graph = graph
        self.node_mapping=node_mapping
        self.edge_mapping=edge_mapping
        self.layer_offsets=layer_offsets
        self.block_offsets=block_offsets
        

def avxth_neighbor_sampler(g, batch_size, num_neighbors,num_worker,current_nodeflow_index,seed_nodes, num_hops, neighbor_type='out', shuffle=False):
    """
    函数版的 AVXthNeighborSampler 类。
    
    参数:
    - g: 图对象
    - batch_size: 每次采样的种子节点数量
    - num_neighbors: 每个节点的邻居采样数量
    - seed_nodes: 初始种子节点
    - num_hops: 采样的跳数
    - neighbor_type: 邻居类型，'in' 或 'out'（默认 'out'）
    - shuffle: 是否打乱种子节点顺序（默认 False）
    
    返回:
    - 生成器，返回 (batch_seed_nodes, seed_nodes, all_result)
    """
    def construct_layer_off_data(layer_sizes):
        # 初始化 layer_off_data，长度比 layer_sizes 多 1，因为需要存储最后的偏移量
        num_layers = len(layer_sizes)
        layer_off_data = [0] * (num_layers + 1)
        
        # 从最外层到最内层，倒序计算
        for i in range(1, num_layers + 1):
            # layer_off_data[i] 是前一层的偏移量 + 前一层的节点数
            layer_off_data[i] = layer_off_data[i - 1] + layer_sizes[-i]
        
        return layer_off_data
    
    def get_noddmapping(index,node_mapping,temp_layer_offsets):
        if index==0:
            return node_mapping[0:temp_layer_offsets[index]]
        else:
            return node_mapping[temp_layer_offsets[index-1]:temp_layer_offsets[index]]

    
    def sample_neighbors(g, nodes, num_neighbors, neighbor_type,node_mapping,map):
        neighbors = []
        node_neighbors = []
        num_neighbors_list = []
        temp_sampled_neighbors=[]
        num_samples = []
        not_need_sample_neighbors=[]
        not_need_sample_flatten_neighbors=[]
        for node in nodes:
            if neighbor_type == 'in':
                temp_node_neighbors = list(set(g.predecessors(node).numpy().tolist()))
                temp_node_neighbors=[item for item in temp_node_neighbors if map[item]==0]
            elif neighbor_type == 'out':
                temp_node_neighbors = list(set(g.successors(node).numpy().tolist()))
                temp_node_neighbors=[item for item in temp_node_neighbors if map[item]==0]
            else:
                raise ValueError("neighbor_type should be either 'in' or 'out'")
            
            n = len(temp_node_neighbors)
            if n<=num_neighbors:
               for item in temp_node_neighbors:
                   map[item]=1
               not_need_sample_neighbors.append(temp_node_neighbors)
               not_need_sample_flatten_neighbors.extend(temp_node_neighbors)
            else:
               num_neighbors_list.append(n)
               node_neighbors.extend(temp_node_neighbors)
               num_samples.append(num_neighbors)      
        neighbors_array = np.array(node_neighbors, dtype=np.int32)
        num_neighbors_array = np.array(num_neighbors_list, dtype=np.int32)
        num_samples_array = np.array(num_samples, dtype=np.int32)
        sampled_neighbors_array = np.zeros(sum(num_samples), dtype=np.int32)
        if node_neighbors!=[]:
            dll = ctypes.cdll.LoadLibrary('.//examples/profile/kgcn/test_th.so')
            dll.avx_th_neighbor_sampling.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
            dll.avx_th_neighbor_sampling(
                neighbors_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                num_neighbors_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),      
                ctypes.c_int(len(num_neighbors_list)), 
                sampled_neighbors_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                num_samples_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            )
            
            temp_sampled_neighbors = sampled_neighbors_array.tolist()
            start = 0
            for size in num_samples:
                neighbors.append(temp_sampled_neighbors[start:start + size])
                start += size
            neighbors.append(temp_sampled_neighbors[start:])
            for item in temp_sampled_neighbors:
                map[item]=1
        neighbors+=not_need_sample_neighbors
        temp_sampled_neighbors.extend(not_need_sample_flatten_neighbors)
        return neighbors,temp_sampled_neighbors

    ofobjs=[]
    for ii in range(0,num_worker):
        max_node_id = g.nodes().max().item()
        map= np.zeros(max_node_id+1, dtype=np.int32)
        node_mapping=[]
        edge_mapping=[]
        layer_offsets=temp_layer_offsets=[]
        block_offsets=temp_block_offsets=[]
        start=(current_nodeflow_index+ii)*batch_size
        batch_seed_nodes = seed_nodes[start:min(start+batch_size,len(seed_nodes))]
        temp_seed_nodes = batch_seed_nodes
        for item in temp_seed_nodes:
            map[item]=1
        node_mapping.extend(temp_seed_nodes)
        temp_layer_offsets.append(len(temp_seed_nodes))
        if temp_seed_nodes!=[]:
            for j in range(num_hops):
                sampled_neighbors,flattened_list = sample_neighbors(g, temp_seed_nodes, num_neighbors, neighbor_type,node_mapping,map)
                temp_layer_offsets.append(len(flattened_list))
                for i, node in enumerate(temp_seed_nodes):
                    # 获取当前节点和它的邻居
                    if sampled_neighbors[i]!=[]:
                        current_neighbors = sampled_neighbors[i]
                        # 将节点和它的邻居都转换为 torch 张量
                        dst_nodes = torch.tensor([node] * len(current_neighbors))  # 重复节点作为源节点
                        src_nodes = torch.tensor(current_neighbors)  # 邻居作为目标节点

                                    # 提取这些节点之间的边 ID，使用 return_uv=True 获取节点对和边 ID
                        src_uv, dst_uv, edge_ids = g.edge_ids(src_nodes, dst_nodes, return_uv=True)
                        
                        # 处理可能有多条边的情况，只保留第一条边
                        unique_edges = {}
                        for u, v, eid in zip(src_uv.numpy(), dst_uv.numpy(), edge_ids.numpy()):
                            if (u, v) not in unique_edges:  # 如果该节点对尚未处理，保留第一个边 ID
                                unique_edges[(u, v)] = eid

                        # 将选择的边 ID 添加到 edge_mapping
                        edge_mapping.extend(list(unique_edges.values()))
                temp_block_offsets.append(len(flattened_list))
                #temp_seed_nodes = [item for sublist in sampled_neighbors for item in random.sample(sublist, min(1000, len(sublist)))]
                temp_seed_nodes=flattened_list
                node_mapping.extend(flattened_list)
        subgraph = g.subgraph(node_mapping)
        layer_offsets=construct_layer_off_data(temp_layer_offsets)
        block_offsets=construct_layer_off_data(temp_block_offsets)
        node_mapping.reverse()
        edge_mapping.reverse()
        #block_offsets.reverse()
        
def construct_layer_off_data(layer_sizes):
        # 初始化 layer_off_data，长度比 layer_sizes 多 1，因为需要存储最后的偏移量
        num_layers = len(layer_sizes)
        layer_off_data = [0] * (num_layers + 1)
        
        # 从最外层到最内层，倒序计算
        for i in range(1, num_layers + 1):
            # layer_off_data[i] 是前一层的偏移量 + 前一层的节点数
            layer_off_data[i] = layer_off_data[i - 1] + layer_sizes[-i]
        
        return layer_off_data

def prepare_for_sampler(g, batch_size, num_neighbors,num_workers,current_nodeflow_index,seed_nodes, num_hops, neighbor_type='in', shuffle=False):
    ofobjs=[]
    start=(current_nodeflow_index)*batch_size
    if(start>=len(seed_nodes)):
       return ofobjs 
    dll = ctypes.cdll.LoadLibrary('.//examples/profile/kgcn/ctest_th1.so')
    dll.neighbor_sampler.argtypes = [
        ctypes.POINTER(ctypes.c_int),   # indptr
        ctypes.POINTER(ctypes.c_int),   # indices
        ctypes.POINTER(ctypes.c_int),   # data
        ctypes.c_int,                   # indptr_size
        ctypes.c_int,                   # indices_size
        ctypes.c_int,                   # data_size
        ctypes.c_int,                   # batch_size
        ctypes.c_int,                   # num_neighbors
        ctypes.POINTER(ctypes.c_int),   # seed_nodes
        ctypes.c_int,                   # seed_nodes_size
        ctypes.c_int,                   # num_hops
        ctypes.c_int,                   # current_nodeflow_index
        ctypes.POINTER(ctypes.c_int),   # node_mapping
        ctypes.POINTER(ctypes.c_int),   # edge_mapping
        ctypes.POINTER(ctypes.c_int),   # layer_offsets
        ctypes.POINTER(ctypes.c_int),   # block_offsets
        ctypes.POINTER(ctypes.c_int),   # src_node
        ctypes.POINTER(ctypes.c_int),   # dst_node
        # ctypes.POINTER(ctypes.c_int),   # data_out
    ]
    global indptr_array,indices_array,edge_data_array,max_node_id
    if current_nodeflow_index == 0:
        # 假设我们有图 g_csc 并通过 adj_sparse 得到了 indptr, indices, edge_data
        # 获取邻接矩阵，默认为 COO 格式
        adj_coo = g.adjacency_matrix_scipy()

        # 将邻接矩阵转换为 CSC 格式
        adj_csc = adj_coo.tocsc()

        # 获取 CSC 格式的三个数组
        indptr = adj_csc.indptr    # 指向每一列的起始位置
        indices = adj_csc.indices  # 行索引
        edge_data = adj_csc.data   # 边的权重或边的信息（如果有）
        # # 将 numpy 数组转换为 ctypes 数组
        #indptr, indices, edge_data = g.adj_sparse(fmt='csc')
        indptr_array = np.array(indptr, dtype=np.int32)
        indices_array = np.array(indices, dtype=np.int32)
        edge_data_array = np.array(edge_data, dtype=np.int32)
        # 初始化 seed_nodes（假设已经有）
        max_node_id = g.nodes().max().item()
    # 初始化输出参数的数组和大小
    size=len(edge_data_array)
    node_mapping = np.zeros(size+100, dtype=np.int32)  # 假设的初始大小
    edge_mapping = np.zeros(size+100, dtype=np.int32)
    layer_offsets = np.zeros(num_hops+2, dtype=np.int32)
    block_offsets = np.zeros(num_hops+1, dtype=np.int32)
    indptr_out = np.zeros(size+100, dtype=np.int32)  # 假设的初始大小
    indices_out = np.zeros(size+100, dtype=np.int32)
    data_out = np.zeros(len(edge_data_array)+100, dtype=np.int32)  # 假设的初始大小
    seed_nodes = np.array(seed_nodes, dtype=np.int32)
    # 调用 C 函数
    #print(f"inedx为{current_nodeflow_index}")
    dll.neighbor_sampler_th(
        indptr_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        indices_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        edge_data_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(len(indptr_array)),    # indptr_size
        ctypes.c_int(len(indices_array)),   # indices_size
        ctypes.c_int(len(edge_data_array)), # data_size
        ctypes.c_int(batch_size),           # batch_size
        ctypes.c_int(num_neighbors),                   # num_neighbors (假设的值)
        seed_nodes.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(len(seed_nodes)),           # seed_nodes_size
        ctypes.c_int(num_hops),                    # num_hops (假设的跳数)
        ctypes.c_int(current_nodeflow_index),      # current_nodeflow_index
        node_mapping.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        edge_mapping.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        layer_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        block_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        indptr_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        indices_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        data_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    )
    max_node=node_mapping[len(edge_data_array)+6]
    node_mapping_size1 =node_mapping[len(edge_data_array)+5]
    edge_mapping_size1 =edge_mapping[len(edge_data_array)+5]
    node_mapping=node_mapping[0:node_mapping_size1]
    edge_mapping=edge_mapping[0:edge_mapping_size1]
    indptr_out=indptr_out[0:max_node+2]
    indices_out=indices_out[0:edge_mapping_size1]
    if(node_mapping!=[]):
        # num_node=max(max(src_node), max(dst_node))+1
        # shape = (num_node, num_node)
        # sub_adj = scipy.sparse.coo_matrix((data_out, (src_node, dst_node)), shape=shape)
        # subgraph = DGLGraph(sub_adj, readonly=True)
        #subgraph = g.subgraph(node_mapping)
        # subgraph = g.edge_subgraph(edges)
        sparse_matrix = sp.csr_matrix((edge_mapping, indices_out, indptr_out))
        subgraph = dgl.DGLGraph(sparse_matrix, readonly=True)
        ofobjs.append(OFOBJ(subgraph,node_mapping[::-1],edge_mapping[::-1],layer_offsets,block_offsets))
    return ofobjs
    
    
        

        
class SamplerIter(object):
    def __init__(self, sampler):
        super(SamplerIter, self).__init__()
        self._sampler = sampler
        self._batches = []
        self._batch_idx = 0

    def prefetch(self):
        batches = self._sampler.fetch(self._batch_idx)
        self._batches.extend(batches)
        self._batch_idx += len(batches)

    def __next__(self):
        if len(self._batches) == 0:
            self.prefetch()
        if len(self._batches) == 0:
            raise StopIteration
        return self._batches.pop(0)

class PrefetchingWrapper(object):
    """Internal shared prefetcher logic. It can be sub-classed by a Thread-based implementation
    or Process-based implementation."""
    _dataq = None  # Data queue transmits prefetched elements
    _controlq = None  # Control queue to instruct thread / process shutdown
    _errorq = None  # Error queue to transmit exceptions from worker to master

    _checked_start = False  # True once startup has been checkd by _check_start

    def __init__(self, sampler_iter, num_prefetch):
        super(PrefetchingWrapper, self).__init__()
        self.sampler_iter = sampler_iter
        assert num_prefetch > 0, 'Unbounded Prefetcher is unsupported.'
        self.num_prefetch = num_prefetch

    def run(self):
        """Method representing the process activity."""
        # Startup - Master waits for this
        try:
            loader_iter = self.sampler_iter
            self._errorq.put(None)
        except Exception as e:  # pylint: disable=broad-except
            tb = traceback.format_exc()
            self._errorq.put((e, tb))

        while True:
            try:  # Check control queue
                c = self._controlq.get(False)
                if c is None:
                    break
                else:
                    raise RuntimeError('Got unexpected control code {}'.format(repr(c)))
            except queue.Empty:
                pass
            except RuntimeError as e:
                tb = traceback.format_exc()
                self._errorq.put((e, tb))
                self._dataq.put(None)

            try:
                data = next(loader_iter)
                error = None
            except Exception as e:  # pylint: disable=broad-except
                tb = traceback.format_exc()
                error = (e, tb)
                data = None
            finally:
                self._errorq.put(error)
                self._dataq.put(data)

    def __next__(self):
        next_item = self._dataq.get()
        next_error = self._errorq.get()

        if next_error is None:
            return next_item
        else:
            self._controlq.put(None)
            if isinstance(next_error[0], StopIteration):
                raise StopIteration
            else:
                return self._reraise(*next_error)

    def _reraise(self, e, tb):
        print('Reraising exception from Prefetcher', file=sys.stderr)
        print(tb, file=sys.stderr)
        raise e

    def _check_start(self):
        assert not self._checked_start
        self._checked_start = True
        next_error = self._errorq.get(block=True)
        if next_error is not None:
            self._reraise(*next_error)

    def next(self):
        return self.__next__()

class ThreadPrefetchingWrapper(PrefetchingWrapper, threading.Thread):
    """Internal threaded prefetcher."""

    def __init__(self, *args, **kwargs):
        super(ThreadPrefetchingWrapper, self).__init__(*args, **kwargs)
        self._dataq = queue.Queue(self.num_prefetch)
        self._controlq = queue.Queue()
        self._errorq = queue.Queue(self.num_prefetch)
        self.daemon = True
        self.start()
        self._check_start()


class NodeFlowSampler(object):
    '''Base class that generates NodeFlows from a graph.

    Class properties
    ----------------
    immutable_only : bool
        Whether the sampler only works on immutable graphs.
        Subclasses can override this property.
    '''
    immutable_only = False

    def __init__(
            self,
            g,
            batch_size,
            seed_nodes,
            shuffle,
            num_prefetch,
            prefetching_wrapper_class):
        self._g = g
        if self.immutable_only and not g._graph.is_readonly():
            raise NotImplementedError("This loader only support read-only graphs.")

        self._batch_size = int(batch_size)

        if seed_nodes is None:
            self._seed_nodes = F.arange(0, g.number_of_nodes())
        else:
            self._seed_nodes = seed_nodes
        

        if num_prefetch:
            self._prefetching_wrapper_class = prefetching_wrapper_class
        self._num_prefetch = num_prefetch

    def fetch(self, current_nodeflow_index):
        '''
        Method that returns the next "bunch" of NodeFlows.
        Each worker will return a single NodeFlow constructed from a single
        batch.

        Subclasses of NodeFlowSampler should override this method.

        Parameters
        ----------
        current_nodeflow_index : int
            How many NodeFlows the sampler has generated so far.

        Returns
        -------
        list[NodeFlow]
            Next "bunch" of nodeflows to be processed.
        '''
        raise NotImplementedError

    def __iter__(self):
        it = SamplerIter(self)
        if self._num_prefetch:
            return self._prefetching_wrapper_class(it, self._num_prefetch)
        else:
            return it

    @property
    def g(self):
        return self._g

    @property
    def seed_nodes(self):
        return self._seed_nodes

    @property
    def batch_size(self):
        return self._batch_size
    


class NeighborSampler(NodeFlowSampler):
    r'''Create a sampler that samples neighborhood.

    It returns a generator of :class:`~dgl.NodeFlow`. This can be viewed as
    an analogy of *mini-batch training* on graph data -- the given graph represents
    the whole dataset and the returned generator produces mini-batches (in the form
    of :class:`~dgl.NodeFlow` objects).
    
    A NodeFlow grows from sampled nodes. It first samples a set of nodes from the given
    ``seed_nodes`` (or all the nodes if not given), then samples their neighbors
    and extracts the subgraph. If the number of hops is :math:`k(>1)`, the process is repeated
    recursively, with the neighbor nodes just sampled become the new seed nodes.
    The result is a graph we defined as :class:`~dgl.NodeFlow` that contains :math:`k+1`
    layers. The last layer is the initial seed nodes. The sampled neighbor nodes in
    layer :math:`i+1` are in layer :math:`i`. All the edges are from nodes
    in layer :math:`i` to layer :math:`i+1`.

    .. image:: https://data.dgl.ai/tutorial/sampling/NodeFlow.png
    
    As an analogy to mini-batch training, the ``batch_size`` here is equal to the number
    of the initial seed nodes (number of nodes in the last layer).
    The number of nodeflow objects (the number of batches) is calculated by
    ``len(seed_nodes) // batch_size`` (if ``seed_nodes`` is None, then it is equal
    to the set of all nodes in the graph).

    Note: NeighborSampler currently only supprts immutable graphs.

    Parameters
    ----------
    g : DGLGraph
        The DGLGraph where we sample NodeFlows.
    batch_size : int
        The batch size (i.e, the number of nodes in the last layer)
    expand_factor : int
        The number of neighbors sampled from the neighbor list of a vertex.

        Note that no matter how large the expand_factor, the max number of sampled neighbors
        is the neighborhood size.
    num_hops : int, optional
        The number of hops to sample (i.e, the number of layers in the NodeFlow).
        Default: 1
    neighbor_type: str, optional
        Indicates the neighbors on different types of edges.

        * "in": the neighbors on the in-edges.
        * "out": the neighbors on the out-edges.

        Default: "in"
    transition_prob : str, optional
        A 1D tensor containing the (unnormalized) transition probability.

        The probability of a node v being sampled from a neighbor u is proportional to
        the edge weight, normalized by the sum over edge weights grouping by the
        destination node.

        In other words, given a node v, the probability of node u and edge (u, v)
        included in the NodeFlow layer preceding that of v is given by:

        .. math::

           p(u, v) = \frac{w_{u, v}}{\sum_{u', (u', v) \in E} w_{u', v}}

        If neighbor type is "out", then the probability is instead normalized by the sum
        grouping by source node:

        .. math::

           p(v, u) = \frac{w_{v, u}}{\sum_{u', (v, u') \in E} w_{v, u'}}

        If a str is given, the edge weight will be loaded from the edge feature column with
        the same name.  The feature column must be a scalar column in this case.

        Default: None
    seed_nodes : Tensor, optional
        A 1D tensor  list of nodes where we sample NodeFlows from.
        If None, the seed vertices are all the vertices in the graph.
        Default: None
    shuffle : bool, optional
        Indicates the sampled NodeFlows are shuffled. Default: False
    num_workers : int, optional
        The number of worker threads that sample NodeFlows in parallel. Default: 1
    prefetch : bool, optional
        If true, prefetch the samples in the next batch. Default: False
    add_self_loop : bool, optional
        If true, add self loop to the sampled NodeFlow.
        The edge IDs of the self loop edges are -1. Default: False
    '''

    immutable_only = True

    def __init__(
            self,
            g,
            batch_size,
            expand_factor=None,
            num_hops=1,
            neighbor_type='in',
            transition_prob=None,
            seed_nodes=None,
            shuffle=False,
            num_workers=1,
            prefetch=True,
            add_self_loop=False):
        super(NeighborSampler, self).__init__(
                g, batch_size, seed_nodes, shuffle, num_workers * 2 if prefetch else 0,
                ThreadPrefetchingWrapper)

        assert g.is_readonly, "NeighborSampler doesn't support mutable graphs. " + \
                "Please turn it into an immutable graph with DGLGraph.readonly"
        assert isinstance(expand_factor, Integral), 'non-int expand_factor not supported'
 
        self._expand_factor = int(expand_factor)
        self._num_hops = int(num_hops)
        self._add_self_loop = add_self_loop
        self._num_workers = int(num_workers)
        self._neighbor_type = neighbor_type
        self._transition_prob = transition_prob
        

    def fetch(self, current_nodeflow_index):
        if self._transition_prob is None:
            prob = F.tensor([], F.float32)
        elif isinstance(self._transition_prob, str):
            prob = self.g.edata[self._transition_prob]
        else:
            prob = self._transition_prob
        nfobjs = prepare_for_sampler(
            self.g,
            self.batch_size,  # batch size
            self._expand_factor,
            self._num_workers,
            current_nodeflow_index,
            self.seed_nodes,            
            self._num_hops,
            self._neighbor_type,
            False
            )
        nflows = [NodeFlow(self.g, obj) for obj in nfobjs]
        return nflows