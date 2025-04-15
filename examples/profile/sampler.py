import torch
import dgl

class CustomNeighborSampler:
    def __init__(self, g, batch_size, num_neighbors, neighbor_type='in', shuffle=True, num_workers=0, num_hops=2, seed_nodes=None, prefetch=False):
        self.g = g
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors
        self.neighbor_type = neighbor_type
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.num_hops = num_hops
        self.seed_nodes = seed_nodes
        self.prefetch = prefetch

    def sample_neighbors(self, nodes):
        neighbors = []
        for node in nodes:
            if self.neighbor_type == 'in':
                node_neighbors = self.g.predecessors(node)
            elif self.neighbor_type == 'out':
                node_neighbors = self.g.successors(node)
            
            if len(node_neighbors) > self.num_neighbors:
                sampled_neighbors = torch.randperm(len(node_neighbors))[:self.num_neighbors]
                node_neighbors = node_neighbors[sampled_neighbors]
            neighbors.extend(node_neighbors)
        
        return torch.tensor(neighbors)

    def __iter__(self):
        if self.shuffle:
            self.seed_nodes = self.seed_nodes[torch.randperm(len(self.seed_nodes))]
        
        for i in range(0, len(self.seed_nodes), self.batch_size):
            batch_seed_nodes = self.seed_nodes[i:i+self.batch_size]
            blocks = []
            seed_nodes = batch_seed_nodes

            for _ in range(self.num_hops):
                sampled_neighbors = self.sample_neighbors(seed_nodes)
                block = dgl.to_block(dgl.graph((sampled_neighbors, seed_nodes)), seed_nodes)
                seed_nodes = block.srcdata[dgl.NID]
                blocks.insert(0, block)
            
            yield batch_seed_nodes, seed_nodes, blocks


g = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 4]))
g = dgl.add_self_loop(g)
train_nid = torch.tensor([0, 1, 2, 3, 4])

sampler = CustomNeighborSampler(g, batch_size=2, num_neighbors=2, num_hops=3, seed_nodes=train_nid)

for batch_seed_nodes, seed_nodes, blocks in sampler:
    print("Batch seed nodes:", batch_seed_nodes)
    print("Seed nodes:", seed_nodes)
    print("Blocks:", blocks)