import sys
import torch
import torch.nn.functional as F
import random
import numpy as np
import copy
from aggregator import Aggregator
from utils_lcy import *
import pdb

class KGCN(torch.nn.Module):
    def __init__(self, num_user, num_ent, num_rel, kg, args, device):
        super(KGCN, self).__init__()
        self.num_user = num_user
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.dim = args.dim
        self.n_neighbor = args.neighbor_sample_size
        self.kg = kg
        self.device = device
        self.aggregator = Aggregator(self.batch_size, self.dim, args.aggregator)
        
        self._gen_adj()
            
        self.usr = torch.nn.Embedding(num_user, args.dim)
        self.ent = torch.nn.Embedding(num_ent, args.dim)
        self.rel = torch.nn.Embedding(num_rel, args.dim)
        # pdb.set_trace()
        # print(num_ent)
        # print(self.ent)
        
        
    def _gen_adj(self):
        '''
        Generate adjacency matrix for entities and relations
        Only cares about fixed number of samples
        '''
        #self.adj_ent = torch.empty(self.num_ent, self.n_neighbor, dtype=torch.long).to('cuda')
        #self.adj_rel = torch.empty(self.num_ent, self.n_neighbor, dtype=torch.long).to('cuda')
        self.adj_ent = torch.empty(self.num_ent, self.n_neighbor, dtype=torch.long)
        self.adj_rel = torch.empty(self.num_ent, self.n_neighbor, dtype=torch.long)
        # print(self.adj_ent.device)   #cpu
        for e in self.kg:
            # print(e)
            if len(self.kg[e]) >= self.n_neighbor:
                neighbors = random.sample(self.kg[e], self.n_neighbor)
            else:
                neighbors = random.choices(self.kg[e], k=self.n_neighbor)   
            self.adj_ent[e] = torch.LongTensor([ent for _, ent in neighbors])
            self.adj_rel[e] = torch.LongTensor([rel for rel, _ in neighbors])
            # print(self.adj_ent[e])    #tensor([5213, 5213, 5213, 5213, 5213, 5213, 5213, 5213])

        
    def forward(self, u, v):
        '''
        input: u, v are batch sized indices for users and items
        u: [batch_size]
        v: [batch_size]
        '''
        batch_size = u.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        # change to [batch_size, 1]
        u = u.view((-1, 1))   #张量
        v = v.view((-1, 1))   #张量
        
        # [batch_size, dim]
        user_embeddings = self.usr(u).squeeze(dim = 1)
        
        entities, relations = self._get_neighbors(v)
        
        item_embeddings = self._aggregate(user_embeddings, entities, relations)
        
        scores = (user_embeddings * item_embeddings).sum(dim = 1)
        # print("用户特征向量：")
        # print(user_embeddings.device)
        # print(user_embeddings.shape)
        # print("项目特征向量：")
        # print(item_embeddings)
        # print(entities)
        # print(entities[2].shape)
            
        return torch.sigmoid(scores)
    
    def _get_neighbors(self, v):
        '''
        v is batch sized indices for items
        v: [batch_size, 1]
        '''
        entities = [v.to('cpu')]
        relations = []
        #print(entities[0].device)
        #entities, relations = entities.to(self.device),  relations.to(self.device)
        for h in range(self.n_iter):
            neighbor_entities = torch.LongTensor(self.adj_ent[entities[h]]).view((self.batch_size, -1))
            neighbor_relations = torch.LongTensor(self.adj_rel[entities[h]]).view((self.batch_size, -1))
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        # print(entities)
        # print(relations)
        entities = [entity.to(self.device) for entity in entities]  # 装着8个采样邻居的编号ID，共采样n_iter层
        relations = [relation.to(self.device) for relation in relations]  # 装着8个与采样邻居的关系ID，共采样n_iter层
        # print("实体")
        # print(entities)
        # print("关系")
        # print(relations)
        return entities, relations
    
    # 聚合其自身的向量和邻居的向量
    def _aggregate(self, user_embeddings, entities, relations):
        '''
        Make item embeddings by aggregating neighbor vectors
        '''
        entity_vectors = [self.ent(entity) for entity in entities]  # ID变成了embedding特征值
        relation_vectors = [self.rel(relation) for relation in relations]   # ID变成了embedding特征值
        # pdb.set_trace()
        # print("实体embedding特征")
        # print(entity_vectors)
        # print(entity_vectors[0].shape)  # torch.Size([32, 1, 16])
        # print(entity_vectors[1].shape)
        # print(entity_vectors[2].shape)
        # print(relation_vectors[0].shape)  # torch.Size([32, 8, 16])
        # print(relation_vectors[1].shape)  # torch.Size([32, 64, 16])
        
        
        for i in range(self.n_iter):   # self.n_iter为计算entity特征表示的迭代次数，默认为1
            if i == self.n_iter - 1:
                #双曲正切函数的输出范围为(-1，1)，因此将强负输入映射为负值。
                # 与sigmoid函数不同，仅将接近零的值映射到接近零的输出，这在某种程度上解决了“vanishing gradients”问题。
                act = torch.tanh  
            else:
                act = torch.sigmoid    # 将值映射到0-1之间
            
            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):   # self.n_iter为计算entity特征表示的迭代次数,目前只做一层
                vector = self.aggregator(
                    self_vectors=entity_vectors[hop],
                    neighbor_vectors=entity_vectors[hop + 1].view((self.batch_size, -1, self.n_neighbor, self.dim)),
                    neighbor_relations=relation_vectors[hop].view((self.batch_size, -1, self.n_neighbor, self.dim)),
                    user_embeddings=user_embeddings,
                    act=act)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter
        # print(entity_vectors[0].device)    
        return entity_vectors[0].view((self.batch_size, self.dim))