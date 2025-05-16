import sys
import torch
import torch.nn.functional as F
import random
import numpy as np
import copy
from aggregator import Aggregator
import dgl
import time

class KGCN(torch.nn.Module):
    def __init__(self, usr_emb, rel_emb, kg, num_nodes, args, device):
        super(KGCN, self).__init__()
        self.usr = usr_emb
        self.rel = rel_emb
        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.dim = args.dim
        self.n_neighbor = args.neighbor_sample_size
        self.kg = kg
        self.num_ent = num_nodes
        self.device = device
        self.aggregator = Aggregator(self.batch_size, self.dim, args.aggregator)
        self._gen_adj()
        
    def _gen_adj(self):
        '''
        Generate adjacency matrix for entities and relations
        Only cares about fixed number of samples
        '''
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

        
    def forward(self, nf, u, train_nids):
        '''
        input: u, v are batch sized indices for users and items
        u: [batch_size]
        v: [batch_size]
        '''
        # print(u)
        # torch.cuda.synchronize()  # 等gpu操作完
        # start = time.time()

        batch_size = u.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
       
        # change to [batch_size, 1]
        u = u.view((-1, 1))   #张量
        train_nids = train_nids.view((-1, 1))

        user_embeddings = self.usr(u).squeeze(dim = 1).to(self.device)               
         
        
        entities, relations = self._get_neighbors(train_nids)
        
        entity_emb_vectors = dict()
        flag=0
        temp_index=self.n_iter-1
        for ii in range(self.n_iter+1):
            # print(self.device)
            batch_nids = nf.layer_parent_nid(self.n_iter-ii).numpy()
            # 打开文件并以追加模式写入
            if flag:
                test1_array=nf.block_parent_eid(temp_index).numpy()
                temp_index-=1
                with open('nei.txt', 'a') as file:
                    file.write(f"第{self.n_iter-1-temp_index}层边 ")
                    for item in test1_array:
                        file.write(f"{item} ")  # 每个元素换行写入
                    file.write(f"\n该层边数量为：{len(test1_array)} ")
                    file.write("over\n")
            # 打开文件并以追加模式写入
            with open('nei2.txt', 'a') as file:
                file.write(f"第{ii}层 ")
                for item in batch_nids:
                    file.write(f"{item} ")  # 每个元素换行写入
                file.write(f"\n该层节点数量为：{len(batch_nids)} ")
                file.write("over\n")
            flag=1
            if self.device=='cuda:0':
                print("batch_nids: ", batch_nids)
            entity_emb = nf.layers[self.n_iter-ii].data['vfeatures']
            for i in range(len(batch_nids)):
                if batch_nids[i] not in entity_emb_vectors:
                    entity_emb_vectors[batch_nids[i]] = entity_emb[i]
        item_embeddings = self._aggregate(user_embeddings, entities, relations, entity_emb_vectors)       
        
        # torch.cuda.synchronize()  # 等gpu操作完
        # end2 = time.time()
        # print('[聚合]time: {:.4f}s'.format(end2 - end1)) 
       
        scores = (user_embeddings * item_embeddings).sum(dim = 1)            
        
        return torch.sigmoid(scores)
    
    def _get_neighbors(self, v):
        '''
        v is batch sized indices for items
        v: [batch_size, 1]
        '''
        entities = [v.to('cpu')]
        relations = []
        for h in range(self.n_iter):
            neighbor_entities = torch.LongTensor(self.adj_ent[entities[h]]).view((self.batch_size, -1))
            neighbor_relations = torch.LongTensor(self.adj_rel[entities[h]]).view((self.batch_size, -1))
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        entities = [entity.to(self.device) for entity in entities]
        relations = [relation.to(self.device) for relation in relations]
        return entities, relations
    
    def _aggregate(self, user_embeddings, entities, relations, entity_emb_vectors):
        '''
        Make item embeddings by aggregating neighbor vectors
        '''
        # print("entity_emb_vectors: ",entity_emb_vectors)
        # print("entities: ", entities)
        entity_vectors = []
        flag_vector=[]
        for entity in entities: 
            # print("entity: ", entity)  
            entity_ve = []          
            keys_list = entity.tolist()
            for keys in keys_list:
                entity_vector = []
                for key in keys:
                    vector = entity_emb_vectors.get(key, None)
                    if vector is not None:
                        flag_vector=vector.tolist()
                        entity_vector.append(vector.tolist())
                    else:
                        # 对缺失的向量进行处理，比如跳过或使用默认值
                        with open('nei.txt', 'a') as file:
                            file.write("缺失了")
                            file.write(f"{key} ")  # 每个元素换行写入
                        entity_vector.append(flag_vector)              
                    #entity_vector.append(entity_emb_vectors.get(key, None).tolist())
                entity_ve.append(entity_vector)
            entity_vectors.append(torch.tensor(entity_ve).to(self.device))
        
        relation_vectors = [self.rel(relation) for relation in relations]
        
        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                act = torch.tanh
            else:
                act = torch.sigmoid
            
            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                vector = self.aggregator(
                    self_vectors=entity_vectors[hop],
                    neighbor_vectors=entity_vectors[hop + 1].view((self.batch_size, -1, self.n_neighbor, self.dim)),
                    neighbor_relations=relation_vectors[hop].view((self.batch_size, -1, self.n_neighbor, self.dim)),
                    user_embeddings=user_embeddings,
                    act=act)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter
        
        return entity_vectors[0].view((self.batch_size, self.dim))