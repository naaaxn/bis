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
        batch_size = u.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
       
        # change to [batch_size, 1]
        u = u.view((-1, 1))   #张量
        train_nids = train_nids.view((-1, 1))

        user_embeddings = self.usr(u).squeeze(dim = 1).to(self.device)
        
        entity_emb_vectors = dict()
        for i in range(3):
            batch_nids = nf.layer_parent_nid(2-i).numpy()
            entity_emb = nf.layers[2-i].data['vfeatures']
            for i in range(len(batch_nids)):
                if batch_nids[i] not in entity_emb_vectors:
                    entity_emb_vectors[batch_nids[i]] = entity_emb[i]            
           
        entities, relations = self._get_neighbors(train_nids)        
        
        item_embeddings = self._aggregate(user_embeddings, entities, relations, entity_emb_vectors)       
        
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
        entity_vectors = []
        for entity in entities: 
            entity_ve = []          
            keys_list = entity.tolist()
            for keys in keys_list:
                entity_vector = []
                for key in keys:                
                    entity_vector.append(entity_emb_vectors.get(key, None).tolist())
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