import pandas as pd
import numpy as np
import argparse
import random
# from model import KGCN
from data_loader import DataLoader
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os
from torch.distributed import init_process_group, destroy_process_group, barrier
from utils_lcy import *
import gc
import dgl
# from model import KGCN
from KGCN_train_wc import KGCN
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import torch.distributed as dist

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12350"
    # 初始化进程组, rank 表示当前进程的唯一标识符，而 world_size 表示参与分布式训练的总进程数
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


# Dataset class
class KGCNDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        user_id = np.array(self.df.iloc[idx]['userID'])
        item_id = np.array(self.df.iloc[idx]['itemID'])
        label = np.array(self.df.iloc[idx]['label'], dtype=np.float32)
        # print(user_id, 'a', item_id, 'b',label, 'c')
        user_id = np.reshape(user_id,(1,))
        item_id = np.reshape(item_id,(1,))
        label = np.reshape(label,(1,))
        # print(user_id.shape, 'a', item_id.shape, 'b',label.shape, 'c')
        return user_id, item_id, label
    

def reduce_value(value, average=True):
    world_size = dist.get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value
 
    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size
 
        return value

def train(rank: int,      #  当前进程的编号
         world_size: int,    # 总的进程数量
         config: RunConfig,
         data_loader: DataLoader,
         kg,
         args,):
    ddp_setup(rank, world_size)

    # train test split
    df_dataset = data_loader.load_dataset()
    # print(df_dataset)
    x_train, x_test, y_train, y_test = train_test_split(df_dataset, df_dataset['label'], test_size=1 - args.ratio, shuffle=False, random_state=999)
    train_dataset = KGCNDataset(x_train)
    test_dataset = KGCNDataset(x_test)

    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)  
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    assert args.batch_size % world_size == 0
    batch_size_per_GPU = args.batch_size // world_size
    # 这里的batch_size指的是每个进程下的batch_size。也就是说，总batch_size是这里的batch_size再乘以并行数(world_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_per_GPU, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_per_GPU, sampler=test_sampler)

    # ######################################### 训练 #########################################
    # prepare network, loss function, optimizer
    num_user, num_entity, num_relation = data_loader.get_num()
    print("数量：",num_user, num_entity, num_relation)
    user_encoder, entity_encoder, relation_encoder = data_loader.get_encoders()
    device = torch.device(f"cuda:{rank}")
    kgcn = KGCN(num_user, num_entity, num_relation, kg, args, device).to(device)
    net = DDP(kgcn, device_ids=[rank], output_device=rank)
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.l2_weight)
    # print('device: ', device)

    # train
    loss_list = []
    test_loss_list = []
    auc_score_list = []

    torch.cuda.synchronize()  # 等gpu操作完
    start = time.time()
    for epoch in range(args.n_epochs):
        # torch.cuda.synchronize()  # 等gpu操作完
        # start1 = time.time()
        
        # 设置sampler的epoch，DistributedSampler需要这个来维持各个进程之间的相同随机数种子
        train_loader.sampler.set_epoch(config.total_epoch)
        # running_loss = 0.0
        # print(train_loader)
        for i, (user_ids, item_ids, labels) in enumerate(train_loader):
            # print(user_ids," ",item_ids," ",labels)
            user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = net(user_ids, item_ids).to(device)    #传入的是一个批次的数据量，是batch*1维张量
            
            outputs = outputs.reshape(-1,1)
            # print("sddddddddd", outputs, labels)
            loss = criterion(outputs, labels)       
            loss.backward()
           
            loss = reduce_value(loss, average=True)            
            optimizer.step()
            # running_loss += loss.item()
        
        # print train loss per every epoch
        if rank ==0:
            print('[Epoch {}]train_loss: '.format(epoch+1), loss.item())
            # torch.cuda.synchronize()  # 等gpu操作完
            # end1 = time.time()
            # print('[Epoch {}]Time: {:.4f}s'.format(epoch+1, end1 - start1))
        # loss_list.append(running_loss / len(train_loader))
        
        # evaluate per every epoch
        with torch.no_grad():
            test_loss = 0
            total_roc = 0
            test_loader.sampler.set_epoch(config.total_epoch)
            for user_ids, item_ids, labels in test_loader:
                user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
                # print(user_ids, user_ids.type)
                outputs = net(user_ids, item_ids)
                outputs = outputs.reshape(-1,1)
                test_loss = criterion(outputs, labels)
                total_roc += roc_auc_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
                test_loss = reduce_value(test_loss, average=True)
            if rank ==0:
                print('[Epoch {}]test_loss: '.format(epoch+1), test_loss.item())
            test_loss_list.append(test_loss / len(test_loader))
            auc_score_list.append(total_roc / len(test_loader))

    # trainer = KGCNTrainer(config, net, train_loader, test_loader, loc_feat, optimizer, nid_dtype=torch.int32)
    # trainer.train()
    # ########################################## end #########################################
    torch.cuda.synchronize()  # 等gpu操作完
    end = time.time()
    print('Total Time: {:.4f}s'.format(end - start))
    destroy_process_group()

if __name__=="__main__":
    # prepare arguments (hyperparameters)
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
    parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
    parser.add_argument('--n_epochs', type=int, default=20, help='the number of epochs')
    parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
    parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
    parser.add_argument('--n_iter', type=int, default=2, help='number of iterations when computing entity representation')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--ratio', type=float, default=0.8, help='size of training dataset')
    parser.add_argument('--nprocs', default=4, type=int, help='Number of GPUs / processes')
    # parser.add_argument('--save_every', default=150, type=int, help='How often to save a snapshot')
    # parser.add_argument('--hid_feats', default=256, type=int, help='Size of a hidden feature')
    # parser.add_argument('--topo', default="uva", type=str, help='sampling via: uva, gpu, cpu', choices=["cpu", "uva", "gpu"])
    # parser.add_argument('--feat', default="uva", type=str, help='feature extraction via: uva, gpu, cpu', choices=["cpu", "uva", "gpu"])
    # parser.add_argument('--model', default="gat", type=str, help='Model type: sage or gat', choices=['sage', 'gat'])
    # parser.add_argument('--num_heads', default=4, type=int, help='Number of heads for GAT model')
    args = parser.parse_args(['--l2_weight', '1e-4'])
    
    config = RunConfig()
    world_size = min(args.nprocs, torch.cuda.device_count())
    print(f"using {world_size} GPUs")

    # 做一些运行配置
    config.total_epoch = args.n_epochs
    config.batch_size = args.batch_size

    # ######################################对数据进行处理#####################################
    # build dataset and knowledge graph
    # 其中kg三元组头尾都是编码ID，用户项目交互矩阵dataset的user和item都是编码ID，且三元组中的头ID与dataset中item ID对齐
    # 知识图谱三元组（item，关系，其他属性实体），用户项目交互矩阵（用户ID，itemID，label标签）
    data_loader = DataLoader(args.dataset)
    kg = data_loader.load_kg()
    # print(data_loader.df_kg)
    # print(kg[289])

    # 单机多卡，spawn要放入 if __name__=="__main__": 中，不然会引发错误
    # mp.spawn(train, args=(world_size, config, ), nprocs=world_size, daemon=True) 
    mp.spawn(train, args=(world_size, config, data_loader, kg, args, ), nprocs=world_size, daemon=True)           




