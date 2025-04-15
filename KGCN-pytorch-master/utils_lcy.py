# 开发时间：2024/1/3 15:05
# import quiver
import torch
import csv
from dataclasses import dataclass
# from dgl import create_block

# This function split the feature data horizontally
# each node's data is partitioned into 'world_size' chunks
# return the partition corresponding to the 'rank'
# Input args:
# rank: [0, world_size - 1]
# Output: feat
def get_local_feat(rank: int, world_size:int, feat: torch.Tensor, padding=True) -> torch.Tensor:
    org_feat_width = feat[0].shape[1]
    # print(rank,feat)
    # print(rank,org_feat_width)
    if org_feat_width==8:
        org_feat_width = feat[0][0].shape[1]
    # print(rank,org_feat_width)

    if padding and org_feat_width % world_size != 0:
        step = int(org_feat_width / world_size)
        pad = world_size - org_feat_width + step * world_size
        padded_width = org_feat_width + pad
        assert(padded_width % world_size == 0)
        step = int(padded_width / world_size)
        start_idx = rank * step
        end_idx = start_idx + step
        local_feat = None
        if rank == world_size - 1:  #处理最后一个进程
            # padding is required for P3 to work correctly
            local_feat = feat[:, start_idx : org_feat_width]
            zeros = torch.zeros((local_feat.shape[0], pad), dtype=local_feat.dtype).to(rank)
            local_feat = torch.concat([local_feat, zeros], dim=1)
        else:
            local_feat = feat[:, start_idx : end_idx]
        return local_feat
    else:
        # 计算出当前进程的起始和结束索引
        step = int(feat.shape[1] / world_size)
        start_idx = rank * step
        end_idx = min(start_idx + step, feat.shape[1])
        if rank == world_size - 1:
            end_idx = feat.shape[1]
        # 从特征中提取出本地特征
        local_feat = feat[:, start_idx : end_idx]
        return local_feat

@dataclass
class RunConfig:
    rank: int = 0
    world_size: int = 1
    # topo: str = "uva"
    feat: str = "uva"
    global_in_feats: int = -1
    local_in_feats: int = -1
    hid_feats: int = 128
    num_classes: int = -1 # output feature size
    batch_size: int = 1024
    total_epoch: int = 30
    save_every: int = 30
    # fanouts: list[int] = None
    log_path: str = "log.csv" # logging output path
    checkpt_path: str = "checkpt.pt" # checkpt path


    topo: str = "uva"
    # feat: str = "uva"
    # global_in_feats: int = -1
    # local_in_feats: int = -1
    # hid_feats: int = 128
    # num_classes: int = -1 # output feature size
    # batch_size: int = 1024
    # total_epoch: int = 30
    # save_every: int = 30
    # fanouts: list[int] = None

    def uva_sample(self) -> bool:
        return self.topo == 'uva'
    
    def uva_feat(self) -> bool:
        return self.feat == 'uva'
    
