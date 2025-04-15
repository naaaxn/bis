# import dgl
# import torch as th
# # 创建一个具有3种节点类型和3种边类型的异构图
# graph_data = {
#    ('drug', 'interacts', 'drug'): (th.tensor([0, 1]), th.tensor([1, 2])),
#    ('drug', 'interacts', 'gene'): (th.tensor([0, 1]), th.tensor([2, 3])),
#    ('drug', 'treats', 'disease'): (th.tensor([1]), th.tensor([2]))
# }
# g = dgl.heterograph(graph_data)
# print(g.num_nodes())

import torch
import torchvision
 
print("Torch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)

import numpy  
print(numpy.__version__)  
print(numpy.__file__)  # 这将显示numpy的安装位置