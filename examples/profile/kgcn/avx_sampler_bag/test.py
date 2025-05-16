import dgl

# 假设这是原图 g，有两个节点类型 'user' 和 'item'
g = dgl.heterograph({
    ('user', 'rates', 'item'): [(0, 0), (1, 1), (2, 2)],
    ('item', 'rated_by', 'user'): [(0, 0), (1, 1), (2, 2)]
})

canonical_etypes = g.canonical_etypes

# 打印所有边类型的源节点类型和目标节点类型
for src_ntype, etype, dst_ntype in canonical_etypes:
    subgraph = g[etype]
        
    src, dst = subgraph.edges()

    # 将边索引从张量转换为 NumPy 数组
    src = src.numpy()
    dst = dst.numpy()

    # 获取边的数量
    num_edges = len(src)
    print(num_edges)
    print(f"Edge type: {etype}")
    print(f"Source node type: {src_ntype}")
    print(f"Destination node type: {dst_ntype}")

for etype in g.etypes:
    print(f"Edge type: {etype}")