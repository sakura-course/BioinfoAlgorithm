"""
最小生成树计算、networkx 图转换与网络分析
"""

import networkx as nx

from network import Edge


class UnionFind:
    """
    并查集，用于 Kruskal 算法判断节点是否在同一连通分量

    rank 表示树的高度上界，union 时矮树挂到高树下，
    配合 find 的路径压缩，均摊接近 O(1)。

    Attributes:
        parent: 每个节点的父节点索引
        rank: 每个根节点对应树的高度上界
    """

    def __init__(self, n: int) -> None:
        """
        Args:
            n: 节点总数
        """
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """
        查找 x 所在连通分量的根节点，同时做路径压缩

        Args:
            x: 节点索引

        Returns:
            根节点索引
        """
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        """
        合并 x 和 y 所在的两个连通分量

        Args:
            x: 节点索引
            y: 节点索引

        Returns:
            True 表示合并成功（原先不在同一分量），False 表示已在同一分量
        """
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            self.parent[px] = py
        elif self.rank[px] > self.rank[py]:
            self.parent[py] = px
        else:
            self.parent[py] = px
            self.rank[px] += 1
        return True


def compute_mst(edges: list[Edge]) -> list[Edge]:
    """
    用 Kruskal 算法求最小生成树

    按边权重升序排列，依次用并查集选出不形成环的边，
    直到树边数 = 节点数 - 1。
    图不连通时返回森林（边数少于节点数 - 1）。

    Args:
        edges: 边列表，包含所有 intracellular + intercellular 边

    Returns:
        MST 中的边列表，每条边保留原始的 node_a / node_b / weight / edge_type
    """
    node_names: set[str] = set()
    for e in edges:
        node_names.add(e.node_a)
        node_names.add(e.node_b)

    name_to_idx = {name: i for i, name in enumerate(sorted(node_names))}
    uf = UnionFind(len(node_names))

    mst: list[Edge] = []
    for e in sorted(edges, key=lambda e: e.weight):
        if uf.union(name_to_idx[e.node_a], name_to_idx[e.node_b]):
            mst.append(e)
            if len(mst) == len(node_names) - 1:
                break

    return mst


def mst_to_nx(mst_edges: list[Edge]) -> nx.Graph:
    """
    将 MST 转换为 networkx 图

    每条边的 edge_type 属性保留在图中，后续可按类型筛选或着色。

    Args:
        mst_edges: MST 边列表

    Returns:
        networkx.Graph，边属性包含 weight 和 edge_type
    """
    G = nx.Graph()
    for e in mst_edges:
        G.add_edge(e.node_a, e.node_b, weight=e.weight, edge_type=e.edge_type)
    return G


def intercellular_in_mst(mst_edges: list[Edge]) -> list[Edge]:
    """
    从 MST 边列表中筛选出 intercellular 类型的边

    这些边是最终识别的癌细胞-T 细胞通讯桥梁，
    用于步骤 6 的信号通路可视化。

    Args:
        mst_edges: MST 边列表

    Returns:
        intercellular 类型的 Edge 列表
    """
    return [e for e in mst_edges if e.edge_type == "intercellular"]


def get_k_hop_subgraph(
    G: nx.Graph,
    seed_edges: list[Edge],
    k: int = 5,
) -> nx.Graph:
    """
    从种子边出发，向两端展开 k 阶邻居，得到子图

    用于步骤 6：展示 intercellular 边及其向两端延展的 k 阶邻居。

    Args:
        G: 完整 MST 图
        seed_edges: 种子边（通常为 intercellular 边）
        k: 延展阶数，默认 5

    Returns:
        包含种子边邻域的 networkx 子图
    """
    seed_nodes: set[str] = set()
    for e in seed_edges:
        seed_nodes.add(e.node_a)
        seed_nodes.add(e.node_b)

    all_nodes = set(seed_nodes)
    for node in seed_nodes:
        paths = nx.single_source_shortest_path_length(G, node, cutoff=k)
        all_nodes.update(paths.keys())

    return G.subgraph(all_nodes).copy()


def top_degree_centrality(G: nx.Graph, n: int = 50) -> list[str]:
    """
    度中心性前 n 名节点

    Args:
        G: MST 图
        n: 取前几名

    Returns:
        节点 ID 列表，按度中心性从高到低排序
    """
    centrality = nx.degree_centrality(G)
    return sorted(centrality, key=centrality.get, reverse=True)[:n]


def top_betweenness_centrality(G: nx.Graph, n: int = 50) -> list[str]:
    """
    介数中心性前 n 名节点

    Args:
        G: MST 图
        n: 取前几名

    Returns:
        节点 ID 列表，按介数中心性从高到低排序
    """
    centrality = nx.betweenness_centrality(G, normalized=True)
    return sorted(centrality, key=centrality.get, reverse=True)[:n]
