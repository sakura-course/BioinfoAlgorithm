"""
通讯通路子图可视化与中心性韦恩图绘制
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx

from graph import (
    Edge,
    get_k_hop_subgraph,
    top_betweenness_centrality,
    top_degree_centrality,
)

# CancerCell 节点颜色
CANCER_COLOR = "#E57373"
# TCell 节点颜色
TCELL_COLOR = "#64B5F6"
# Intercellular 边颜色
INTERCELL_EDGE_COLOR = "#D32F2F"
# Intracellular 边颜色
INTRACELL_EDGE_COLOR = "#CFD8DC"


def plot_communication_subgraph(
    G: nx.Graph,
    tumor_type: str,
    seed_edges: list[Edge],
    k: int,
    save_dir: str,
) -> None:
    """
    绘制单个癌症的 MST 中 intercellular 边及其 k 阶邻居子图

    CancerCell 节点红色，TCell 节点蓝色；
    intercellular 边红色粗线，intracellular 边灰色细线。

    Args:
        G: 完整 MST 图
        tumor_type: 癌症类型名
        seed_edges: MST 中的 intercellular 边列表
        k: 邻居展开阶数
        save_dir: 保存目录
    """
    filepath = f"{save_dir}/{tumor_type}_communication.png"

    if not seed_edges:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(
            0.5,
            0.5,
            "No intercellular edges in MST",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.set_title(f"{tumor_type} — Intercellular Communication")
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    sub = get_k_hop_subgraph(G, seed_edges, k=k)

    cancer_nodes = [n for n in sub.nodes() if n.endswith("/CancerCell")]
    tcell_nodes = [n for n in sub.nodes() if n.endswith("/TCell")]
    inter_edges = [
        (u, v)
        for u, v, d in sub.edges(data=True)
        if d.get("edge_type") == "intercellular"
    ]
    intra_edges = [
        (u, v)
        for u, v, d in sub.edges(data=True)
        if d.get("edge_type") != "intercellular"
    ]

    fig, ax = plt.subplots(figsize=(16, 12))
    pos = nx.spring_layout(sub, seed=42, k=1.5, iterations=100)

    nx.draw_networkx_nodes(
        sub,
        pos,
        nodelist=cancer_nodes,
        node_color=CANCER_COLOR,
        node_size=60,
        label="CancerCell",
        ax=ax,
    )
    nx.draw_networkx_nodes(
        sub,
        pos,
        nodelist=tcell_nodes,
        node_color=TCELL_COLOR,
        node_size=60,
        label="TCell",
        ax=ax,
    )
    nx.draw_networkx_edges(
        sub,
        pos,
        edgelist=intra_edges,
        edge_color=INTRACELL_EDGE_COLOR,
        width=0.5,
        ax=ax,
    )
    nx.draw_networkx_edges(
        sub,
        pos,
        edgelist=inter_edges,
        edge_color=INTERCELL_EDGE_COLOR,
        width=2.0,
        ax=ax,
    )

    ax.set_title(
        f"{tumor_type} — Intercellular Edges with {k}-hop Neighbors\n"
        f"({len(seed_edges)} intercellular edges, {sub.number_of_nodes()} nodes)",
        fontsize=14,
    )
    ax.legend(
        handles=[
            mpatches.Patch(color=CANCER_COLOR, label="CancerCell"),
            mpatches.Patch(color=TCELL_COLOR, label="TCell"),
            mpatches.Patch(color=INTERCELL_EDGE_COLOR, label="Intercellular edge"),
            mpatches.Patch(color=INTRACELL_EDGE_COLOR, label="Intracellular edge"),
        ],
        loc="upper right",
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _gene_name(node_id: str) -> str:
    """
    从节点 ID 中提取基因名

    节点 ID 格式为 "基因名/CancerCell" 或 "基因名/TCell"，
    去掉 / 后的后缀，仅保留基因名。

    Args:
        node_id: 节点标识符

    Returns:
        纯基因名
    """
    if "/" in node_id:
        return node_id.split("/")[0]
    return node_id


def _venn_regions(sets: dict[str, set[str]]) -> dict[str, list[str]]:
    """
    将基因分配到韦恩图 7 个区域

    三个集合 A、B、C 产生 8 个组合（含全空），
    取 7 个非空区域。

    Args:
        sets: {癌症名: 基因集合}，需恰好三个癌症

    Returns:
        key 为 "100"/"010"/"001"/"110"/"101"/"011"/"111" 的字典，
        值为该区域内的基因列表
    """
    names = list(sets.keys())
    A, B, C = sets[names[0]], sets[names[1]], sets[names[2]]

    regions = {k: [] for k in ["100", "010", "001", "110", "101", "011", "111"]}

    all_genes = A | B | C
    for gene in sorted(all_genes):
        in_a = gene in A
        in_b = gene in B
        in_c = gene in C
        key = f"{int(in_a)}{int(in_b)}{int(in_c)}"
        if key in regions:
            regions[key].append(gene)

    return regions


def _draw_venn(
    sets: dict[str, set[str]],
    title: str,
    filepath: str,
) -> None:
    """
    绘制三圆韦恩图并标注基因名

    Args:
        sets: {癌症名: 基因集合}
        title: 图表标题
        filepath: 保存路径
    """
    names = list(sets.keys())
    regions = _venn_regions(sets)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect("equal")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3, 4)

    # 三圆颜色
    colors = ["#FFCDD2", "#C8E6C9", "#BBDEFB"]
    # 三圆圆心
    centers = [(-1.2, 0), (1.2, 0), (0, 1.83)]
    # 三圆标签位置
    label_poses = [(-1.7, 0), (1.7, 0), (0, 2.2)]
    radius = 2.0

    for i, name in enumerate(names):
        circle = mpatches.Circle(
            centers[i],
            radius,
            facecolor=colors[i],
            edgecolor="black",
            alpha=0.4,
            linewidth=1.5,
        )
        ax.add_patch(circle)
        label_pos = label_poses[i]
        ax.annotate(name, xy=label_pos, fontsize=12, fontweight="bold", ha="center")

    # 七个区域的文本位置
    text_positions = {
        "100": (-1.5, -1.2),
        "010": (1.5, -1.2),
        "001": (0, 3.5),
        "110": (0, -1.5),
        "101": (-1.0, 1.0),
        "011": (1.0, 1.0),
        "111": (0, 0.6),
    }

    for key, pos in text_positions.items():
        genes = regions[key]
        if not genes:
            continue
        lines = genes[:15]
        if len(genes) > 15:
            lines.append(f"... (+{len(genes) - 15} more)")
        text = "\n".join(lines)
        ax.text(
            pos[0],
            pos[1],
            text,
            fontsize=6,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_venn_centrality(
    cancer_data: dict[str, nx.Graph],
    centrality_func: str,
    top_n: int,
    save_dir: str,
) -> None:
    """
    三种癌症中心性 top N 韦恩图

    Args:
        cancer_data: {癌症名: MST 图}
        centrality_func: "degree" 或 "betweenness"
        top_n: 取前几名
        save_dir: 保存目录
    """
    if centrality_func == "degree":
        label = "Degree"
        filepath = f"{save_dir}/venn_degree_centrality.png"
        calc = top_degree_centrality
    elif centrality_func == "betweenness":
        label = "Betweenness"
        filepath = f"{save_dir}/venn_betweenness_centrality.png"
        calc = top_betweenness_centrality
    else:
        raise ValueError(f"Unknown centrality_func: {centrality_func}")

    sets: dict[str, set[str]] = {}
    for cancer_type, G in cancer_data.items():
        top_nodes = calc(G, n=top_n)
        sets[cancer_type] = {_gene_name(n) for n in top_nodes}

    _draw_venn(sets, f"{label} Centrality Top {top_n}", filepath)
