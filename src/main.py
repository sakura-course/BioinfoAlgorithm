"""
主程序：三种癌症的癌细胞-T 细胞通讯分析全流程
"""

import os

import networkx as nx

from graph import (
    compute_mst,
    intercellular_in_mst,
    mst_to_nx,
)
from network import (
    Edge,
    build_intercellular_edges,
    build_intracellular_edges,
    filter_top_variance_genes,
    load_gene_expression,
    load_ligand_receptor,
)
from visualize import (
    plot_communication_subgraph,
    plot_venn_centrality,
)


def pipeline(tumor_type: str) -> tuple[list[Edge], nx.Graph, list[Edge]]:
    """
    单种癌症的完整管线：步骤 1-5

    加载数据 → 筛选高变异基因 → 构建 intracellular 完全图
    → 构建 intercellular 边 → Kruskal MST

    Args:
        tumor_type: 癌症类型名

    Returns:
        mst_edges: MST 边列表
        G: networkx 图
        inter_mst: MST 中 intercellular 边列表
    """
    data = load_gene_expression(tumor_type, "InputData")
    cancer_genes, tcell_genes = filter_top_variance_genes(data, n_top=1000)

    intra_edges = build_intracellular_edges(cancer_genes, tcell_genes, data)
    lr_pairs = load_ligand_receptor("InputData/LigandReceptor_Human.txt")
    inter_edges = build_intercellular_edges(lr_pairs, cancer_genes, tcell_genes, data)

    all_edges = intra_edges + inter_edges
    mst_edges = compute_mst(all_edges)
    G = mst_to_nx(mst_edges)
    inter_mst = intercellular_in_mst(mst_edges)

    print(
        f"[{tumor_type}] {len(all_edges)} edges -> MST {len(mst_edges)} edges, "
        f"{len(inter_mst)} intercellular"
    )

    return mst_edges, G, inter_mst


def main() -> None:
    """
    三种癌症的完整分析流程：步骤 1-8
    """
    cancer_types = ["BreastTumor", "ColonTumor", "LungTumor"]
    results: dict[str, tuple[list[Edge], nx.Graph, list[Edge]]] = {}

    for ct in cancer_types:
        print(f"\n{'=' * 50}")
        print(f"Processing {ct} ...")
        results[ct] = pipeline(ct)

    save_dir = "output"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'=' * 50}")
    print("步骤 6: 通讯通路可视化 ...")
    for ct in cancer_types:
        _, G, inter_mst = results[ct]
        plot_communication_subgraph(G, ct, inter_mst, k=5, save_dir=save_dir)
        print(f"  {ct}_communication.png saved")

    graph_only = {ct: G for ct, (_, G, _) in results.items()}

    print("步骤 7: 度中心性韦恩图 ...")
    plot_venn_centrality(graph_only, centrality_func="degree", top_n=50, save_dir=save_dir)
    print("  venn_degree_centrality.png saved")

    print("步骤 8: 介数中心性韦恩图 ...")
    plot_venn_centrality(graph_only, centrality_func="betweenness", top_n=50, save_dir=save_dir)
    print("  venn_betweenness_centrality.png saved")

    print("\nDone.")


if __name__ == "__main__":
    main()
