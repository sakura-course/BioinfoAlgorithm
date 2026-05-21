"""
基因表达数据加载、基因筛选与细胞内/细胞间通讯边构建
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class GeneExpressionData:
    """
    存储肿瘤基因表达数据的数据类

    Attributes:
        cancer_cell: CancerCell（癌细胞）的基因表达矩阵
                     行=基因名，列=样本/细胞
        tcell: T细胞 的基因表达矩阵
               行=基因名，列=样本/细胞
        tumor_type: 肿瘤类型（如 BreastTumor, ColonTumor, LungTumor）
    """

    cancer_cell: pd.DataFrame
    tcell: pd.DataFrame
    tumor_type: str


@dataclass
class Edge:
    """
    统一的图边

    不区分细胞内/细胞间——两种边共用同一结构。
    节点标识符格式: "基因名/CancerCell" 或 "基因名/TCell"

    Attributes:
        node_a: 端点A的节点ID
        node_b: 端点B的节点ID
        weight: 边权重（代价）
        edge_type: "intracellular" 或 "intercellular"
    """

    node_a: str
    node_b: str
    weight: float
    edge_type: str


def load_gene_expression(tumor_type: str, data_dir: str) -> GeneExpressionData:
    """
    加载指定肿瘤类型的基因表达数据

    从 InputData/{TumorType}_CancerCell_GeneExpression.csv 和
    InputData/{TumorType}_TCell_GeneExpression.csv 读取数据

    Args:
        tumor_type: 肿瘤类型，支持 BreastTumor / ColonTumor / LungTumor
        data_dir: 数据目录路径

    Returns:
        GeneExpressionData 对象，包含癌细胞和T细胞的表达矩阵
    """
    cancer_path = f"{data_dir}/{tumor_type}_CancerCell_GeneExpression.csv"
    tcell_path = f"{data_dir}/{tumor_type}_TCell_GeneExpression.csv"

    cancer_cell = pd.read_csv(cancer_path, index_col=0).astype(np.float32)
    tcell = pd.read_csv(tcell_path, index_col=0).astype(np.float32)

    return GeneExpressionData(
        cancer_cell=cancer_cell,
        tcell=tcell,
        tumor_type=tumor_type,
    )


def filter_top_variance_genes(
    data: GeneExpressionData,
    n_top: int,
) -> tuple[list[str], list[str]]:
    """
    筛选每种细胞类型中方差最大的前 n 个基因

    计算每个基因（行）在所有样本（列）上的方差，
    方差越大表示基因表达差异越大，越可能是感兴趣的基因。
    筛选后方差大的基因作为后续构建细胞内通讯网络的节点。

    Args:
        data: 基因表达数据对象
        n_top: 筛选的基因数量

    Returns:
        tuple[list[str], list[str]]:
            - 第一个列表：CancerCell 中高方差基因列表
            - 第二个列表：TCell 中高方差基因列表
            每个列表内的基因按方差从大到小排序
    """
    # 计算CancerCell每个基因的方差（沿列方向，即跨样本）
    cancer_var = data.cancer_cell.var(axis=1)
    # 取方差最大的前n_top个基因
    cancer_genes = cancer_var.nlargest(n_top).index.tolist()

    # 计算TCell每个基因的方差
    tcell_var = data.tcell.var(axis=1)
    # 取方差最大的前n_top个基因
    tcell_genes = tcell_var.nlargest(n_top).index.tolist()

    return cancer_genes, tcell_genes


def build_intracellular_edges(
    cancer_genes: list[str],
    tcell_genes: list[str],
    data: GeneExpressionData,
) -> list[Edge]:
    """
    构建细胞内通讯边（完全图），填充归一化后的权重

    对 CancerCell 和 TCell 各自的 top1000 高变基因，
    两两计算 Spearman 相关系数，构成各自的无向完全图。

    权重 = 1 - MinMax(|Spearman|)，MinMax 在所有边（两种细胞合并）上全局归一化。
    节点标识符：基因名/CancerCell 或 基因名/TCell。

    Args:
        cancer_genes: CancerCell 高变异基因列表
        tcell_genes: TCell 高变异基因列表
        data: 基因表达数据

    Returns:
        统一格式的 Edge 列表，edge_type 均为 "intracellular"
    """

    def _ranked(expr: pd.DataFrame, genes: list[str]) -> np.ndarray:
        """
        把基因表达矩阵转为秩矩阵（按行排秩），用于后续向量化计算 Spearman 相关

        Spearman 相关 = 对秩做 Pearson 相关，所以只需预先排秩，
        后续用矩阵乘法一次性算出所有基因对的相关系数，避免 O(n²) 的循环

        Args:
            expr: 基因表达矩阵
            genes: 需排秩的基因列表

        Returns:
            归一化后的秩矩阵，每行为单位向量，shape: (n_genes, n_cells)
        """
        sub = expr.loc[genes].values.astype(np.float32)
        ranked = sub.argsort(axis=1).argsort(axis=1).astype(np.float32)
        # 中心化（减去行均值），用于后续点积算相关
        ranked -= ranked.mean(axis=1, keepdims=True)
        # 每行归一化为单位向量
        norms = np.linalg.norm(ranked, axis=1, keepdims=True)
        norms[norms == 0] = 1.0

        return ranked / norms  # shape: (n_genes, n_cells)

    def _pairwise_abs_spearman(expr: pd.DataFrame, genes: list[str]) -> np.ndarray:
        """
        返回上三角的 |Spearman| 值

        Args:
            expr: 基因表达矩阵
            genes: 基因列表

        Returns:
            |Spearman| 值数组，shape: (C(n,2),)
        """
        normed = _ranked(expr, genes)
        corr = (
            normed @ normed.T
        )  # shape: (n_genes, n_cells) @ (n_cells, n_genes) -> (n_genes, n_genes)，值域 [-1, 1]
        n = len(genes)
        i_idx, j_idx = np.triu_indices(n, k=1)

        return np.abs(corr[i_idx, j_idx])  # shape: (C(n,2),)

    cancer_abs = _pairwise_abs_spearman(data.cancer_cell, cancer_genes)
    tcell_abs = _pairwise_abs_spearman(data.tcell, tcell_genes)

    # 全局 MinMax：两种细胞的所有边合并归一化，保证尺度一致
    all_vals = np.concatenate([cancer_abs, tcell_abs])
    mn, mx = all_vals.min(), all_vals.max()
    if mx == mn:
        mx = mn + 1e-8

    cancer_norm = (cancer_abs - mn) / (mx - mn)
    tcell_norm = (tcell_abs - mn) / (mx - mn)

    # 构建边列表
    edges: list[Edge] = []

    n_c = len(cancer_genes)
    ci, cj = np.triu_indices(n_c, k=1)
    for k, (i, j) in enumerate(zip(ci, cj)):
        edges.append(
            Edge(
                node_a=f"{cancer_genes[i]}/CancerCell",
                node_b=f"{cancer_genes[j]}/CancerCell",
                weight=float(1.0 - cancer_norm[k]),
                edge_type="intracellular",
            )
        )

    n_t = len(tcell_genes)
    ti, tj = np.triu_indices(n_t, k=1)
    for k, (i, j) in enumerate(zip(ti, tj)):
        edges.append(
            Edge(
                node_a=f"{tcell_genes[i]}/TCell",
                node_b=f"{tcell_genes[j]}/TCell",
                weight=float(1.0 - tcell_norm[k]),
                edge_type="intracellular",
            )
        )

    return edges


def load_ligand_receptor(filepath: str) -> list[tuple[str, str]]:
    """
    读取配体受体相互作用数据

    文件格式：每行一个配体-受体对，第一列为配体(Ligand)，第二列为受体(Receptor)

    Args:
        filepath: 文件路径

    Returns:
        list[tuple[str, str]]: 配体-受体对列表，如 [("ICOSLG", "ICOS"), ...]
    """
    lr_pairs = []
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                lr_pairs.append((parts[0], parts[1]))

    return lr_pairs


def build_intercellular_edges(
    lr_pairs: list[tuple[str, str]],
    cancer_genes: list[str],
    tcell_genes: list[str],
    data: GeneExpressionData,
) -> list[Edge]:
    """
    构建跨细胞通讯边（配体-受体对）

    对每个配体-受体对，分别检查两个通讯方向：
      - 配体在 CancerCell 中 + 受体在 TCell 中 → cancer→tcell 边
      - 配体在 TCell 中 + 受体在 CancerCell 中 → tcell→cancer 边
    两个方向独立判断，每个 LR 对可能产生 0~2 条边。

    权重 = 1 - MinMax(配体均值 × 受体均值)，MinMax 在所有 intercellular 边上全局归一化。

    Args:
        lr_pairs: 所有配体-受体对列表
        cancer_genes: CancerCell 的 top1000 高变异基因列表
        tcell_genes: TCell 的 top1000 高变异基因列表
        data: 基因表达数据

    Returns:
        统一格式的 Edge 列表，edge_type 均为 "intercellular"
    """
    cancer_set = set(cancer_genes)
    tcell_set = set(tcell_genes)

    raw_weights: list[float] = []
    raw_edges: list[tuple[str, str]] = []

    for ligand, receptor in lr_pairs:
        if ligand in cancer_set and receptor in tcell_set:
            lig_mean = data.cancer_cell.loc[ligand].mean()
            rec_mean = data.tcell.loc[receptor].mean()
            raw_weights.append(float(lig_mean * rec_mean))
            raw_edges.append(
                (
                    f"{ligand}/CancerCell",
                    f"{receptor}/TCell",
                )
            )
        if ligand in tcell_set and receptor in cancer_set:
            lig_mean = data.tcell.loc[ligand].mean()
            rec_mean = data.cancer_cell.loc[receptor].mean()
            raw_weights.append(float(lig_mean * rec_mean))
            raw_edges.append(
                (
                    f"{ligand}/TCell",
                    f"{receptor}/CancerCell",
                )
            )

    if not raw_weights:
        return []

    arr = np.array(raw_weights, dtype=np.float32)
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        mx = mn + 1e-8
    normed = (arr - mn) / (mx - mn)

    return [
        Edge(
            node_a=node_a,
            node_b=node_b,
            weight=float(1.0 - n),
            edge_type="intercellular",
        )
        for (node_a, node_b), n in zip(raw_edges, normed)
    ]
