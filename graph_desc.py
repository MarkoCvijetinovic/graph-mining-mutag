from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import math
import numpy as np
import pandas as pd
import networkx as nx
from torch_geometric.data import Data

import matplotlib
matplotlib.use("TkAgg") 
from graph_utils import load_mutag_pyg, draw_molecules_grid, pyg_to_nx


@dataclass(frozen=True)
class GraphDescriptors:
    # Size & density
    n_nodes: int
    n_edges: int
    density: float

    # Degree statistics
    deg_min: int
    deg_mean: float
    deg_max: int
    deg_var: float
    deg_entropy: float

    # Connectivity
    n_components: int
    largest_cc_size: int

    # Distance-based (largest CC; NaN if undefined)
    avg_shortest_path_len: float
    diameter: float

    # Cycles & clustering
    cyclomatic_number: int
    cycle_basis_count: int
    avg_clustering: float
    transitivity: float


def _safe_float(x: Optional[float]) -> float:
    return float("nan") if x is None else float(x)


def _degree_entropy(degrees: List[int]) -> float:
    """
    Shannon entropy of the degree distribution (over node degrees).
    Uses log2; returns 0 for trivial graphs.
    """
    if not degrees:
        return 0.0
    # histogram of degrees
    vals, counts = np.unique(degrees, return_counts=True)
    p = counts / counts.sum()
    # avoid log(0)
    ent = -np.sum(p * np.log2(p))
    return float(ent)


def compute_graph_descriptors(G: nx.Graph) -> GraphDescriptors:
    """
    Compute a set of topological descriptors for an undirected graph.

    Notes:
    - Distance-based metrics are computed on the largest connected component.
    - For graphs with <2 nodes in the largest CC, avg path length & diameter are NaN.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()

    density = nx.density(G) if n > 1 else 0.0

    degrees = [d for _, d in G.degree()]
    if degrees:
        deg_min = int(min(degrees))
        deg_max = int(max(degrees))
        deg_mean = float(np.mean(degrees))
        deg_var = float(np.var(degrees, ddof=0))
        deg_entropy = _degree_entropy(degrees)
    else:
        deg_min = deg_max = 0
        deg_mean = deg_var = deg_entropy = 0.0

    # Connectivity
    if n == 0:
        n_components = 0
        largest_cc_size = 0
        avg_spl = float("nan")
        diam = float("nan")
    else:
        components = list(nx.connected_components(G))
        n_components = len(components)
        largest_cc = max(components, key=len)
        largest_cc_size = len(largest_cc)

        # Distance-based metrics on largest CC
        if largest_cc_size >= 2:
            H = G.subgraph(largest_cc).copy()
            avg_spl = float(nx.average_shortest_path_length(H))
            diam = float(nx.diameter(H))
        else:
            avg_spl = float("nan")
            diam = float("nan")

    # Cycles / complexity
    # Cyclomatic number = m - n + c for undirected graphs, where c = number of connected components.
    cyclomatic_number = int(m - n + n_components) if n > 0 else 0

    # cycle_basis gives a list of simple cycles in a basis (per connected component)
    # It’s not "number of cycles in graph" (that can be huge), but a meaningful descriptor.
    try:
        cycle_basis_count = len(nx.cycle_basis(G))
    except nx.NetworkXError:
        cycle_basis_count = 0

    # Clustering
    # average_clustering defined for any graph; for 0/1 nodes it returns 0.0
    avg_clustering = float(nx.average_clustering(G)) if n > 1 else 0.0
    transitivity = float(nx.transitivity(G)) if n > 2 else 0.0

    return GraphDescriptors(
        n_nodes=n,
        n_edges=m,
        density=float(density),

        deg_min=deg_min,
        deg_mean=deg_mean,
        deg_max=deg_max,
        deg_var=deg_var,
        deg_entropy=deg_entropy,

        n_components=n_components,
        largest_cc_size=largest_cc_size,

        avg_shortest_path_len=avg_spl,
        diameter=diam,

        cyclomatic_number=cyclomatic_number,
        cycle_basis_count=cycle_basis_count,
        avg_clustering=avg_clustering,
        transitivity=transitivity,
    )


def compute_descriptors_for_dataset(
    data_list: Sequence[Data],
    *,
    with_atom_labels: bool = False,
) -> pd.DataFrame:
    """
    Compute descriptors for a list of PyG graphs.
    Returns a DataFrame with descriptors + label y.
    """
    rows: List[Dict[str, float]] = []

    for i, data in enumerate(data_list):
        G = pyg_to_nx(data, with_atom_labels=with_atom_labels)
        desc = compute_graph_descriptors(G)
        row = asdict(desc)
        row["y"] = int(data.y.item()) if hasattr(data, "y") else -1
        row["index"] = i
        rows.append(row)

    df = pd.DataFrame(rows)

    # Keep a stable column order
    cols = ["index", "y"] + [c for c in df.columns if c not in ("index", "y")]
    return df[cols]

if __name__ == "__main__":
    data_list = load_mutag_pyg("train")
    df = compute_descriptors_for_dataset(data_list[:20])

    print(df.head())
    print(df.describe(include="all"))