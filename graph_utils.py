from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Dict

from pathlib import Path

import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from datasets import load_dataset
from matplotlib.colors import to_rgb


# ---- Domain constants ----

ATOM_TYPES: List[str] = ["C", "N", "O", "F", "I", "Cl", "Br"]

DEFAULT_COLOR_MAP: Dict[str, str] = {
    "C": "lightgray",
    "N": "blue",
    "O": "red",
    "F": "green",
    "Cl": "green",
    "Br": "brown",
    "I": "purple",
}

# ---- Color mixing ----

def blend_colors(base_color, highlight_color="gold", alpha=0.33):
    """
    Blend base_color with highlight_color.
    alpha = fraction of highlight_color.
    """
    b = to_rgb(base_color)
    h = to_rgb(highlight_color)
    return tuple((1 - alpha) * b[i] + alpha * h[i] for i in range(3))

# ---- Dataset loading ----

def _default_cache_path(split: str) -> Path:
    # default cache location
    return Path("data") / f"mutag_pyg_{split}.pt"


def load_mutag_pyg(
    split: str = "train",
    *,
    cache_path: Optional[str | Path] = None,
    prefer_cache: bool = True,
    save_to_cache: bool = True,
) -> List[Data]:
    """
    Load MUTAG as a list of PyTorch Geometric Data objects.

    Strategy:
      1) If prefer_cache=True and cache exists -> load from file (offline, fast).
      2) Else try to download from Hugging Face.
      3) Optionally save to cache for next runs.

    Args:
      split: "train" / "test" / etc (whatever HF dataset provides).
      cache_path: path to .pt file (torch.save format). Default: data/mutag_pyg_{split}.pt
      prefer_cache: if True, load from cache when available.
      save_to_cache: if True, save downloaded dataset to cache.

    Returns:
      list[Data]
    """
    cache_file = Path(cache_path) if cache_path is not None else _default_cache_path(split)

    if prefer_cache and cache_file.exists():
        try:
            obj = torch.load(cache_file, map_location="cpu", weights_only=False)
            if not isinstance(obj, list):
                raise TypeError(f"Cache file {cache_file} does not contain a list.")
            # very light validation
            if len(obj) > 0 and not isinstance(obj[0], Data):
                raise TypeError(f"Cache file {cache_file} list items are not PyG Data objects.")
            return obj
        except Exception as e:
            # If cache is corrupted or from old version, log warning and try to download
            print(f"[WARN] Failed to load cache '{cache_file}': {e}. Will try to download...")

    # Fallback: download from HF
    try:
        dataset_hf = load_dataset("graphs-datasets/MUTAG")
    except Exception as e:
        raise RuntimeError(
            "Could not download MUTAG from Hugging Face and no valid cache file was found.\n"
            f"Cache expected at: {cache_file.resolve()}\n"
            "Fix options:\n"
            "  - Connect to internet once (hotspot) and rerun; it will cache for later.\n"
            "  - Or manually copy a prepared cache file to that location.\n"
            f"Original error: {e}"
        ) from e

    if split not in dataset_hf:
        raise ValueError(f"Split '{split}' not found. Available splits: {list(dataset_hf.keys())}")

    data_list: List[Data] = []
    for g in dataset_hf[split]:
        edge_index = torch.tensor(g["edge_index"], dtype=torch.long)
        x = torch.tensor(g["node_feat"], dtype=torch.float)
        y = torch.tensor(g["y"], dtype=torch.long)

        # (optional) Make sure y is 0-dim tensor if single value
        if y.ndim > 0:
            y = y.view(-1)
            if y.numel() == 1:
                y = y[0]

        data_list.append(Data(x=x, edge_index=edge_index, y=y))

    if save_to_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data_list, cache_file)

    return data_list

# ---- Conversions ----

def pyg_to_nx(
    data: Data,
    *,
    with_atom_labels: bool = False,
    atom_types: Sequence[str] = ATOM_TYPES
) -> nx.Graph:
    """
    Convert a PyG graph to an undirected NetworkX graph.
    If with_atom_labels=True, adds node attribute 'atom' based on one-hot x.
    """
    G = nx.Graph()
    edge_index = data.edge_index.detach().cpu().numpy()

    # nodes
    for i in range(int(data.num_nodes)):
        if with_atom_labels and getattr(data, "x", None) is not None:
            atom_idx = int(data.x[i].argmax().item())
            atom = atom_types[atom_idx] if 0 <= atom_idx < len(atom_types) else "?"
            G.add_node(i, atom=atom)
        else:
            G.add_node(i)

    # edges
    for u, v in edge_index.T:
        G.add_edge(int(u), int(v))

    return G


# ---- Visualization ----

def mutag_class_name(y_value: int) -> str:
    return "Mutagenic" if int(y_value) == 1 else "Non-mutagenic"


def draw_molecules_grid(
    data_list: Sequence[Data],
    *,
    n_rows: int,
    n_cols: int,
    atom_types: Sequence[str] = ATOM_TYPES,
    color_map: Optional[Dict[str, str]] = None,
    highlight_nodes: Optional[Dict[int, set[int]]] = None,
    # optional (seed for layout reproducibility)
    seed: int = None,
    title_prefix: str = "Molecule",
    show: bool = True,
    save_path: Optional[str] = None
) -> None:
    if color_map is None:
        color_map = DEFAULT_COLOR_MAP

    if highlight_nodes is None:
        highlight_nodes = {}

    n = min(len(data_list), n_rows * n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i in range(n_rows * n_cols):
        ax = axes[i]
        ax.axis("off")

        if i >= n:
            continue

        data = data_list[i]
        G = pyg_to_nx(data, with_atom_labels=True, atom_types=atom_types)

        labels = nx.get_node_attributes(G, "atom")
        pos = nx.spring_layout(G, seed=seed)

        matched = highlight_nodes.get(i, set())

        node_colors = []
        node_sizes = []
        node_borders = []
        
        for node in G.nodes():
            base_color = color_map.get(G.nodes[node]["atom"], "lightgray")

            if node in matched:
                # (optional) blended color for highlighting 
                # blended = blend_colors(base_color, "gold", alpha=0.33)
                node_colors.append(base_color)
                node_sizes.append(900)
                node_borders.append("black") 
            else:
                node_colors.append(base_color)
                node_sizes.append(500)
                node_borders.append("black") 

        nx.draw(
            G,
            pos=pos,
            ax=ax,
            with_labels=True,
            labels=labels,
            node_color=node_colors,
            node_size=node_sizes,
            edge_color="black",
            linewidths=2,
        )

        # draw node borders separately
        matched_node_colors = [node_colors[i] for i in matched]
        nx.draw_networkx_nodes(
            G,
            pos=pos,
            ax=ax,
            nodelist=list(matched),
            node_color=matched_node_colors,
            node_size=900,
            edgecolors="black",
            linewidths=2,
        )

        # highlight edges fully inside the matched subgraph
        matched_edges = [
            (u, v) for (u, v) in G.edges()
            if u in matched and v in matched
        ]

        if matched_edges:
            nx.draw_networkx_edges(
                G,
                pos=pos,
                ax=ax,
                edgelist=matched_edges,
                edge_color="black",
                width=2,
            )

        cls = mutag_class_name(int(data.y.item()))
        ax.set_title(f"{title_prefix} {i} ({cls})", fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig)


# ---- quick demo ----
if __name__ == "__main__":
    data_list = load_mutag_pyg("train", prefer_cache=True, save_to_cache=True)
    draw_molecules_grid(data_list[:12], n_rows=3, n_cols=4)
