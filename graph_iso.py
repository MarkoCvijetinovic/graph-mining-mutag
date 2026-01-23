from __future__ import annotations
import matplotlib
matplotlib.use("TkAgg") 

from graph_utils import load_mutag_pyg, draw_molecules_grid, pyg_to_nx

from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, List, Set, Optional, Tuple
from torch_geometric.data import Data

import networkx as nx
import random


Node = Hashable
Mapping = Dict[Node, Node]  # pattern_node -> target_node

Pattern = nx.Graph
Target = nx.Graph
Match = Dict[int, int]
Highlight = Set[int]            # target node ids

def _node_label(G: Target, n: Node, label_key: Optional[str]) -> Optional[object]:
    if label_key is None:
        return None
    return G.nodes[n].get(label_key, None)


def _is_compatible_extension(
    P: Pattern,
    G: Target,
    p_node: Node,
    g_node: Node,
    mapping: Mapping,
    *,
    label_key: Optional[str],
) -> bool:
    """
    Check whether extending mapping with p_node -> g_node is a valid partial match:
    - injective (no two pattern nodes map to same target node)
    - labels match (if label_key provided)
    - edge consistency: for every already-mapped neighbor p' of p_node,
      (p_node, p') in P implies (g_node, mapping[p']) in G.
    """
    # injective
    if g_node in mapping.values():
        return False

    # label constraint
    if label_key is not None:
        if _node_label(P, p_node, label_key) != _node_label(G, g_node, label_key):
            return False

    # adjacency consistency (only need to check edges from p_node to already-mapped nodes)
    for p_other, g_other in mapping.items():
        if P.has_edge(p_node, p_other) and not G.has_edge(g_node, g_other):
            return False
    
    return True


def ullmann_first_match(
    P: nx.Graph,
    G: nx.Graph,
    *,
    label_key: Optional[str] = None,
) -> Optional[Mapping]:
    """
    Ullmann-style backtracking for (non-induced) subgraph isomorphism.
    No pruning beyond basic feasibility checks.
    Returns the first found mapping P->G, or None.
    """
    p_nodes = list(P.nodes())
    g_nodes = list(G.nodes())

    def backtrack(i: int, mapping: Mapping) -> Optional[Mapping]:
        if i == len(p_nodes):
            return dict(mapping)

        p = p_nodes[i]

        # Candidate set C: all label-matching node pairs (p -> g) not already used
        for g in g_nodes:
            if _is_compatible_extension(P, G, p, g, mapping, label_key=label_key):
                mapping[p] = g
                res = backtrack(i + 1, mapping)
                if res is not None:
                    return res
                del mapping[p]

        return None

    return backtrack(0, {})


def ullmann_all_matches(
    P: Pattern,
    G: Pattern,
    *,
    label_key: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Mapping]:
    """
    Same as ullmann_first_match, but returns all mappings (up to 'limit' if given).
    WARNING: can explode combinatorially on larger graphs/patterns.
    """
    p_nodes = list(P.nodes())
    g_nodes = list(G.nodes())
    out: List[Mapping] = []

    def backtrack(i: int, mapping: Mapping) -> None:
        nonlocal out
        if limit is not None and len(out) >= limit:
            return

        if i == len(p_nodes):
            out.append(dict(mapping))
            return

        p = p_nodes[i]
        for g in g_nodes:
            if _is_compatible_extension(P, G, p, g, mapping, label_key=label_key):
                mapping[p] = g
                backtrack(i + 1, mapping)
                del mapping[p]

    backtrack(0, {})
    return out

# ---- Common groups in molecules ----

def make_nitro_motif() -> Pattern:
    P = nx.Graph()
    P.add_node(0, atom="N")
    P.add_node(1, atom="O")
    P.add_node(2, atom="O")
    P.add_node(3, atom="C")
    P.add_edges_from([(0, 1), (0, 2), (0, 3)])
    return P

def make_benzene() -> Pattern:
    P = nx.Graph()
    for i in range(6):
        P.add_node(i, atom="C")
    edges = [(i, (i + 1) % 6) for i in range(6)]
    P.add_edges_from(edges)
    return P

def make_fused_benzene() -> Pattern:
    P = nx.Graph()
    for i in range(10):
        P.add_node(i, atom="C")

    edges = [
        # first ring
        (0,1),(1,2),(2,3),(3,4),(4,5),(5,0),
        # second ring shares edge (2,3)
        (2,6),(6,7),(7,8),(8,9),(9,3)
    ]
    P.add_edges_from(edges)
    return P

def make_carbonyl_like() -> Pattern:
    P = nx.Graph()
    P.add_node(0, atom="C")
    P.add_node(1, atom="O")
    P.add_node(2, atom="O")
    P.add_edges_from([(0,1), (0,2)])
    return P

def make_five_ring() -> Pattern:
    P = nx.Graph()
    for i in range(5):
        P.add_node(i, atom="C")
    edges = [(i, (i + 1) % 5) for i in range(5)]
    P.add_edges_from(edges)
    return P

def make_amino() -> Pattern:
    P = nx.Graph()
    P.add_node(0, atom="N")
    P.add_node(1, atom="C")
    P.add_node(2, atom="C")
    P.add_edges_from([(0,1), (0,2)])
    return P

def make_halogen_substitution(atom: str = "Cl") -> Pattern:
    assert atom in {"F", "Cl", "Br", "I"}
    P = nx.Graph()
    P.add_node(0, atom="C")
    P.add_node(1, atom=atom)
    P.add_edge(0,1)
    return P

def make_ring_with_nitro() -> Pattern:
    P = make_benzene()
    base = len(P.nodes)

    # attach nitro group to node 0
    P.add_node(base, atom="N")
    P.add_node(base+1, atom="O")
    P.add_node(base+2, atom="O")

    P.add_edges_from([
        (0, base),
        (base, base+1),
        (base, base+2),
    ])
    return P

def make_random_group() -> Pattern:
    motifs = [
        make_nitro_motif(),
        make_ring_with_nitro(),
        make_five_ring(),
        make_fused_benzene()
    ]

    return random.choice(motifs)

def single_graph_match(
    data_list: List[Data],
    i: int,
    P: Optional[Pattern] = None,
) -> Tuple[Optional[Match], Highlight]:
    """
    Find a single Ullmann subgraph match in data_list[i].

    Returns:
        match: pattern_node -> target_node mapping, or None
        highlight: set of target node ids to highlight
    """
    if P is None:
        P = make_nitro_motif()

    G: Target = pyg_to_nx(data_list[i], with_atom_labels=True)

    match: Optional[Match] = ullmann_first_match(P, G, label_key="atom")

    if match is None:
        return None, set()

    highlight: Highlight = set(match.values())
    return match, highlight


def all_graph_matches(
    data_list: List[Data],
    i: int,
    P: Optional[Pattern] = None,
) -> Tuple[List[Match], Highlight]:
    """
    Find all Ullmann subgraph matches in data_list[i].

    Returns:
        matches: list of pattern_node -> target_node mappings
        highlight: union of all matched target node ids
    """
    if P is None:
        P = make_nitro_motif()

    G: Target = pyg_to_nx(data_list[i], with_atom_labels=True)

    matches: List[Match] = ullmann_all_matches(P, G, label_key="atom")

    highlight: Highlight = set()
    for match in matches:
        highlight.update(match.values())

    return matches, highlight


def collect_highlights(
    data_list: List[Data],
    *,
    P: Optional[Pattern] = None,
    use_all: bool = True,
) -> Dict[int, Highlight]:
    """
    Run subgraph matching on all graphs in data_list
    and collect highlight sets per index.
    """
    highlights: Dict[int, Highlight] = {}

    for i in range(len(data_list)):
        if use_all:
            _, hl = all_graph_matches(data_list, i, P)
        else:
            _, hl = single_graph_match(data_list, i, P)

        if hl:
            highlights[i] = hl

    return highlights

if __name__ == "__main__":
    data_list: List[Data] = load_mutag_pyg("train")

    n_rows = 3
    n_cols = 4
    sample: List[Data] = random.sample(data_list, n_rows * n_cols)

    P = make_random_group()

    highlight_nodes = collect_highlights(
        sample,
        P=P,
        use_all=True,
    )

    draw_molecules_grid(
        sample,
        n_rows=n_rows,
        n_cols=n_cols,
        highlight_nodes=highlight_nodes,
    )