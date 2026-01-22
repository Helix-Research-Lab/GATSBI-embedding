import os
import gc
import gzip
import pickle
import random
import hashlib
import argparse
from collections import defaultdict
from typing import Dict, List

import numpy as np
import networkx as nx


# -----------------------------
# Helpers
# -----------------------------
def _h64(s: str) -> int:
    # deterministic across runs + machines
    return int(hashlib.blake2b(s.encode("utf-8"), digest_size=8).hexdigest(), 16)


def split_bucket_from_key(key: str, seed: int, train_frac: float, val_frac: float) -> str:
    x = _h64(f"{seed}|{key}") / (2**64 - 1)
    if x < train_frac:
        return "train"
    elif x < train_frac + val_frac:
        return "val"
    else:
        return "test"


def open_gz(path: str):
    return gzip.open(path, "wt", encoding="utf-8")


def dump_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


# -----------------------------
# Sequence-similarity clustering
# -----------------------------
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int):
        rx = self.find(x)
        ry = self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1


def build_similarity_clusters(
    seq_matrix: np.ndarray,
    proteins: np.ndarray,
    threshold: float = 0.30,
) -> Dict[str, int]:
    """
    Build clusters of proteins such that any pair with similarity > threshold
    is in the same cluster. Returns a mapping protein_id -> cluster_id.
    """
    n = seq_matrix.shape[0]
    uf = UnionFind(n)

    # union all pairs with similarity > threshold
    # NOTE: this is O(n^2); if n is huge, this will be expensive.
    for i in range(n):
        row = seq_matrix[i]
        # only upper triangle to avoid double work
        for j in range(i + 1, n):
            if row[j] > threshold:
                uf.union(i, j)

    # collect components
    root_to_idx = defaultdict(list)
    for i in range(n):
        r = uf.find(i)
        root_to_idx[r].append(i)

    # assign cluster IDs
    protein_to_cluster = {}
    for cid, (root, idxs) in enumerate(root_to_idx.items()):
        for i in idxs:
            protein_to_cluster[str(proteins[i])] = cid

    return protein_to_cluster


def assign_cluster_splits(
    protein_to_cluster: Dict[str, int],
    seed: int,
    train_frac: float,
    val_frac: float,
) -> Dict[int, str]:
    """
    Assign each cluster to train/val/test using a deterministic hash on the
    sorted protein IDs in that cluster.
    """
    # invert mapping: cluster_id -> list of proteins
    cluster_to_proteins: Dict[int, List[str]] = defaultdict(list)
    for p, c in protein_to_cluster.items():
        cluster_to_proteins[c].append(p)

    cluster_to_bucket: Dict[int, str] = {}
    for cid, plist in cluster_to_proteins.items():
        key = "|".join(sorted(plist))
        b = split_bucket_from_key(key, seed, train_frac, val_frac)
        cluster_to_bucket[cid] = b

    return cluster_to_bucket


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Split a NetworkX graph into train/val/test node and edge splits "
                    "with sequence-similarity and edge-leakage constraints."
    )
    parser.add_argument(
        "--graph_path",
        type=str,
        required=True,
        help="Path to input NetworkX graph (gpickle).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for splits.",
    )
    parser.add_argument(
        "--seq_matrix",
        type=str,
        required=True,
        help="Path to NumPy .npy file containing pairwise sequence similarity matrix.",
    )
    parser.add_argument(
        "--protein_list",
        type=str,
        required=True,
        help="Path to NumPy .npy file containing protein IDs (order matches seq_matrix).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for deterministic hashing.",
    )
    parser.add_argument(
        "--train_frac",
        type=float,
        default=0.70,
        help="Fraction of data for train split.",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.10,
        help="Fraction of data for val split.",
    )
    parser.add_argument(
        "--test_frac",
        type=float,
        default=0.20,
        help="Fraction of data for test split.",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.30,
        help="Maximum allowed sequence similarity between proteins in different splits.",
    )

    args = parser.parse_args()

    assert abs((args.train_frac + args.val_frac + args.test_frac) - 1.0) < 1e-9, \
        "Train/val/test fractions must sum to 1.0"

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # -----------------------------
    # Load graph
    # -----------------------------
    print("Loading graph...")
    with open(args.graph_path, "rb") as f:
        G = pickle.load(f)
    print(f"Loaded {type(G)} nodes={G.number_of_nodes():,} "
          f"edges={G.number_of_edges():,} directed={G.is_directed()}")

    is_multi = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))

    # -----------------------------
    # Load sequence similarity data
    # -----------------------------
    print("Loading sequence similarity matrix and protein list...")
    seq_matrix = np.load(args.seq_matrix)
    proteins = np.load(args.protein_list)

    if seq_matrix.shape[0] != seq_matrix.shape[1]:
        raise ValueError("Sequence similarity matrix must be square.")
    if seq_matrix.shape[0] != len(proteins):
        raise ValueError("Sequence similarity matrix and protein list size mismatch.")

    # Build protein -> cluster mapping (similarity > threshold => same cluster)
    print("Building similarity clusters (threshold = {:.2f})...".format(args.similarity_threshold))
    protein_to_cluster = build_similarity_clusters(
        seq_matrix,
        proteins,
        threshold=args.similarity_threshold,
    )
    print(f"  Found {len(set(protein_to_cluster.values())):,} similarity clusters.")

    # Assign each cluster to a split
    cluster_to_bucket = assign_cluster_splits(
        protein_to_cluster,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )

    # -----------------------------
    # 1) NODE SPLIT (respecting similarity clusters)
    # -----------------------------
    print("Node split: assigning nodes with similarity constraint ...")
    node_lists = {"train": [], "val": [], "test": []}

    for n in G.nodes():
        n_str = str(n)
        if n_str in protein_to_cluster:
            cid = protein_to_cluster[n_str]
            b = cluster_to_bucket[cid]
        else:
            # Node not in protein list: fall back to hash-based split
            b = split_bucket_from_key(n_str, args.seed, args.train_frac, args.val_frac)
        node_lists[b].append(n)

    dump_pickle(node_lists, os.path.join(args.out_dir, "node_split_nodes.pkl"))
    print("  Node counts:", {k: f"{len(v):,}" for k, v in node_lists.items()})

    # Optional: write induced edges for each node split (streaming)
    print("Node split: streaming induced edges -> gz edgelists ...")
    node_set = {k: set(v) for k, v in node_lists.items()}  # sets for fast membership

    node_edge_files = {
        "train": open_gz(os.path.join(args.out_dir, "node_split_train_induced.edgelist.gz")),
        "val":   open_gz(os.path.join(args.out_dir, "node_split_val_induced.edgelist.gz")),
        "test":  open_gz(os.path.join(args.out_dir, "node_split_test_induced.edgelist.gz")),
    }
    node_edge_counts = {"train": 0, "val": 0, "test": 0}

    if is_multi:
        for u, v, k in G.edges(keys=True):
            for b in ("train", "val", "test"):
                if u in node_set[b] and v in node_set[b]:
                    node_edge_files[b].write(f"{u}\t{v}\t{k}\n")
                    node_edge_counts[b] += 1
                    break
    else:
        for u, v in G.edges():
            for b in ("train", "val", "test"):
                if u in node_set[b] and v in node_set[b]:
                    node_edge_files[b].write(f"{u}\t{v}\n")
                    node_edge_counts[b] += 1
                    break

    for f in node_edge_files.values():
        f.close()

    print("  Induced edge counts:", {k: f"{v:,}" for k, v in node_edge_counts.items()})

    # -----------------------------
    # 2) EDGE SPLIT (no leakage across edges of same node pair)
    # -----------------------------
    print("Edge split: streaming edges -> gz edgelists (no pair leakage) ...")
    edge_files = {
        "train": open_gz(os.path.join(args.out_dir, "edge_split_train.edgelist.gz")),
        "val":   open_gz(os.path.join(args.out_dir, "edge_split_val.edgelist.gz")),
        "test":  open_gz(os.path.join(args.out_dir, "edge_split_test.edgelist.gz")),
    }
    edge_counts = {"train": 0, "val": 0, "test": 0}

    # We ensure that all edges between a given unordered node pair (u, v)
    # go to the same split. For multigraphs, this means all parallel edges
    # between u and v share the same bucket.
    pair_to_bucket: Dict[tuple, str] = {}

    if is_multi:
        for u, v, k in G.edges(keys=True):
            # unordered pair key
            pair = (u, v) if u <= v else (v, u)
            if pair not in pair_to_bucket:
                # use pair as key for deterministic split
                pair_key = f"{pair[0]}\t{pair[1]}"
                b = split_bucket_from_key(pair_key, args.seed, args.train_frac, args.val_frac)
                pair_to_bucket[pair] = b
            else:
                b = pair_to_bucket[pair]
            edge_files[b].write(f"{u}\t{v}\t{k}\n")
            edge_counts[b] += 1
    else:
        for u, v in G.edges():
            pair = (u, v) if u <= v else (v, u)
            if pair not in pair_to_bucket:
                pair_key = f"{pair[0]}\t{pair[1]}"
                b = split_bucket_from_key(pair_key, args.seed, args.train_frac, args.val_frac)
                pair_to_bucket[pair] = b
            else:
                b = pair_to_bucket[pair]
            edge_files[b].write(f"{u}\t{v}\n")
            edge_counts[b] += 1

    for f in edge_files.values():
        f.close()

    print("  Edge counts:", {k: f"{v:,}" for k, v in edge_counts.items()})

    # -----------------------------
    # Cleanup
    # -----------------------------
    del node_set
    gc.collect()

    print(f"\nDone. Outputs written to: {args.out_dir}")
    print("Files:")
    print("  edge_split_{train,val,test}.edgelist.gz")
    print("  node_split_nodes.pkl + node_split_*_induced.edgelist.gz")


if __name__ == "__main__":
    main()
