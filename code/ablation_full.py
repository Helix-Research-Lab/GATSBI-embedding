#!/usr/bin/env python3
"""
Full ablation study over input graph composition and node-feature initialization.

Graph data sources (3 boolean):
    interaction  -- STRING protein-protein interaction edges
    expression   -- STRING/GEO co-expression edges
    hb           -- HumanBase tissue-specific edges

8 graph configs (power-set of 3 sources):
    esm_only             -- no graph at all; raw feature vectors are the embeddings
    interaction          -- interaction edges only
    expression           -- expression edges only
    hb                   -- HumanBase edges only
    interaction+expr     -- interaction + expression
    interaction+hb       -- interaction + HumanBase
    expression+hb        -- expression + HumanBase
    full                 -- interaction + expression + HumanBase

2 node-feature initialisations:
    esm    -- ESM protein-language-model vectors (1280-d)
    random -- IID Gaussian noise (mean=0, std=0.02, 1280-d)

2 split strategies:
    edge   -- edges partitioned, all nodes visible
    node   -- nodes partitioned, test nodes unseen

For the "esm_only" graph config the GAT is skipped; the raw feature
vectors (ESM or random) are used directly as embeddings.  Split type is
irrelevant for this baseline, so it is evaluated once per init.

Total: 2 baselines + 7 graphs x 2 inits x 2 splits = 30 rows in the table.

Each configuration is evaluated on three downstream tasks:
    1. EC function prediction   (multi-label, 6 classes)
    2. Interaction prediction    (binary)
    3. Pathway set prediction    (binary)

Usage:
    python ablation_full.py                       # run everything
    python ablation_full.py --skip_training       # eval only (requires pre-trained embeddings)
"""

import os, sys, csv, gzip, pickle, random, hashlib, argparse, warnings, shutil, gc
from datetime import datetime
from typing import Dict, Set, List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_undirected

from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ==================================================================
# Paths
# ==================================================================
DATA_ROOT   = "/scratch/groups/rbaltman/gnayar/graph_embed_scratch/data"
GRAPH_DIR   = "/scratch/groups/rbaltman/gnayar/embed_nx_graphs"
ESM_PATH    = os.path.join(DATA_ROOT, "esm_uniprot_vec.pkl")
EC_TSV      = os.path.join(DATA_ROOT, "EC_data.tsv")
BIOGRID     = os.path.join(DATA_ROOT, "biogrid.txt")
PATHWAYS    = os.path.join(DATA_ROOT, "pathway_proteinSet_filtered.pkl")
OUT_DIR     = os.path.join(DATA_ROOT, "ablation_full_results")

# Source graphs
GRAPH_FILES = {
    "interaction": os.path.join(GRAPH_DIR, "string_interaction.gpickle"),
    "expression":  os.path.join(GRAPH_DIR, "string_coexpression.gpickle"),
    "hb":          os.path.join(GRAPH_DIR, "merged_humanBase_singleEdge_small.gpickle"),
}

# 8 graph configs as tuples of source names (empty = no graph)
GRAPH_CONFIGS = [
    ("esm_only",            ()),
    ("interaction",         ("interaction",)),
    ("expression",          ("expression",)),
    ("hb",                  ("hb",)),
    ("interaction+expr",    ("interaction", "expression")),
    ("interaction+hb",      ("interaction", "hb")),
    ("expression+hb",       ("expression", "hb")),
    ("full",                ("interaction", "expression", "hb")),
]

INIT_TYPES  = ["esm", "random"]
SPLIT_TYPES = ["edge", "node"]

# Hyper-parameters  (match existing training scripts)
DIM         = 1280
GAT_EPOCHS  = 50
GAT_LR      = 1e-3
GAT_WD      = 1e-4
POS_BATCH   = 20_000
NEG_RATIO   = 5
NEG_MAX     = 20
EMB_L2      = 1e-7
PROJ_DIM    = 512
OUT_DIM     = 512
HIDDEN      = 512
HEADS1, HEADS2, HEADS3 = 4, 4, 2
DROPOUT     = 0.2

EVAL_EPOCHS_EC  = 20
EVAL_EPOCHS_INT = 20
EVAL_EPOCHS_PW  = 30
EVAL_LR         = 1e-3
EVAL_SEED       = 42

TRAIN_SEED  = 123


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--skip_training", action="store_true",
                   help="Skip GAT training; only run downstream evaluation on existing embeddings")
    p.add_argument("--gat_epochs", type=int, default=GAT_EPOCHS)
    p.add_argument("--configs", nargs="+", default=None,
                   help="Only run these graph config names (default: all)")
    p.add_argument("--inits", nargs="+", default=INIT_TYPES)
    p.add_argument("--splits", nargs="+", default=SPLIT_TYPES)
    return p.parse_args()


# ==================================================================
# Utility
# ==================================================================
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def dump_pkl(obj, path):
    with open(path, "wb") as f: pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pkl(path):
    with open(path, "rb") as f: return pickle.load(f)

def _h64(s):
    return int(hashlib.blake2b(s.encode(), digest_size=8).hexdigest(), 16)

def split_bucket(key, seed, train_frac=0.70, val_frac=0.10):
    x = _h64(f"{seed}|{key}") / (2**64 - 1)
    if x < train_frac: return "train"
    elif x < train_frac + val_frac: return "val"
    return "test"


# ==================================================================
# 1. Graph building
# ==================================================================
def load_source_graph(name):
    path = GRAPH_FILES[name]
    with open(path, "rb") as f:
        G = pickle.load(f)
    print(f"    Loaded {name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def merge_graphs(sources):
    """Merge one or more source graphs into a single nx.Graph."""
    Gs = [load_source_graph(s) for s in sources]
    merged = nx.Graph()
    for G, name in zip(Gs, sources):
        if G.is_multigraph():
            for u, v, _k, d in G.edges(keys=True, data=True):
                if merged.has_edge(u, v):
                    old_w = merged[u][v].get("weight", 0)
                    new_w = d.get("weight", 0)
                    if new_w > old_w:
                        merged[u][v]["weight"] = new_w
                else:
                    merged.add_edge(u, v, weight=d.get("weight", 1.0))
        else:
            for u, v, d in G.edges(data=True):
                if merged.has_edge(u, v):
                    old_w = merged[u][v].get("weight", 0)
                    new_w = d.get("weight", 0)
                    if new_w > old_w:
                        merged[u][v]["weight"] = new_w
                else:
                    merged.add_edge(u, v, weight=d.get("weight", 1.0))
    return merged


# ==================================================================
# 2. Splitting
# ==================================================================
def create_edge_split(G, out_dir, seed=TRAIN_SEED):
    """Deterministic hash-based edge split -> gzipped edgelists."""
    os.makedirs(out_dir, exist_ok=True)
    files = {b: gzip.open(os.path.join(out_dir, f"edge_split_{b}.edgelist.gz"), "wt") for b in ["train","val","test"]}
    counts = {"train":0, "val":0, "test":0}
    for u, v in G.edges():
        pair = (u, v) if u <= v else (v, u)
        b = split_bucket(f"{pair[0]}\t{pair[1]}", seed)
        files[b].write(f"{u}\t{v}\n")
        counts[b] += 1
    for f in files.values(): f.close()
    print(f"    Edge split: {counts}")
    return counts

def create_node_split(G, out_dir, seed=TRAIN_SEED):
    """Deterministic hash-based node split -> pkl + induced edgelists."""
    os.makedirs(out_dir, exist_ok=True)
    node_lists = {"train":[], "val":[], "test":[]}
    for n in G.nodes():
        b = split_bucket(str(n), seed)
        node_lists[b].append(n)
    dump_pkl(node_lists, os.path.join(out_dir, "node_split_nodes.pkl"))

    node_set = {k: set(v) for k, v in node_lists.items()}
    files = {b: gzip.open(os.path.join(out_dir, f"node_split_{b}_induced.edgelist.gz"), "wt") for b in ["train","val","test"]}
    counts = {"train":0, "val":0, "test":0}
    for u, v in G.edges():
        for b in ["train","val","test"]:
            if u in node_set[b] and v in node_set[b]:
                files[b].write(f"{u}\t{v}\n")
                counts[b] += 1
                break
    for f in files.values(): f.close()
    node_counts = {k: len(v) for k, v in node_lists.items()}
    print(f"    Node split: nodes={node_counts}, induced_edges={counts}")
    return node_lists, counts


# ==================================================================
# 3. GAT encoder / decoder  (identical to GATSBI_*_embed.py)
# ==================================================================
class GATEncoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.dropout = DROPOUT
        self.proj = nn.Linear(in_dim, PROJ_DIM)
        self.gat1 = GATConv(PROJ_DIM, HIDDEN, heads=HEADS1, concat=True, dropout=DROPOUT)
        self.gat2 = GATConv(HIDDEN*HEADS1, HIDDEN, heads=HEADS2, concat=True, dropout=DROPOUT)
        self.gat3 = GATConv(HIDDEN*HEADS2, OUT_DIM, heads=HEADS3, concat=False, dropout=DROPOUT)

    def forward(self, x, edge_index):
        x = self.proj(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.gat3(x, edge_index)

class EdgeMLPDecoder(nn.Module):
    def __init__(self, dim, hidden=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4*dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
    def forward(self, z, ei):
        zu, zv = z[ei[0]], z[ei[1]]
        return self.net(torch.cat([zu, zv, zu*zv, (zu-zv).abs()], dim=-1)).squeeze(-1)


# ==================================================================
# 4. Feature initialisation
# ==================================================================
_esm_cache = None

def get_esm_dict():
    global _esm_cache
    if _esm_cache is None:
        print("  Loading ESM vectors (this may take a while)...")
        _esm_cache = load_pkl(ESM_PATH)
        print(f"  Loaded {len(_esm_cache)} ESM vectors")
    return _esm_cache

def build_features(node2id, init_type, dim=DIM, seed=TRAIN_SEED):
    N = len(node2id)
    rng = np.random.default_rng(seed)

    if init_type == "esm":
        esm = get_esm_dict()
        valid = []
        for name in node2id:
            v = esm.get(name)
            if v is not None:
                vv = np.asarray(v, dtype=np.float32)
                if vv.shape == (dim,):
                    valid.append(vv)
        centroid = np.mean(valid, axis=0).astype(np.float32) if valid else np.zeros(dim, np.float32)
        X = np.empty((N, dim), np.float32)
        for name, idx in node2id.items():
            v = esm.get(name)
            if v is not None:
                vv = np.asarray(v, dtype=np.float32)
                if vv.shape == (dim,):
                    X[idx] = vv - centroid
                    continue
            X[idx] = rng.normal(0, 0.02, size=(dim,)).astype(np.float32)
    else:  # random
        X = rng.normal(0, 0.02, size=(N, dim)).astype(np.float32)

    return torch.from_numpy(X)


# ==================================================================
# 5. GAT training  (condensed from GATSBI_{edge,node}_embed.py)
# ==================================================================
def iter_edges_gz(path):
    with gzip.open(path, "rt") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2: yield parts[0], parts[1]

def load_edges(path, node2id):
    src, dst = [], []
    if not os.path.exists(path): return torch.empty((2,0), dtype=torch.long)
    for u, v in iter_edges_gz(path):
        if u in node2id and v in node2id:
            src.append(node2id[u]); dst.append(node2id[v])
    if not src: return torch.empty((2,0), dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)

def build_node2id_from_edges(paths):
    node2id = {}
    for p in paths:
        if not os.path.exists(p): continue
        for u, v in iter_edges_gz(p):
            if u not in node2id: node2id[u] = len(node2id)
            if v not in node2id: node2id[v] = len(node2id)
    return node2id

def build_node2id_from_node_split(pkl_path):
    splits = load_pkl(pkl_path)
    node2id = {}
    for b in ["train","val","test"]:
        for n in splits.get(b, []):
            if n not in node2id: node2id[n] = len(node2id)
    return node2id, splits

def build_obs_set(ei):
    ei = ei.cpu(); src, dst = ei[0].numpy(), ei[1].numpy()
    obs = set()
    for u, v in zip(src, dst):
        a, b = (int(u), int(v)) if u < v else (int(v), int(u))
        if a != b: obs.add((a, b))
    return obs

def degree_sampler(ei, N, alpha=1.0):
    deg = torch.zeros(N, dtype=torch.long)
    if ei.numel() > 0: deg.scatter_add_(0, ei.cpu()[0], torch.ones(ei.size(1), dtype=torch.long))
    p = deg.float().clamp_min(1).pow(alpha); p = (p / p.sum()).numpy()
    return deg.numpy(), p

def sample_neg(rng, src, p_dst, obs, N, neg_ratio, max_tries, dev):
    src_np = src.detach().cpu().numpy().astype(np.int64)
    total = len(src_np) * neg_ratio
    ns = np.repeat(src_np, neg_ratio)
    nd = np.empty(total, np.int64)
    for i in range(total):
        s = int(ns[i]); ok = False
        for _ in range(max_tries):
            t = int(rng.choice(N, p=p_dst))
            if t == s: continue
            a, b = (s, t) if s < t else (t, s)
            if (a, b) in obs: continue
            nd[i] = t; ok = True; break
        if not ok:
            t = int(rng.integers(0, N))
            if t == s: t = (t+1) % N
            nd[i] = t
    return torch.stack([torch.from_numpy(ns).to(dev, torch.long), torch.from_numpy(nd).to(dev, torch.long)])


def train_gat(split_dir, split_type, init_type, out_dir, epochs=GAT_EPOCHS):
    """Train a GAT model and return {protein_name -> 512-d vector}."""
    set_seed(TRAIN_SEED)
    DEV = device()
    os.makedirs(out_dir, exist_ok=True)

    if split_type == "edge":
        train_p = os.path.join(split_dir, "edge_split_train.edgelist.gz")
        val_p   = os.path.join(split_dir, "edge_split_val.edgelist.gz")
        test_p  = os.path.join(split_dir, "edge_split_test.edgelist.gz")
        node2id = build_node2id_from_edges([train_p, val_p, test_p])
    else:
        node2id, node_splits = build_node2id_from_node_split(os.path.join(split_dir, "node_split_nodes.pkl"))
        train_p = os.path.join(split_dir, "node_split_train_induced.edgelist.gz")
        val_p   = os.path.join(split_dir, "node_split_val_induced.edgelist.gz")
        test_p  = os.path.join(split_dir, "node_split_test_induced.edgelist.gz")

    id2node = {i: n for n, i in node2id.items()}
    N = len(node2id)
    print(f"    num_nodes={N}")

    x_init = build_features(node2id, init_type).to(DEV)

    train_ei = to_undirected(load_edges(train_p, node2id), num_nodes=N)
    val_ei   = load_edges(val_p, node2id)
    test_ei  = load_edges(test_p, node2id)
    full_ei  = torch.cat([train_ei, to_undirected(val_ei, num_nodes=N), to_undirected(test_ei, num_nodes=N)], dim=1)

    data = Data(edge_index=train_ei, num_nodes=N).to(DEV)
    pos_pool = train_ei.to(DEV)

    obs = build_obs_set(full_ei)
    deg, p_dst = degree_sampler(train_ei, N)

    if split_type == "node":
        train_mask = np.zeros(N, dtype=bool)
        for n in node_splits.get("train", []):
            if n in node2id: train_mask[node2id[n]] = True
        p_dst_masked = p_dst.copy(); p_dst_masked[~train_mask] = 0.0
        s = p_dst_masked.sum()
        if s > 0: p_dst_masked /= s
        else: p_dst_masked = np.ones(N) / N
        p_dst = p_dst_masked

    rng = np.random.default_rng(TRAIN_SEED)

    node_emb = nn.Embedding(N, DIM).to(DEV)
    with torch.no_grad(): node_emb.weight.copy_(x_init)

    encoder = GATEncoder(DIM).to(DEV)
    decoder = EdgeMLPDecoder(OUT_DIM).to(DEV)

    opt = torch.optim.Adam(list(encoder.parameters()) + list(node_emb.parameters()) + list(decoder.parameters()),
                           lr=GAT_LR, weight_decay=GAT_WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(1, epochs + 1):
        encoder.train(); node_emb.train(); decoder.train()
        opt.zero_grad(set_to_none=True)
        z = encoder(node_emb.weight, data.edge_index)
        E = pos_pool.size(1)
        bp = min(POS_BATCH, E)
        perm = torch.randint(0, E, (bp,), device=DEV)
        pe = pos_pool[:, perm]
        ne = sample_neg(rng, pe[0], p_dst, obs, N, NEG_RATIO, NEG_MAX, DEV)
        pl = decoder(z, pe); nl = decoder(z, ne)
        logits = torch.cat([pl, nl]); labels = torch.cat([torch.ones_like(pl), torch.zeros_like(nl)]) * 0.9 + 0.05
        loss = F.binary_cross_entropy_with_logits(logits, labels) + EMB_L2 * node_emb.weight.norm(p=2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 2.0)
        torch.nn.utils.clip_grad_norm_(node_emb.parameters(), 2.0)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 2.0)
        opt.step(); sched.step()
        if epoch % 10 == 0 or epoch == 1:
            print(f"      epoch {epoch:3d}/{epochs}  loss={loss.item():.4f}")

    encoder.eval(); node_emb.eval()
    with torch.no_grad():
        z = encoder(node_emb.weight, data.edge_index).cpu().numpy().astype(np.float32)
    emb_by_name = {id2node[i]: z[i] for i in range(N)}
    emb_path = os.path.join(out_dir, "embeddings.pkl")
    dump_pkl(emb_by_name, emb_path)
    print(f"    Saved {len(emb_by_name)} embeddings -> {emb_path}")
    return emb_by_name


# ==================================================================
# 6. Baseline embeddings (no graph)
# ==================================================================
def build_baseline_embeddings(init_type):
    """Return raw ESM or random vectors as the 'embedding' dict."""
    esm = get_esm_dict()
    proteins = list(esm.keys())
    if init_type == "esm":
        valid = []
        for p in proteins:
            v = np.asarray(esm[p], dtype=np.float32)
            if v.shape == (DIM,): valid.append((p, v))
        centroid = np.mean([v for _, v in valid], axis=0)
        return {p: (v - centroid) for p, v in valid}
    else:
        rng = np.random.default_rng(TRAIN_SEED)
        return {p: rng.normal(0, 0.02, size=(DIM,)).astype(np.float32) for p in proteins}


# ==================================================================
# 7. Downstream evaluation  (from ablation_study.py)
# ==================================================================
# --- EC ---
def _parse_ec(path):
    df = pd.read_csv(path, sep="\t", dtype=str)
    df["EC number"] = df["EC number"].fillna("")
    df["EC_list"] = df["EC number"].apply(lambda x: [e.strip() for e in x.split(";") if e.strip()])
    return {a: set(e) for a, e in zip(df["Entry"], df["EC_list"])}

class FunctionMLP(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in,512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(512,256), nn.ReLU(), nn.Linear(256, d_out))
    def forward(self, x): return self.net(x)

def eval_ec(emb, seed=EVAL_SEED):
    p2ec = _parse_ec(EC_TSV)
    ec1s = sorted({e.split(".")[0] for es in p2ec.values() for e in es if "." in e})
    e2i = {e:i for i,e in enumerate(ec1s)}
    prots = [p for p, es in p2ec.items() if es and p in emb]
    if len(prots) < 20: return _nan("ec")
    random.seed(seed); random.shuffle(prots)
    n=len(prots); nt=int(.8*n); nv=int(.1*n)
    def mkXY(ps):
        X = np.array([emb[p] for p in ps])
        Y = np.zeros((len(ps), len(ec1s)), np.float32)
        for i, p in enumerate(ps):
            for e in p2ec[p]:
                if "." in e:
                    k = e.split(".")[0]
                    if k in e2i: Y[i, e2i[k]] = 1
        return X, Y
    Xtr,Ytr = mkXY(prots[:nt]); Xv,Yv = mkXY(prots[nt:nt+nv]); Xte,Yte = mkXY(prots[nt+nv:])
    dev = device(); m = FunctionMLP(Xtr.shape[1], len(ec1s)).to(dev)
    Xtr_t=torch.tensor(Xtr).to(dev); Ytr_t=torch.tensor(Ytr).to(dev)
    Xv_t=torch.tensor(Xv).to(dev); Yv_t=torch.tensor(Yv).to(dev)
    pw = ((Ytr_t.shape[0]-Ytr_t.sum(0))/Ytr_t.sum(0).clamp(min=1)).to(dev)
    opt=torch.optim.Adam(m.parameters(),lr=EVAL_LR); lf=nn.BCEWithLogitsLoss(pos_weight=pw)
    best=-1; bs=None
    for _ in range(EVAL_EPOCHS_EC):
        m.train(); opt.zero_grad(); lf(m(Xtr_t),Ytr_t).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vp=torch.sigmoid(m(Xv_t)).cpu().numpy(); vt=Yv_t.cpu().numpy()
            ok=np.where(vt.sum(0)>0)[0]
            va=np.mean([roc_auc_score(vt[:,c],vp[:,c]) for c in ok]) if len(ok) else 0
            if va>best: best=va; bs={k:v.clone() for k,v in m.state_dict().items()}
    if bs: m.load_state_dict(bs)
    m.eval()
    with torch.no_grad(): pr=torch.sigmoid(m(torch.tensor(Xte).to(dev))).cpu().numpy()
    ok=np.where(Yte.sum(0)>0)[0]
    if not len(ok): return _nan("ec")
    pd_ = (pr>=.5).astype(int)
    return {"ec_auc": np.mean([roc_auc_score(Yte[:,c],pr[:,c]) for c in ok]),
            "ec_auprc": np.mean([average_precision_score(Yte[:,c],pr[:,c]) for c in ok]),
            "ec_acc": accuracy_score(Yte,pd_), "ec_prec": precision_score(Yte,pd_,average="micro",zero_division=0),
            "ec_rec": recall_score(Yte,pd_,average="micro",zero_division=0), "ec_f1": f1_score(Yte,pd_,average="micro",zero_division=0)}

# --- Interaction ---
_biogrid_cache = None
def _get_biogrid():
    global _biogrid_cache
    if _biogrid_cache is None:
        def fs(x):
            if pd.isna(x): return None
            x=str(x).strip()
            return x.split("|")[0] if x not in ("","-","nan","None") else None
        df=pd.read_csv(BIOGRID,sep="\t",low_memory=False)
        h=df[(df["Organism ID Interactor A"]==9606)&(df["Organism ID Interactor B"]==9606)].copy()
        a=h["SWISS-PROT Accessions Interactor A"].map(fs); b=h["SWISS-PROT Accessions Interactor B"].map(fs)
        e=pd.DataFrame({"uA":a,"uB":b}).dropna(); e=e[e.uA!=e.uB]
        e["pair"]=e.apply(lambda r:tuple(sorted((r.uA,r.uB))),axis=1); e=e.drop_duplicates("pair").drop(columns=["pair"])
        _biogrid_cache = e[["uA","uB"]].values.tolist()
        print(f"  BioGRID: {len(_biogrid_cache)} edges")
    return _biogrid_cache

class EdgeMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(.3),
                                 nn.Linear(512,256),nn.ReLU(),nn.Linear(256,1))
    def forward(self,x): return self.net(x)

def _neg_sample(edges, nodes, n, seed):
    rng=random.Random(seed); es=set((u,v) for u,v in edges)|set((v,u) for u,v in edges)
    neg=[]
    while len(neg)<n:
        u=rng.choice(nodes); v=rng.choice(nodes)
        if u!=v and (u,v) not in es: neg.append((u,v))
    return neg

def eval_interaction(emb, seed=EVAL_SEED):
    edges=_get_biogrid(); nodes=list(emb.keys())
    random.seed(seed); ec=list(edges); random.shuffle(ec)
    n=len(ec); nt=int(.8*n); nv=int(.1*n)
    tp,vp,sp = ec[:nt],ec[nt:nt+nv],ec[nt+nv:]
    tn=_neg_sample(tp,nodes,len(tp),seed); vn=_neg_sample(vp,nodes,len(vp),seed+1); sn=_neg_sample(sp,nodes,len(sp),seed+2)
    def mk(pos,neg):
        X,y=[],[]
        for u,v in pos:
            if u in emb and v in emb: X.append(np.concatenate([emb[u]*emb[v],np.abs(emb[u]-emb[v]),(emb[u]-emb[v])**2])); y.append(1)
        for u,v in neg:
            if u in emb and v in emb: X.append(np.concatenate([emb[u]*emb[v],np.abs(emb[u]-emb[v]),(emb[u]-emb[v])**2])); y.append(0)
        return np.array(X),np.array(y)
    Xtr,Ytr=mk(tp,tn); Xv,Yv=mk(vp,vn); Xte,Yte=mk(sp,sn)
    if Xtr.shape[0]<10 or Xte.shape[0]<10: return _nan("int")
    dev=device(); m=EdgeMLP(Xtr.shape[1]).to(dev)
    Xt=torch.tensor(Xtr,dtype=torch.float32).to(dev); Yt=torch.tensor(Ytr,dtype=torch.float32).unsqueeze(1).to(dev)
    Xvt=torch.tensor(Xv,dtype=torch.float32).to(dev); Yvt=torch.tensor(Yv,dtype=torch.float32).unsqueeze(1).to(dev)
    opt=torch.optim.Adam(m.parameters(),lr=EVAL_LR); lf=nn.BCEWithLogitsLoss()
    best=-1; bs=None
    for _ in range(EVAL_EPOCHS_INT):
        m.train(); opt.zero_grad(); lf(m(Xt),Yt).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            va=roc_auc_score(Yvt.cpu().numpy().flatten(),torch.sigmoid(m(Xvt)).cpu().numpy().flatten())
            if va>best: best=va; bs={k:v.clone() for k,v in m.state_dict().items()}
    if bs: m.load_state_dict(bs)
    m.eval()
    with torch.no_grad(): pr=torch.sigmoid(m(torch.tensor(Xte,dtype=torch.float32).to(dev))).cpu().numpy().flatten()
    tl=Yte.flatten(); pd_=(pr>=.5).astype(int)
    return {"int_auc":roc_auc_score(tl,pr),"int_auprc":average_precision_score(tl,pr),
            "int_acc":accuracy_score(tl,pd_),"int_prec":precision_score(tl,pd_,zero_division=0),
            "int_rec":recall_score(tl,pd_,zero_division=0),"int_f1":f1_score(tl,pd_,zero_division=0)}

# --- Pathway ---
class MHAPool(nn.Module):
    def __init__(self,d,nh=4,hd=64):
        super().__init__(); self.nh=nh; self.hd=hd; td=nh*hd
        self.qp=nn.Linear(d,td); self.kp=nn.Linear(d,td); self.vp=nn.Linear(d,td); self.op=nn.Linear(td,d)
    def forward(self,x):
        B,K,D=x.shape
        Q=self.qp(x).view(B,K,self.nh,self.hd).transpose(1,2)
        K_=self.kp(x).view(B,K,self.nh,self.hd).transpose(1,2)
        V=self.vp(x).view(B,K,self.nh,self.hd).transpose(1,2)
        w=torch.softmax(Q@K_.transpose(-2,-1)/np.sqrt(self.hd),dim=-1)
        return self.op((w@V).mean(2).reshape(B,self.nh*self.hd))

class SetMLP(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.pool=MHAPool(d)
        self.net=nn.Sequential(nn.Linear(d,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(.3),
                               nn.Linear(512,256),nn.ReLU(),nn.Linear(256,1))
    def forward(self,x): return self.net(self.pool(x))

def eval_pathway(emb, seed=EVAL_SEED):
    pw2p=load_pkl(PATHWAYS); pws=list(pw2p.keys()); K=int(np.median([len(s) for s in pw2p.values()]))
    random.seed(seed); random.shuffle(pws)
    n=len(pws); nt=int(.8*n); nv=int(.1*n)
    trpw,vpw,tpw = pws[:nt],pws[nt:nt+nv],pws[nt+nv:]
    allp=list({p for s in pw2p.values() for p in s if p in emb})
    edim=next(iter(emb.values())).shape[0]
    def bse(ps):
        ps=list(ps)[:K]; X=[emb[p] if p in emb else np.zeros(edim,np.float32) for p in ps]
        while len(X)<K: X.append(np.zeros(edim,np.float32))
        return np.stack(X)
    def corrupt(ps):
        pl=list(ps); k=max(1,int(len(pl)*.4))
        for i in random.sample(range(len(pl)),k): pl[i]=random.choice(allp)
        return set(pl)
    def mkds(subset):
        X,y=[],[]
        for pw in subset:
            s={p for p in pw2p[pw] if p in emb}
            if not s: continue
            X.append(bse(s)); y.append(1)
        for pw in subset:
            s={p for p in pw2p[pw] if p in emb}
            if not s: continue
            X.append(bse(corrupt(s))); y.append(0)
        return np.array(X,np.float32),np.array(y,np.float32)
    Xtr,Ytr=mkds(trpw); Xv,Yv=mkds(vpw); Xte,Yte=mkds(tpw)
    if Xtr.shape[0]<10 or Xte.shape[0]<10: return _nan("pw")
    dev=device(); m=SetMLP(Xtr.shape[2]).to(dev)
    Xt=torch.tensor(Xtr).to(dev); Yt=torch.tensor(Ytr).unsqueeze(1).to(dev)
    Xvt=torch.tensor(Xv).to(dev); Yvt=torch.tensor(Yv).unsqueeze(1).to(dev)
    opt=torch.optim.Adam(m.parameters(),lr=EVAL_LR,weight_decay=1e-4); lf=nn.BCEWithLogitsLoss()
    best=-1; bs=None
    for _ in range(EVAL_EPOCHS_PW):
        m.train(); opt.zero_grad(); lf(m(Xt),Yt).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            va=roc_auc_score(Yvt.cpu().numpy().flatten(),torch.sigmoid(m(Xvt)).cpu().numpy().flatten())
            if va>best: best=va; bs={k:v.clone() for k,v in m.state_dict().items()}
    if bs: m.load_state_dict(bs)
    m.eval()
    with torch.no_grad(): pr=torch.sigmoid(m(torch.tensor(Xte).to(dev))).cpu().numpy().flatten()
    tl=Yte.flatten(); pd_=(pr>=.5).astype(int)
    return {"pw_auc":roc_auc_score(tl,pr),"pw_auprc":average_precision_score(tl,pr),
            "pw_acc":accuracy_score(tl,pd_),"pw_prec":precision_score(tl,pd_,zero_division=0),
            "pw_rec":recall_score(tl,pd_,zero_division=0),"pw_f1":f1_score(tl,pd_,zero_division=0)}

def _nan(prefix):
    return {f"{prefix}_{m}": float("nan") for m in ["auc","auprc","acc","prec","rec","f1"]}


# ==================================================================
# 8. Output formatting
# ==================================================================
MCOLS = ["ec_auc","ec_auprc","ec_acc","ec_prec","ec_rec","ec_f1",
         "int_auc","int_auprc","int_acc","int_prec","int_rec","int_f1",
         "pw_auc","pw_auprc","pw_acc","pw_prec","pw_rec","pw_f1"]

MDISP = {"ec_auc":"EC AUC","ec_auprc":"EC AUPRC","ec_acc":"EC Acc","ec_prec":"EC Prec","ec_rec":"EC Rec","ec_f1":"EC F1",
         "int_auc":"Int AUC","int_auprc":"Int AUPRC","int_acc":"Int Acc","int_prec":"Int Prec","int_rec":"Int Rec","int_f1":"Int F1",
         "pw_auc":"PW AUC","pw_auprc":"PW AUPRC","pw_acc":"PW Acc","pw_prec":"PW Prec","pw_rec":"PW Rec","pw_f1":"PW F1"}

def fmt_table(results):
    lines = []
    for st in SPLIT_TYPES + ["none"]:
        rows = [r for r in results if r["split"] == st]
        if not rows: continue
        lbl = st.upper() if st != "none" else "NO GRAPH (baseline)"
        lines.append(f"\n{'='*130}\n  Split: {lbl}\n{'='*130}")
        for tname, cols in [("EC Function Prediction",["ec_auc","ec_auprc","ec_acc","ec_prec","ec_rec","ec_f1"]),
                            ("Interaction Prediction",["int_auc","int_auprc","int_acc","int_prec","int_rec","int_f1"]),
                            ("Pathway Set Prediction",["pw_auc","pw_auprc","pw_acc","pw_prec","pw_rec","pw_f1"])]:
            hdr = f"{'Config':<25}{'Init':<8}" + "".join(f"{MDISP[c]:>10}" for c in cols)
            lines.append(f"\n  {tname}\n  {'-'*95}\n  {hdr}\n  {'-'*95}")
            for r in rows:
                vals = "".join(f"{r.get(c,float('nan')):>10.4f}" for c in cols)
                lines.append(f"  {r['graph_config']:<25}{r['init']:<8}{vals}")
    return "\n".join(lines)

def save_csv(results, path):
    flds = ["split","graph_config","init"] + MCOLS
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flds); w.writeheader()
        for r in results: w.writerow({k: r.get(k,"") for k in flds})

def save_latex(results, path):
    lines = [r"\begin{table}[ht]",r"\centering",
             r"\caption{Full ablation: downstream performance by graph composition, initialisation, and split.}",
             r"\label{tab:ablation_full}",r"\scriptsize"]
    for st in SPLIT_TYPES + ["none"]:
        rows = [r for r in results if r["split"] == st]
        if not rows: continue
        lbl = st.capitalize() + " Split" if st != "none" else "No Graph (Baseline)"
        lines += [r"\vspace{0.4em}",r"\textbf{"+lbl+r"}",r"\vspace{0.2em}","",
                  r"\begin{tabular}{ll|cc|cc|cc}",r"\hline",
                  r"Config & Init & \multicolumn{2}{c|}{Int. Pred.} & \multicolumn{2}{c|}{EC Pred.} & \multicolumn{2}{c}{PW Pred.} \\",
                  r" & & AUC & AUPRC & AUC & AUPRC & AUC & AUPRC \\",r"\hline"]
        for r in rows:
            cfg = r["graph_config"].replace("_",r"\_").replace("+",r"+")
            vs = [r.get(k,float("nan")) for k in ["int_auc","int_auprc","ec_auc","ec_auprc","pw_auc","pw_auprc"]]
            ss = [f"{v:.3f}" if not np.isnan(v) else "--" for v in vs]
            lines.append(f"{cfg} & {r['init']} & {' & '.join(ss)} \\\\")
        lines += [r"\hline",r"\end{tabular}",""]
    lines.append(r"\end{table}")
    with open(path,"w") as f: f.write("\n".join(lines))


# ==================================================================
# 9. Main
# ==================================================================
def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    ablation_dir = os.path.join(DATA_ROOT, "ablation_models")
    os.makedirs(ablation_dir, exist_ok=True)

    print("="*80)
    print("  FULL ABLATION STUDY")
    print(f"  Device: {device()}")
    print(f"  GAT epochs: {args.gat_epochs}")
    print(f"  Output: {OUT_DIR}")
    print("="*80)

    # Pre-load biogrid
    _get_biogrid()

    results = []
    cfg_filter = set(args.configs) if args.configs else None

    # ------ BASELINES: no graph ------
    for init_type in args.inits:
        cfg_name = "esm_only"
        if cfg_filter and cfg_name not in cfg_filter:
            continue
        tag = f"{cfg_name}_{init_type}"
        print(f"\n{'#'*80}\n# BASELINE: {tag} (no graph)\n{'#'*80}")
        set_seed(EVAL_SEED)
        emb = build_baseline_embeddings(init_type)
        print(f"  {len(emb)} proteins, dim={next(iter(emb.values())).shape[0]}")

        row = {"split": "none", "graph_config": cfg_name, "init": init_type}
        print("  [1/3] EC ..."); set_seed(EVAL_SEED); row.update(eval_ec(emb))
        print(f"        AUC={row['ec_auc']:.4f}")
        print("  [2/3] Interaction ..."); set_seed(EVAL_SEED); row.update(eval_interaction(emb))
        print(f"        AUC={row['int_auc']:.4f}")
        print("  [3/3] Pathway ..."); set_seed(EVAL_SEED); row.update(eval_pathway(emb))
        print(f"        AUC={row['pw_auc']:.4f}")
        results.append(row)
        del emb; gc.collect()
        # Save incremental results after each baseline
        save_csv(results, os.path.join(OUT_DIR, "ablation_full_results.csv"))

    # ------ GRAPH CONFIGS ------
    for cfg_name, sources in GRAPH_CONFIGS:
        if not sources:  # skip esm_only, handled above
            continue
        if cfg_filter and cfg_name not in cfg_filter:
            continue

        # Build / cache merged graph
        graph_cache = os.path.join(ablation_dir, f"graph_{cfg_name}.gpickle")
        if os.path.exists(graph_cache):
            print(f"\n  Loading cached graph: {graph_cache}")
            G = load_pkl(graph_cache)
            print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        else:
            print(f"\n  Building graph: {cfg_name} from {sources}")
            G = merge_graphs(sources)
            print(f"  Merged: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            dump_pkl(G, graph_cache)

        # Create / cache splits
        split_dir = os.path.join(ablation_dir, f"splits_{cfg_name}")
        if not os.path.exists(os.path.join(split_dir, "edge_split_train.edgelist.gz")):
            print(f"  Creating edge splits...")
            create_edge_split(G, split_dir)
        if not os.path.exists(os.path.join(split_dir, "node_split_nodes.pkl")):
            print(f"  Creating node splits...")
            create_node_split(G, split_dir)

        del G; gc.collect()

        for init_type in args.inits:
            for split_type in args.splits:
                tag = f"{cfg_name}_{init_type}_{split_type}"
                model_dir = os.path.join(ablation_dir, f"model_{tag}")
                emb_path = os.path.join(model_dir, "embeddings.pkl")

                print(f"\n{'#'*80}")
                print(f"# {tag}")
                print(f"{'#'*80}")

                # Train or load
                if os.path.exists(emb_path) and args.skip_training:
                    print(f"  Loading cached embeddings: {emb_path}")
                    emb = load_pkl(emb_path)
                elif os.path.exists(emb_path):
                    print(f"  Embeddings exist, loading: {emb_path}")
                    emb = load_pkl(emb_path)
                else:
                    print(f"  Training GAT ...")
                    emb = train_gat(split_dir, split_type, init_type, model_dir, epochs=args.gat_epochs)

                print(f"  {len(emb)} proteins, dim={next(iter(emb.values())).shape[0]}")

                row = {"split": split_type, "graph_config": cfg_name, "init": init_type}
                print("  [1/3] EC ..."); set_seed(EVAL_SEED); row.update(eval_ec(emb))
                print(f"        AUC={row['ec_auc']:.4f}")
                print("  [2/3] Interaction ..."); set_seed(EVAL_SEED); row.update(eval_interaction(emb))
                print(f"        AUC={row['int_auc']:.4f}")
                print("  [3/3] Pathway ..."); set_seed(EVAL_SEED); row.update(eval_pathway(emb))
                print(f"        AUC={row['pw_auc']:.4f}")
                results.append(row)
                del emb; gc.collect()

                # Save incremental results after each config
                save_csv(results, os.path.join(OUT_DIR, "ablation_full_results.csv"))

    # ------ Output ------
    tbl = fmt_table(results)
    print("\n\n" + "="*130 + "\n  FULL ABLATION RESULTS\n" + "="*130 + tbl)

    csv_p = os.path.join(OUT_DIR, "ablation_full_results.csv")
    save_csv(results, csv_p); print(f"\nCSV:   {csv_p}")
    tex_p = os.path.join(OUT_DIR, "ablation_full_results.tex")
    save_latex(results, tex_p); print(f"LaTeX: {tex_p}")
    txt_p = os.path.join(OUT_DIR, "ablation_full_results.txt")
    with open(txt_p,"w") as f: f.write("FULL ABLATION RESULTS\n"+"="*130+"\n"+tbl+"\n")
    print(f"Text:  {txt_p}")


if __name__ == "__main__":
    main()
