# gat_node_split_train_unified.py
# Node-split training:
#   V_mp = V_train
#   E_mp = E[V_train] (train-induced edges)
#   E+   = E[V_train]
# Degree-matched negatives (5:1), GAT encoder as specified.

import os
import gzip
import pickle
import random
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_undirected

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt


# -------------------------
# Helpers
# -------------------------
def iter_edges_gz(path: str):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                yield parts[0], parts[1]


def dump_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_node_splits(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)  # dict: {"train": [...], "val": [...], "test": [...]}


def build_or_load_node_id_map_from_nodes(node_split_pkl, cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    node_splits = load_node_splits(node_split_pkl)
    node2id = {}
    def get_id(x):
        if x not in node2id:
            node2id[x] = len(node2id)
        return node2id[x]

    for split in ["train", "val", "test"]:
        for n in node_splits.get(split, []):
            get_id(n)

    with open(cache_path, "wb") as f:
        pickle.dump(node2id, f, protocol=pickle.HIGHEST_PROTOCOL)
    return node2id


def load_edges_all(path: str, node2id):
    src, dst = [], []
    if not os.path.exists(path):
        return torch.empty((2, 0), dtype=torch.long)
    for u, v in iter_edges_gz(path):
        if u in node2id and v in node2id:
            src.append(node2id[u])
            dst.append(node2id[v])
    if len(src) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


def build_x_init_from_esm(node2id, esm_path, dim=1280, seed=123):
    with open(esm_path, "rb") as f:
        esm = pickle.load(f)

    N = len(node2id)
    rng = np.random.default_rng(seed)
    X = np.empty((N, dim), dtype=np.float32)

    valid_vecs = []
    for name in node2id:
        v = esm.get(name, None)
        if v is not None:
            vv = np.asarray(v, dtype=np.float32)
            if vv.shape == (dim,):
                valid_vecs.append(vv)
    if len(valid_vecs) == 0:
        raise RuntimeError("No valid ESM vectors found to compute centroid")
    centroid = np.mean(valid_vecs, axis=0).astype(np.float32)

    missing = 0
    badshape = 0
    for name, idx in node2id.items():
        v = esm.get(name, None)
        if v is None:
            missing += 1
            X[idx] = rng.normal(0.0, 0.02, size=(dim,)).astype(np.float32)
        else:
            vv = np.asarray(v, dtype=np.float32)
            if vv.shape != (dim,):
                badshape += 1
                X[idx] = rng.normal(0.0, 0.02, size=(dim,)).astype(np.float32)
            else:
                X[idx] = vv - centroid

    print(f"ESM init: loaded={N-missing-badshape:,} missing={missing:,} badshape={badshape:,}")
    return torch.from_numpy(X)


def build_observed_undirected_edge_set(edge_index_undirected: torch.Tensor):
    ei = edge_index_undirected.cpu()
    src = ei[0].numpy()
    dst = ei[1].numpy()
    obs = set()
    for u, v in zip(src, dst):
        a = int(u); b = int(v)
        if a == b:
            continue
        if a < b:
            obs.add((a, b))
        else:
            obs.add((b, a))
    return obs


def build_degree_sampler(edge_index_undirected: torch.Tensor, num_nodes: int, alpha: float = 1.0):
    deg = torch.zeros(num_nodes, dtype=torch.long)
    ei = edge_index_undirected.cpu()
    if ei.numel() > 0:
        deg.scatter_add_(0, ei[0], torch.ones(ei.size(1), dtype=torch.long))
    p = deg.float().clamp_min(1.0).pow(alpha)
    p = (p / p.sum()).numpy()
    return deg.numpy(), p


def sample_negatives_degree_reject(
    rng: np.random.Generator,
    src_nodes: torch.Tensor,
    p_dst: np.ndarray,
    observed_undir: set,
    num_nodes: int,
    neg_ratio: int,
    max_tries: int,
    device: str
):
    src_np = src_nodes.detach().cpu().numpy().astype(np.int64)
    B = src_np.shape[0]
    total = B * neg_ratio

    neg_src = np.repeat(src_np, neg_ratio)
    neg_dst = np.empty(total, dtype=np.int64)

    for i in range(total):
        s = int(neg_src[i])

        ok = False
        for _ in range(max_tries):
            t = int(rng.choice(num_nodes, p=p_dst))
            if t == s:
                continue
            a, b = (s, t) if s < t else (t, s)
            if (a, b) in observed_undir:
                continue
            neg_dst[i] = t
            ok = True
            break

        if not ok:
            t = int(rng.integers(0, num_nodes))
            if t == s:
                t = (t + 1) % num_nodes
            neg_dst[i] = t

    neg_src_t = torch.from_numpy(neg_src).to(device=device, dtype=torch.long)
    neg_dst_t = torch.from_numpy(neg_dst).to(device=device, dtype=torch.long)
    return torch.stack([neg_src_t, neg_dst_t], dim=0)


def bce_loss_logits(pos_logits, neg_logits):
    logits = torch.cat([pos_logits, neg_logits], dim=0)
    labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)
    labels = labels * 0.9 + 0.05
    return F.binary_cross_entropy_with_logits(logits, labels)


def compute_auc_ap(pos_logits, neg_logits):
    logits = torch.cat([pos_logits, neg_logits], dim=0).detach().cpu().numpy()
    labels = np.concatenate([
        np.ones(len(pos_logits)),
        np.zeros(len(neg_logits))
    ])

    try:
        auc_val = roc_auc_score(labels, logits)
    except ValueError:
        auc_val = float('nan')

    try:
        ap = average_precision_score(labels, logits)
    except ValueError:
        ap = float('nan')

    return auc_val, ap


# -------------------------
# Encoder / Decoder
# -------------------------
PROJ_DIM = 512
OUT_DIM = 512
HIDDEN = 512
HEADS1 = 4
HEADS2 = 4
HEADS3 = 2
DROPOUT = 0.2


class GATEncoder(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.dropout = DROPOUT
        self.proj = nn.Linear(in_dim, PROJ_DIM)
        self.gat1 = GATConv(PROJ_DIM, HIDDEN, heads=HEADS1, concat=True, dropout=DROPOUT)
        self.gat2 = GATConv(HIDDEN * HEADS1, HIDDEN, heads=HEADS2, concat=True, dropout=DROPOUT)
        self.gat3 = GATConv(HIDDEN * HEADS2, OUT_DIM, heads=HEADS3, concat=False, dropout=DROPOUT)

    def forward(self, x, edge_index):
        x = self.proj(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat3(x, edge_index)
        return x


class EdgeMLPDecoder(nn.Module):
    def __init__(self, dim: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        in_dim = 4 * dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, z, edge_index):
        u = edge_index[0]
        v = edge_index[1]
        zu = z[u]
        zv = z[v]
        x = torch.cat([zu, zv, zu * zv, (zu - zv).abs()], dim=-1)
        return self.net(x).squeeze(-1)


@torch.no_grad()
def compute_val_loss_batched(
    encoder, node_emb, decoder, data, pos_edge_index,
    num_nodes, p_dst, observed_undir, rng,
    device, batch_edges=100_000, neg_ratio=5, max_tries=20
):
    encoder.eval()
    node_emb.eval()
    decoder.eval()

    use_amp = False
    z = encoder(node_emb.weight, data.edge_index)

    total_loss = 0.0
    total_count = 0

    E = pos_edge_index.size(1)

    val_pos_all = []
    val_neg_all = []
    for start in range(0, E, batch_edges):
        end = min(start + batch_edges, E)
        pos_e = pos_edge_index[:, start:end].to(device)
        src_nodes = pos_e[0]

        neg_e = sample_negatives_degree_reject(
            rng=rng,
            src_nodes=src_nodes,
            p_dst=p_dst,
            observed_undir=observed_undir,
            num_nodes=num_nodes,
            neg_ratio=neg_ratio,
            max_tries=max_tries,
            device=device
        )

        with torch.cuda.amp.autocast(enabled=use_amp):
            pos_logits = decoder(z, pos_e)
            neg_logits = decoder(z, neg_e)

            val_pos_all.append(pos_logits.detach().cpu())
            val_neg_all.append(neg_logits.detach().cpu())
            logits = torch.cat([pos_logits, neg_logits], dim=0)
            labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)
            loss_sum = F.binary_cross_entropy_with_logits(logits, labels, reduction="sum")

        total_loss += float(loss_sum.detach().cpu())
        total_count += int(labels.numel())

    val_pos_all = torch.cat(val_pos_all, dim=0)
    val_neg_all = torch.cat(val_neg_all, dim=0)
    val_auc, val_ap = compute_auc_ap(val_pos_all, val_neg_all)
    return total_loss / total_count, val_auc, val_ap


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Node-split GAT training with degree-matched negatives.")
    parser.add_argument("--split_dir", type=str, required=True, help="Directory containing node_split_* files")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for model, embeddings, curves.")
    parser.add_argument("--esm_path", type=str, required=True, help="Path to esm_uniprot_vec.pkl")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--pos_batch", type=int, default=20_000)
    parser.add_argument("--neg_ratio", type=int, default=5)
    parser.add_argument("--val_max_edges", type=int, default=300_000)
    args = parser.parse_args()

    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.out_dir, exist_ok=True)

    NODE_SPLIT_PKL = os.path.join(args.split_dir, "node_split_nodes.pkl")
    TRAIN_EDGES_PATH = os.path.join(args.split_dir, "node_split_train_induced.edgelist.gz")
    VAL_EDGES_PATH   = os.path.join(args.split_dir, "node_split_val_induced.edgelist.gz")
    TEST_EDGES_PATH  = os.path.join(args.split_dir, "node_split_test_induced.edgelist.gz")

    NODEMAP_CACHE = os.path.join(args.out_dir, "node_id_map_node_split.pkl")

    DIM = 1280
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = args.epochs
    POS_BATCH = args.pos_batch
    NEG_RATIO = args.neg_ratio
    NEG_MAX_TRIES = 20
    VAL_BATCH_EDGES = 100_000
    VAL_MAX_EDGES = args.val_max_edges
    EMB_L2_COEF = 1e-7

    RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
    MODEL_OUT   = os.path.join(args.out_dir, f"gat_node_model_{RUN_TAG}.pt")
    EMB_OUT     = os.path.join(args.out_dir, f"gat_node_embeddings_by_name_{RUN_TAG}.pkl")
    NODEMAP_OUT = os.path.join(args.out_dir, f"node_id_map_node_split_{RUN_TAG}.pkl")

    print("Building/loading node map from node_split_nodes.pkl...")
    node2id = build_or_load_node_id_map_from_nodes(NODE_SPLIT_PKL, NODEMAP_CACHE)
    id2node = {i: n for n, i in node2id.items()}
    num_nodes = len(node2id)
    print(f"num_nodes={num_nodes:,}")
    dump_pickle(node2id, NODEMAP_OUT)

    node_splits = load_node_splits(NODE_SPLIT_PKL)
    train_nodes = node_splits.get("train", [])
    train_ids = np.array([node2id[n] for n in train_nodes if n in node2id], dtype=np.int64)
    train_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[train_ids] = True

    print("Loading ESM init features...")
    x_init = build_x_init_from_esm(node2id, args.esm_path, dim=DIM, seed=SEED).to(DEVICE)

    print("Loading train-induced edges for message passing graph (E_mp = E[V_train])...")
    train_e = load_edges_all(TRAIN_EDGES_PATH, node2id)
    train_edge_index = to_undirected(train_e, num_nodes=num_nodes)
    print(f"MP edges (undirected) = {train_edge_index.size(1):,}")

    data = Data(edge_index=train_edge_index, num_nodes=num_nodes).to(DEVICE)
    pos_pool = train_edge_index.to(DEVICE)  # E+ = E[V_train]

    print("Building observed edge set (full node-split network) and degree sampler...")
    val_e  = load_edges_all(VAL_EDGES_PATH, node2id)
    test_e = load_edges_all(TEST_EDGES_PATH, node2id)
    full_e = torch.cat([
        train_edge_index,
        to_undirected(val_e, num_nodes=num_nodes),
        to_undirected(test_e, num_nodes=num_nodes)
    ], dim=1) if val_e.numel() > 0 or test_e.numel() > 0 else train_edge_index

    observed_undir = build_observed_undirected_edge_set(full_e)

    deg, p_dst = build_degree_sampler(train_edge_index, num_nodes=num_nodes, alpha=1.0)
    p_dst = p_dst.copy()
    p_dst[~train_mask] = 0.0
    p_dst = p_dst / p_dst.sum()

    print("  degree stats (train graph): min", int(deg[train_mask].min() if train_mask.any() else 0),
          "median", int(np.median(deg[train_mask]) if train_mask.any() else 0),
          "max", int(deg[train_mask].max() if train_mask.any() else 0))
    print("  observed undirected edges in set:", len(observed_undir))

    rng = np.random.default_rng(SEED)

    print("Loading validation node-split induced edges...")
    val_edge_index = load_edges_all(VAL_EDGES_PATH, node2id)
    print("val positives:", val_edge_index.size(1))
    if VAL_MAX_EDGES is not None and val_edge_index.size(1) > VAL_MAX_EDGES:
        perm = torch.randperm(val_edge_index.size(1))[:VAL_MAX_EDGES]
        val_edge_index = val_edge_index[:, perm]
        print("val positives (subsampled):", val_edge_index.size(1))

    node_emb = nn.Embedding(num_nodes, DIM).to(DEVICE)
    with torch.no_grad():
        node_emb.weight.copy_(x_init)

    encoder = GATEncoder(in_dim=DIM).to(DEVICE)
    decoder = EdgeMLPDecoder(dim=OUT_DIM, hidden=128, dropout=0.2).to(DEVICE)

    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(node_emb.parameters()) + list(decoder.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    use_amp = False
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"Training on {DEVICE} (node-split)...")
    for epoch in range(1, EPOCHS + 1):
        encoder.train()
        node_emb.train()
        decoder.train()

        opt.zero_grad(set_to_none=True)

        z = encoder(node_emb.weight, data.edge_index)

        E = pos_pool.size(1)
        batch_pos = min(POS_BATCH, E)
        perm = torch.randint(0, E, (batch_pos,), device=DEVICE)
        pos_edge_index = pos_pool[:, perm]

        neg_edge_index = sample_negatives_degree_reject(
            rng=rng,
            src_nodes=pos_edge_index[0],
            p_dst=p_dst,
            observed_undir=observed_undir,
            num_nodes=num_nodes,
            neg_ratio=NEG_RATIO,
            max_tries=NEG_MAX_TRIES,
            device=DEVICE
        )

        pos_logits = decoder(z, pos_edge_index)
        neg_logits = decoder(z, neg_edge_index)
        loss = bce_loss_logits(pos_logits, neg_logits)

        train_auc, train_ap = compute_auc_ap(pos_logits, neg_logits)

        if EMB_L2_COEF and EMB_L2_COEF > 0:
            loss = loss + EMB_L2_COEF * node_emb.weight.norm(p=2)

        scaler.scale(loss).backward()

        if epoch == 1:
            g = node_emb.weight.grad
            print("Grad norm on node_emb:", float(g.norm().detach().cpu()) if g is not None else None)

        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 2.0)
        torch.nn.utils.clip_grad_norm_(node_emb.parameters(), 2.0)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 2.0)

        scaler.step(opt)
        scaler.update()
        scheduler.step()

        train_loss = float(loss.detach().cpu())

        val_loss, val_auc, val_ap = compute_val_loss_batched(
            encoder=encoder,
            node_emb=node_emb,
            decoder=decoder,
            data=data,
            pos_edge_index=val_edge_index,
            num_nodes=num_nodes,
            p_dst=p_dst,
            observed_undir=observed_undir,
            rng=rng,
            device=DEVICE,
            batch_edges=VAL_BATCH_EDGES,
            neg_ratio=NEG_RATIO,
            max_tries=NEG_MAX_TRIES
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | train_auc={train_auc:.4f} | train_ap={train_ap:.4f} | "
            f"val_loss={val_loss:.4f} | val_auc={val_auc:.4f} | val_ap={val_ap:.4f}"
        )

    print("Computing full-train and full-val ROC/PR curves (node-split)...")

    encoder.eval()
    node_emb.eval()
    decoder.eval()

    with torch.no_grad():
        z = encoder(node_emb.weight, data.edge_index)

        train_pos = pos_pool
        train_neg = sample_negatives_degree_reject(
            rng=rng,
            src_nodes=train_pos[0],
            p_dst=p_dst,
            observed_undir=observed_undir,
            num_nodes=num_nodes,
            neg_ratio=NEG_RATIO,
            max_tries=NEG_MAX_TRIES,
            device=DEVICE
        )

        train_pos_logits = decoder(z, train_pos).detach().cpu().numpy()
        train_neg_logits = decoder(z, train_neg).detach().cpu().numpy()

        train_labels = np.concatenate([
            np.ones_like(train_pos_logits),
            np.zeros_like(train_neg_logits)
        ])
        train_scores = np.concatenate([train_pos_logits, train_neg_logits])

        val_pos = val_edge_index.to(DEVICE)
        val_neg = sample_negatives_degree_reject(
            rng=rng,
            src_nodes=val_pos[0],
            p_dst=p_dst,
            observed_undir=observed_undir,
            num_nodes=num_nodes,
            neg_ratio=NEG_RATIO,
            max_tries=NEG_MAX_TRIES,
            device=DEVICE
        )

        val_pos_logits = decoder(z, val_pos).detach().cpu().numpy()
        val_neg_logits = decoder(z, val_neg).detach().cpu().numpy()

        val_labels = np.concatenate([
            np.ones_like(val_pos_logits),
            np.zeros_like(val_neg_logits)
        ])
        val_scores = np.concatenate([val_pos_logits, val_neg_logits])

    train_fpr, train_tpr, _ = roc_curve(train_labels, train_scores)
    val_fpr, val_tpr, _ = roc_curve(val_labels, val_scores)

    train_auc_val = auc(train_fpr, train_tpr)
    val_auc_val = auc(val_fpr, val_tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(train_fpr, train_tpr, label=f"Train ROC (AUC={train_auc_val:.3f})")
    plt.plot(val_fpr, val_tpr, label=f"Val ROC (AUC={val_auc_val:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (node-split)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "roc_curve_node_split.svg"))
    plt.close()

    train_prec, train_rec, _ = precision_recall_curve(train_labels, train_scores)
    val_prec, val_rec, _ = precision_recall_curve(val_labels, val_scores)

    train_ap_val = auc(train_rec, train_prec)
    val_ap_val = auc(val_rec, val_prec)

    plt.figure(figsize=(7, 6))
    plt.plot(train_rec, train_prec, label=f"Train PRC (AP={train_ap_val:.3f})")
    plt.plot(val_rec, val_prec, label=f"Val PRC (AP={val_ap_val:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (node-split)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "prc_curve_node_split.svg"))
    plt.close()

    print("Saved ROC/PRC curves to", args.out_dir)

    ckpt = {
        "run_tag": RUN_TAG,
        "seed": SEED,
        "config": {
            "DIM": DIM, "HIDDEN": HIDDEN, "HEADS1": HEADS1, "HEADS2": HEADS2, "HEADS3": HEADS3,
            "DROPOUT": DROPOUT,
            "LR": LR, "WEIGHT_DECAY": WEIGHT_DECAY, "EPOCHS": EPOCHS,
            "POS_BATCH": POS_BATCH, "NEG_RATIO": NEG_RATIO, "NEG_MAX_TRIES": NEG_MAX_TRIES,
            "VAL_BATCH_EDGES": VAL_BATCH_EDGES, "VAL_MAX_EDGES": VAL_MAX_EDGES,
            "EMB_L2_COEF": EMB_L2_COEF,
        },
        "node2id_path": NODEMAP_OUT,
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "node_emb_state_dict": node_emb.state_dict(),
    }
    torch.save(ckpt, MODEL_OUT)
    print("Saved model checkpoint:", MODEL_OUT)

    encoder.eval()
    node_emb.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
        z = encoder(node_emb.weight, data.edge_index).detach().cpu().numpy().astype(np.float32)

    emb_by_name = {id2node[i]: z[i] for i in range(num_nodes)}
    dump_pickle(emb_by_name, EMB_OUT)
    print("Saved embeddings:", EMB_OUT)
    print("Embedding matrix shape:", z.shape)


if __name__ == "__main__":
    main()
