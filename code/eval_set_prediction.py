import os
import pickle
import random
import argparse
from typing import Dict, Set, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    precision_recall_curve
)

# ---------------------------------------------------------
# 1. Argument parsing
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Pathway set prediction using protein embeddings")

    parser.add_argument("--embeddings", type=str, required=True,
                        help="Path to protein embedding dictionary (.pkl)")

    parser.add_argument("--pathways", type=str, required=True,
                        help="Path to pathway->proteinSet pickle")

    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for results")

    parser.add_argument("--epochs", type=int, default=30,
                        help="Training epochs")

    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    return parser.parse_args()


# ---------------------------------------------------------
# 2. Load data
# ---------------------------------------------------------
def load_embed_dict(path):
    with open(path, "rb") as f:
        emb = pickle.load(f)
    print(f"Loaded {len(emb)} embeddings")
    return emb


def load_pathway_sets(path) -> Dict[str, Set[str]]:
    with open(path, "rb") as f:
        pathway_to_prots = pickle.load(f)
    print(f"Loaded {len(pathway_to_prots)} pathways")
    return pathway_to_prots


# ---------------------------------------------------------
# 3. Attention pooling
# ---------------------------------------------------------
class MultiHeadAttentionPool(nn.Module):
    def __init__(self, emb_dim, num_heads=4, head_dim=64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.total_dim = num_heads * head_dim

        self.q_proj = nn.Linear(emb_dim, self.total_dim)
        self.k_proj = nn.Linear(emb_dim, self.total_dim)
        self.v_proj = nn.Linear(emb_dim, self.total_dim)

        self.out_proj = nn.Linear(self.total_dim, emb_dim)

    def forward(self, x):
        B, K, D = x.shape

        Q = self.q_proj(x)
        K_ = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        K_ = K_.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K_.transpose(-2, -1)) / np.sqrt(self.head_dim)
        weights = torch.softmax(scores, dim=-1)

        attended = torch.matmul(weights, V)
        pooled = attended.mean(dim=2)
        pooled = pooled.reshape(B, self.total_dim)

        return self.out_proj(pooled)


# ---------------------------------------------------------
# 4. Build set embeddings
# ---------------------------------------------------------
def build_set_embedding(protein_set: Set[str], emb: Dict[str, np.ndarray], K: int, emb_dim: int):
    proteins = list(protein_set)[:K]

    X = []
    for p in proteins:
        if p in emb:
            X.append(emb[p])
        else:
            X.append(np.zeros(emb_dim, dtype=np.float32))

    while len(X) < K:
        X.append(np.zeros(emb_dim, dtype=np.float32))

    return np.stack(X, axis=0)


# ---------------------------------------------------------
# 5. Corrupted positives as negatives
# ---------------------------------------------------------
def corrupt_pathway(protein_set: Set[str], all_proteins: List[str], corruption_rate=0.4):
    prot_list = list(protein_set)
    n = len(prot_list)
    k = max(1, int(n * corruption_rate))

    replace_idx = random.sample(range(n), k)
    replacements = random.sample(all_proteins, k)

    for i, r in zip(replace_idx, replacements):
        prot_list[i] = r

    return set(prot_list)


# ---------------------------------------------------------
# 6. Pathway-level split
# ---------------------------------------------------------
def split_pathways(pathways: List[str], train_ratio=0.8, val_ratio=0.1, seed=42):
    random.seed(seed)
    random.shuffle(pathways)
    n = len(pathways)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    return pathways[:n_train], pathways[n_train:n_train+n_val], pathways[n_train+n_val:]


# ---------------------------------------------------------
# 7. Build dataset
# ---------------------------------------------------------
def build_dataset(pathway_to_prots, emb, K, pathways_subset, all_proteins, num_neg_per_pos=1):
    emb_dim = next(iter(emb.values())).shape[0]

    X, y = [], []

    for pw in pathways_subset:
        prot_set = {p for p in pathway_to_prots[pw] if p in emb}
        if len(prot_set) == 0:
            continue

        X.append(build_set_embedding(prot_set, emb, K, emb_dim))
        y.append(1)

    for pw in pathways_subset:
        prot_set = {p for p in pathway_to_prots[pw] if p in emb}
        if len(prot_set) == 0:
            continue

        for _ in range(num_neg_per_pos):
            neg_set = corrupt_pathway(prot_set, all_proteins)
            X.append(build_set_embedding(neg_set, emb, K, emb_dim))
            y.append(0)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print("Subset dataset built:", X.shape, y.shape)
    return X, y


# ---------------------------------------------------------
# 8. Model
# ---------------------------------------------------------
class SetMLP(nn.Module):
    def __init__(self, emb_dim, num_heads=4, head_dim=64):
        super().__init__()
        self.pool = MultiHeadAttentionPool(emb_dim, num_heads, head_dim)

        self.fc1 = nn.Sequential(
            nn.Linear(emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        pooled = self.pool(x)
        out = self.fc1(pooled)
        out = self.fc2(out)
        return self.fc3(out)


# ---------------------------------------------------------
# 9. Metrics
# ---------------------------------------------------------
def compute_metrics_binary(logits, labels):
    probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
    labels = labels.detach().cpu().numpy().flatten()

    auc = roc_auc_score(labels, probs)
    auprc = average_precision_score(labels, probs)

    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    return auc, auprc, acc, prec, rec, f1, probs, labels


# ---------------------------------------------------------
# 10. Training
# ---------------------------------------------------------
def train_dnn(model, X_train, y_train, X_val, y_val, out_dir, epochs=30, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        train_logits = model(X_train)
        train_loss = loss_fn(train_logits, y_train)
        train_loss.backward()
        opt.step()

        train_auc, train_auprc, *_ = compute_metrics_binary(train_logits, y_train)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = loss_fn(val_logits, y_val)

        val_auc, val_auprc, *_ = compute_metrics_binary(val_logits, y_val)

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss.item():.4f} | "
            f"Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} | "
            f"Train AUPRC: {train_auprc:.4f} | Val AUPRC: {val_auprc:.4f}"
        )

    torch.save(model.state_dict(), f"{out_dir}/trained_set_mlp.pt")
    print("Saved trained model.")

    return model


# ---------------------------------------------------------
# 11. MAIN
# ---------------------------------------------------------
def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    emb = load_embed_dict(args.embeddings)
    pathway_to_prots = load_pathway_sets(args.pathways)

    pathways = list(pathway_to_prots.keys())

    sizes = [len(s) for s in pathway_to_prots.values()]
    K = int(np.median(sizes))
    print(f"Using median K = {K}")

    train_pw, val_pw, test_pw = split_pathways(pathways, seed=args.seed)

    all_proteins = list({p for s in pathway_to_prots.values() for p in s if p in emb})

    X_train, y_train = build_dataset(pathway_to_prots, emb, K, train_pw, all_proteins)
    X_val, y_val = build_dataset(pathway_to_prots, emb, K, val_pw, all_proteins)
    X_test, y_test = build_dataset(pathway_to_prots, emb, K, test_pw, all_proteins)

    emb_dim = X_train.shape[2]
    model = SetMLP(emb_dim)

    model = train_dnn(model, X_train, y_train, X_val, y_val, args.out_dir,
                      epochs=args.epochs, lr=args.lr)

    print("Evaluating on test set...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        test_logits = model(X_test_t)

    test_auc, test_auprc, test_acc, test_prec, test_rec, test_f1, test_probs, test_labels = \
        compute_metrics_binary(test_logits, y_test_t)

    print("\nTEST METRICS")
    print(f"AUC:     {test_auc:.4f}")
    print(f"AUPRC:   {test_auprc:.4f}")
    print(f"ACC:     {test_acc:.4f}")
    print(f"PREC:    {test_prec:.4f}")
    print(f"RECALL:  {test_rec:.4f}")
    print(f"F1:      {test_f1:.4f}")

    np.save(f"{args.out_dir}/test_probs.npy", test_probs)
    np.save(f"{args.out_dir}/test_labels.npy", test_labels)


if __name__ == "__main__":
    main()
