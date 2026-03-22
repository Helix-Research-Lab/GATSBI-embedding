#!/usr/bin/env python3
"""
D-SCRIPT-inspired architecture for interaction prediction with swappable embeddings.

D-SCRIPT (Sledzieski et al., Cell Systems 2021) uses per-residue embeddings + a contact
module. Since GATSBI/PINNACLE produce per-protein embeddings, we implement D-SCRIPT's
core design principle: shared projection → pairwise interaction features → classification.

This differs from EdgeMLP (which concatenates raw element-wise features) by first projecting
each protein through a learned projection head before computing interaction features.

Uses the same BioGRID edge split (seed=42, 80/10/10) as GATSBI/PINNACLE evaluations.
"""

import os
import sys
import pickle
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ---- paths ----
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
CODE = os.path.join(BASE, "code")

BIOGRID = os.path.join(DATA, "biogrid.txt")

SEED = 42
EPOCHS = 20
LR = 1e-3

# ---- reuse BioGRID loading from eval_interaction_pred ----
sys.path.insert(0, CODE)
from eval_interaction_pred import load_biogrid_data, split_edges, negative_sample, load_embed_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="D-SCRIPT-inspired interaction prediction with swappable embeddings"
    )
    parser.add_argument("--embeddings", type=str, required=True,
                        help="Path to embedding dictionary (.pkl)")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--biogrid", type=str, default=BIOGRID,
                        help="Path to BioGRID file")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


# ---- D-SCRIPT-inspired model ----
class DScriptPPI(nn.Module):
    """
    D-SCRIPT-inspired interaction prediction for per-protein embeddings.

    Architecture:
      1. Shared projection: maps each protein embedding to a lower-dim interaction space
         (analogous to D-SCRIPT's Bepler & Berger encoder + projection)
      2. Interaction features: element-wise product + absolute difference of projected embeddings
         (analogous to D-SCRIPT's contact module output, but for single per-protein vectors)
      3. Classifier: MLP on interaction features → binary prediction
    """

    def __init__(self, in_dim, proj_dim=100, hidden_dim=50):
        super().__init__()
        # Shared projection head (applied to each protein independently)
        self.projection = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, proj_dim),
            nn.ReLU(),
        )
        # Interaction classifier
        self.interaction = nn.Sequential(
            nn.Linear(proj_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, emb_a, emb_b):
        """
        Args:
            emb_a: (B, D) embeddings of protein A
            emb_b: (B, D) embeddings of protein B
        Returns:
            logits: (B, 1)
        """
        z_a = self.projection(emb_a)
        z_b = self.projection(emb_b)
        # Interaction features
        prod = z_a * z_b
        diff = torch.abs(z_a - z_b)
        combined = torch.cat([prod, diff], dim=-1)
        return self.interaction(combined)


def build_pair_dataset(pos_edges, neg_edges, emb):
    """Build dataset of protein embedding pairs (not concatenated features)."""
    A, B, y = [], [], []

    for u, v in pos_edges:
        if u in emb and v in emb:
            A.append(emb[u])
            B.append(emb[v])
            y.append(1)

    for u, v in neg_edges:
        if u in emb and v in emb:
            A.append(emb[u])
            B.append(emb[v])
            y.append(0)

    return np.array(A), np.array(B), np.array(y)


def train_model(model, A_train, B_train, y_train, A_val, B_val, y_val,
                out_dir, epochs=20, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    A_train = torch.tensor(A_train, dtype=torch.float32).to(device)
    B_train = torch.tensor(B_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    A_val = torch.tensor(A_val, dtype=torch.float32).to(device)
    B_val = torch.tensor(B_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        train_logits = model(A_train, B_train)
        train_loss = loss_fn(train_logits, y_train)
        train_loss.backward()
        opt.step()

        train_probs = torch.sigmoid(train_logits).detach().cpu().numpy().flatten()
        train_labels = y_train.detach().cpu().numpy().flatten()
        train_auc = roc_auc_score(train_labels, train_probs)
        train_auprc = average_precision_score(train_labels, train_probs)

        model.eval()
        with torch.no_grad():
            val_logits = model(A_val, B_val)
            val_loss = loss_fn(val_logits, y_val)

        val_probs = torch.sigmoid(val_logits).detach().cpu().numpy().flatten()
        val_labels = y_val.detach().cpu().numpy().flatten()
        val_auc = roc_auc_score(val_labels, val_probs)
        val_auprc = average_precision_score(val_labels, val_probs)

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss.item():.4f} | "
            f"Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} | "
            f"Train AUPRC: {train_auprc:.4f} | Val AUPRC: {val_auprc:.4f}"
        )

    torch.save(model.state_dict(), os.path.join(out_dir, "dscript_swap_model.pt"))
    print("Saved trained model.")
    return model


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- load data ----
    edges = load_biogrid_data(args.biogrid)
    emb = load_embed_dict(args.embeddings)

    # Convert torch tensors to numpy if needed
    sample_val = next(iter(emb.values()))
    if hasattr(sample_val, 'numpy'):
        emb = {k: v.numpy() if hasattr(v, 'numpy') else np.array(v) for k, v in emb.items()}

    nodes = list(emb.keys())
    in_dim = emb[nodes[0]].shape[0]
    print(f"Embedding dim: {in_dim}, Proteins: {len(nodes)}")

    # ---- split (same as GATSBI/PINNACLE) ----
    train_pos, val_pos, test_pos = split_edges(edges, seed=args.seed)
    train_neg = negative_sample(train_pos, nodes, len(train_pos))
    val_neg = negative_sample(val_pos, nodes, len(val_pos))
    test_neg = negative_sample(test_pos, nodes, len(test_pos))

    print(f"Train: {len(train_pos)} pos + {len(train_neg)} neg")
    print(f"Val:   {len(val_pos)} pos + {len(val_neg)} neg")
    print(f"Test:  {len(test_pos)} pos + {len(test_neg)} neg")

    # ---- build pair datasets ----
    A_train, B_train, y_train = build_pair_dataset(train_pos, train_neg, emb)
    A_val, B_val, y_val = build_pair_dataset(val_pos, val_neg, emb)
    A_test, B_test, y_test = build_pair_dataset(test_pos, test_neg, emb)

    print(f"Train: {A_train.shape}, Val: {A_val.shape}, Test: {A_test.shape}")

    # ---- train D-SCRIPT-inspired model ----
    model = DScriptPPI(in_dim=in_dim)
    model = train_model(model, A_train, B_train, y_train, A_val, B_val, y_val,
                        args.out_dir, epochs=args.epochs, lr=args.lr)

    # ---- evaluate on test set ----
    print("\nEvaluating on test set...")
    model.eval()

    A_test_t = torch.tensor(A_test, dtype=torch.float32).to(device)
    B_test_t = torch.tensor(B_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        test_logits = model(A_test_t, B_test_t)

    test_probs = torch.sigmoid(test_logits).detach().cpu().numpy().flatten()
    test_labels = y_test_t.detach().cpu().numpy().flatten()

    auc = roc_auc_score(test_labels, test_probs)
    auprc = average_precision_score(test_labels, test_probs)
    preds = (test_probs >= 0.5).astype(int)
    acc = accuracy_score(test_labels, preds)
    prec = precision_score(test_labels, preds, zero_division=0)
    rec = recall_score(test_labels, preds, zero_division=0)
    f1 = f1_score(test_labels, preds, zero_division=0)

    print(f"\nD-SCRIPT-inspired TEST METRICS")
    print(f"AUC:     {auc:.4f}")
    print(f"AUPRC:   {auprc:.4f}")
    print(f"ACC:     {acc:.4f}")
    print(f"PREC:    {prec:.4f}")
    print(f"RECALL:  {rec:.4f}")
    print(f"F1:      {f1:.4f}")

    # ---- save results ----
    np.save(os.path.join(args.out_dir, "test_probs.npy"), test_probs)
    np.save(os.path.join(args.out_dir, "test_labels.npy"), test_labels)

    print(f"\nResults saved to {args.out_dir}")


if __name__ == "__main__":
    main()
