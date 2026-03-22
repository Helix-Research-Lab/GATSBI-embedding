#!/usr/bin/env python3
"""
DeepSets baseline for pathway set prediction.
Implements the architecture from:
  Zaheer et al., "Deep Sets", NeurIPS 2017.

Uses element-wise MLP → sum pooling → classification MLP.
This contrasts with the existing SetMLP which uses multi-head attention pooling.

Uses the same pathway split (seed=42, 80/10/10) as GATSBI/PINNACLE evaluations.
"""

import os
import sys
import pickle
import random

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

ESM_PKL = os.path.join(DATA, "esm_uniprot_vec.pkl")
PATHWAY_PKL = os.path.join(DATA, "pathway_proteinSet_filtered.pkl")
OUT_DIR = os.path.join(DATA, "pathway_set_pred_deepsets")

SEED = 42
EPOCHS = 30
LR = 1e-3


# ---- reuse pathway functions from eval_set_prediction ----
sys.path.insert(0, CODE)
from eval_set_prediction import (
    load_embed_dict,
    load_pathway_sets,
    build_set_embedding,
    corrupt_pathway,
    split_pathways,
    build_dataset,
)


# ---- DeepSets model ----
class DeepSetsClassifier(nn.Module):
    """
    DeepSets: phi(element) → sum pool → rho(set) → classification.
    Permutation-invariant by construction.
    """

    def __init__(self, in_dim, phi_hidden=512, phi_out=256, rho_hidden=256):
        super().__init__()
        # Per-element transformation
        self.phi = nn.Sequential(
            nn.Linear(in_dim, phi_hidden),
            nn.BatchNorm1d(phi_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(phi_hidden, phi_out),
            nn.ReLU(),
        )
        # Set-level classification
        self.rho = nn.Sequential(
            nn.Linear(phi_out, rho_hidden),
            nn.BatchNorm1d(rho_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(rho_hidden, 1),
        )

    def forward(self, x):
        """
        Args:
            x: (B, K, D) — batch of sets, each with K elements of dimension D
        Returns:
            logits: (B, 1)
        """
        B, K, D = x.shape
        # Apply phi to each element
        x = x.reshape(B * K, D)
        x = self.phi(x)
        x = x.reshape(B, K, -1)
        # Sum pooling (permutation invariant)
        x = x.sum(dim=1)  # (B, phi_out)
        # Classification
        return self.rho(x)


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


def train_deepsets(model, X_train, y_train, X_val, y_val, out_dir, epochs=30, lr=1e-3):
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

    torch.save(model.state_dict(), os.path.join(out_dir, "deepsets_model.pt"))
    print("Saved trained model.")
    return model


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- load data ----
    emb = load_embed_dict(ESM_PKL)

    # Convert torch tensors to numpy if needed
    sample_val = next(iter(emb.values()))
    if hasattr(sample_val, 'numpy'):
        emb = {k: v.numpy() if hasattr(v, 'numpy') else np.array(v) for k, v in emb.items()}

    pathway_to_prots = load_pathway_sets(PATHWAY_PKL)
    pathways = list(pathway_to_prots.keys())

    sizes = [len(s) for s in pathway_to_prots.values()]
    K = int(np.median(sizes))
    print(f"Using median K = {K}")

    # ---- split (same as GATSBI/PINNACLE) ----
    train_pw, val_pw, test_pw = split_pathways(pathways, seed=SEED)
    print(f"Train: {len(train_pw)}, Val: {len(val_pw)}, Test: {len(test_pw)}")

    all_proteins = list({p for s in pathway_to_prots.values() for p in s if p in emb})

    # ---- build datasets ----
    X_train, y_train = build_dataset(pathway_to_prots, emb, K, train_pw, all_proteins)
    X_val, y_val = build_dataset(pathway_to_prots, emb, K, val_pw, all_proteins)
    X_test, y_test = build_dataset(pathway_to_prots, emb, K, test_pw, all_proteins)

    emb_dim = X_train.shape[2]
    print(f"Embedding dim: {emb_dim}, K: {K}")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # ---- train DeepSets ----
    model = DeepSetsClassifier(in_dim=emb_dim)
    model = train_deepsets(model, X_train, y_train, X_val, y_val, OUT_DIR,
                           epochs=EPOCHS, lr=LR)

    # ---- evaluate on test set ----
    print("\nEvaluating on test set...")
    model.eval()

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        test_logits = model(X_test_t)

    test_auc, test_auprc, test_acc, test_prec, test_rec, test_f1, test_probs, test_labels = \
        compute_metrics_binary(test_logits, y_test_t)

    print(f"\nDeepSets TEST METRICS")
    print(f"AUC:     {test_auc:.4f}")
    print(f"AUPRC:   {test_auprc:.4f}")
    print(f"ACC:     {test_acc:.4f}")
    print(f"PREC:    {test_prec:.4f}")
    print(f"RECALL:  {test_rec:.4f}")
    print(f"F1:      {test_f1:.4f}")

    # ---- save results ----
    np.save(os.path.join(OUT_DIR, "test_probs.npy"), test_probs)
    np.save(os.path.join(OUT_DIR, "test_labels.npy"), test_labels)

    # Save test pathway list for understudied evaluation
    with open(os.path.join(OUT_DIR, "test_pathways.pkl"), "wb") as f:
        pickle.dump(test_pw, f)

    print(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
