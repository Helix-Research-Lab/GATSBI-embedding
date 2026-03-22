#!/usr/bin/env python3
"""
DeepSets architecture for pathway set prediction with swappable embeddings.

Implements the same DeepSets architecture as eval_deepsets.py
(phi per-element MLP → sum pooling → rho classification MLP)
but accepts any per-protein embedding via --embeddings.

Uses the same pathway split (seed=42, 80/10/10) as GATSBI/PINNACLE evaluations.
"""

import os
import sys
import pickle
import random
import argparse

import numpy as np
import torch
import torch.nn as nn

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

PATHWAY_PKL = os.path.join(DATA, "pathway_proteinSet_filtered.pkl")

SEED = 42
EPOCHS = 30
LR = 1e-3

# ---- reuse functions ----
sys.path.insert(0, CODE)
from eval_set_prediction import (
    load_embed_dict,
    load_pathway_sets,
    build_set_embedding,
    corrupt_pathway,
    split_pathways,
    build_dataset,
)
from eval_deepsets import (
    DeepSetsClassifier,
    train_deepsets,
    compute_metrics_binary,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="DeepSets architecture for pathway prediction with swappable embeddings"
    )
    parser.add_argument("--embeddings", type=str, required=True,
                        help="Path to embedding dictionary (.pkl)")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--pathways", type=str, default=PATHWAY_PKL,
                        help="Path to pathway pickle")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- load data ----
    emb = load_embed_dict(args.embeddings)

    # Convert torch tensors to numpy if needed
    sample_val = next(iter(emb.values()))
    if hasattr(sample_val, 'numpy'):
        emb = {k: v.numpy() if hasattr(v, 'numpy') else np.array(v) for k, v in emb.items()}

    pathway_to_prots = load_pathway_sets(args.pathways)
    pathways = list(pathway_to_prots.keys())

    sizes = [len(s) for s in pathway_to_prots.values()]
    K = int(np.median(sizes))
    print(f"Using median K = {K}")

    # ---- split (same as GATSBI/PINNACLE) ----
    train_pw, val_pw, test_pw = split_pathways(pathways, seed=args.seed)
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
    model = train_deepsets(model, X_train, y_train, X_val, y_val, args.out_dir,
                           epochs=args.epochs, lr=args.lr)

    # ---- evaluate on test set ----
    print("\nEvaluating on test set...")
    model.eval()

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        test_logits = model(X_test_t)

    test_auc, test_auprc, test_acc, test_prec, test_rec, test_f1, test_probs, test_labels = \
        compute_metrics_binary(test_logits, y_test_t)

    print(f"\nDeepSets (swap) TEST METRICS")
    print(f"AUC:     {test_auc:.4f}")
    print(f"AUPRC:   {test_auprc:.4f}")
    print(f"ACC:     {test_acc:.4f}")
    print(f"PREC:    {test_prec:.4f}")
    print(f"RECALL:  {test_rec:.4f}")
    print(f"F1:      {test_f1:.4f}")

    # ---- save results ----
    np.save(os.path.join(args.out_dir, "test_probs.npy"), test_probs)
    np.save(os.path.join(args.out_dir, "test_labels.npy"), test_labels)

    print(f"\nResults saved to {args.out_dir}")


if __name__ == "__main__":
    main()
