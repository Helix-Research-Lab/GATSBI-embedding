#!/usr/bin/env python3
"""
CLEAN architecture for EC function prediction with swappable embeddings.

Implements the same contrastive learning pipeline as eval_clean.py
(SupCon-Hard loss → projection head → EC cluster centers → distance-based prediction)
but accepts any per-protein embedding via --embeddings.

Uses the same protein split (seed=42, 80/10/10) as GATSBI/PINNACLE evaluations.
"""

import os
import sys
import pickle
import random
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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

EC_TSV = os.path.join(DATA, "EC_data.tsv")

SEED = 42
PROJ_DIM = 128
EPOCHS = 100
LR = 5e-4
BATCH_SIZE = 256
TEMPERATURE = 0.1

# ---- reuse functions ----
sys.path.insert(0, CODE)
from eval_node_pred import (
    parse_uniprot_ec_tsv,
    build_label_index_level1,
    build_multilabel_vector_level1,
    split_data,
    load_embed_dict,
)
from eval_clean import (
    ProjectionHead,
    SupConHardLoss,
    train_contrastive,
    compute_ec_centers,
    predict_level1_scores,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="CLEAN architecture for EC prediction with swappable embeddings"
    )
    parser.add_argument("--embeddings", type=str, required=True,
                        help="Path to embedding dictionary (.pkl)")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--ec_tsv", type=str, default=EC_TSV,
                        help="Path to EC annotation TSV")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
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
    prot_to_ec = parse_uniprot_ec_tsv(args.ec_tsv)
    emb = load_embed_dict(args.embeddings)

    # Convert torch tensors to numpy if needed
    sample_val = next(iter(emb.values()))
    if hasattr(sample_val, 'numpy'):
        emb = {k: v.numpy() if hasattr(v, 'numpy') else np.array(v) for k, v in emb.items()}

    proteins = [p for p, ecs in prot_to_ec.items() if len(ecs) > 0 and p in emb]
    print(f"Proteins with EC + embeddings: {len(proteins)}")

    ec1_to_idx, all_ec1 = build_label_index_level1(prot_to_ec)
    num_classes = len(all_ec1)
    print(f"EC level-1 classes: {all_ec1}")

    # ---- split (same as GATSBI/PINNACLE) ----
    train_p, val_p, test_p = split_data(proteins, seed=args.seed)
    print(f"Train: {len(train_p)}, Val: {len(val_p)}, Test: {len(test_p)}")

    # ---- prepare data ----
    in_dim = emb[proteins[0]].shape[0]
    print(f"Embedding dim: {in_dim}")

    X_train = np.array([emb[p] for p in train_p])
    X_val = np.array([emb[p] for p in val_p])
    X_test = np.array([emb[p] for p in test_p])

    y_train = np.array([build_multilabel_vector_level1(prot_to_ec[p], ec1_to_idx) for p in train_p])
    y_val = np.array([build_multilabel_vector_level1(prot_to_ec[p], ec1_to_idx) for p in val_p])
    y_test = np.array([build_multilabel_vector_level1(prot_to_ec[p], ec1_to_idx) for p in test_p])

    # For contrastive learning, assign each protein a single EC label
    train_ec_labels = []
    train_full_ec = []
    for p in train_p:
        ecs = prot_to_ec[p]
        full_ecs = [ec for ec in ecs if "." in ec]
        if full_ecs:
            ec1 = full_ecs[0].split(".")[0]
            if ec1 in ec1_to_idx:
                train_ec_labels.append(ec1_to_idx[ec1])
                train_full_ec.append(full_ecs[0])
            else:
                train_ec_labels.append(0)
                train_full_ec.append(full_ecs[0])
        else:
            train_ec_labels.append(0)
            train_full_ec.append("")

    train_ec_labels = np.array(train_ec_labels)

    # ---- train contrastive model ----
    print("\nTraining CLEAN contrastive model...")
    model = ProjectionHead(in_dim=in_dim, hidden_dim=512, out_dim=PROJ_DIM)
    model = train_contrastive(
        model, X_train, train_ec_labels,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        temperature=TEMPERATURE, device=device
    )

    # ---- compute EC cluster centers from training set ----
    print("\nComputing EC cluster centers...")
    full_ec_for_centers = []
    X_for_centers = []
    for i, p in enumerate(train_p):
        ecs = prot_to_ec[p]
        for ec in ecs:
            if "." in ec:
                full_ec_for_centers.append(ec)
                X_for_centers.append(X_train[i])

    X_for_centers = np.array(X_for_centers)
    ec_centers = compute_ec_centers(model, X_for_centers, full_ec_for_centers, device=device)
    print(f"  Computed {len(ec_centers)} EC cluster centers")

    # ---- predict on test set ----
    print("\nPredicting on test set...")
    test_scores = predict_level1_scores(model, X_test, ec_centers, ec1_to_idx, device=device)

    # ---- compute metrics ----
    valid = np.where(y_test.sum(axis=0) > 0)[0]

    aucs = [roc_auc_score(y_test[:, c], test_scores[:, c]) for c in valid]
    auprcs = [average_precision_score(y_test[:, c], test_scores[:, c]) for c in valid]

    macro_auc = np.mean(aucs)
    macro_auprc = np.mean(auprcs)

    preds = (test_scores >= 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="micro", zero_division=0)
    rec = recall_score(y_test, preds, average="micro", zero_division=0)
    f1 = f1_score(y_test, preds, average="micro", zero_division=0)

    print(f"\nCLEAN (swap) TEST METRICS")
    print(f"AUC:     {macro_auc:.4f}")
    print(f"AUPRC:   {macro_auprc:.4f}")
    print(f"ACC:     {acc:.4f}")
    print(f"PREC:    {prec:.4f}")
    print(f"RECALL:  {rec:.4f}")
    print(f"F1:      {f1:.4f}")

    # ---- save results ----
    np.save(os.path.join(args.out_dir, "test_probs.npy"), test_scores)
    np.save(os.path.join(args.out_dir, "test_labels.npy"), y_test)
    np.save(os.path.join(args.out_dir, "ec_classes.npy"), np.array(all_ec1))

    torch.save(model.state_dict(), os.path.join(args.out_dir, "clean_swap_projection.pt"))
    print(f"\nResults saved to {args.out_dir}")


if __name__ == "__main__":
    main()
