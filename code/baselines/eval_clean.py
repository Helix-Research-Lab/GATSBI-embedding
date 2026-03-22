#!/usr/bin/env python3
"""
CLEAN-style baseline for EC function prediction.
Implements the core approach from:
  Yu et al., "Enzyme function prediction using contrastive learning", Science 2023.

Uses SupCon-Hard loss on ESM-1b embeddings to learn a contrastive embedding space
where proteins with the same EC number cluster together. For level-1 EC prediction,
distances to EC cluster centers are converted to scores.

Uses the same protein split (seed=42, 80/10/10) as GATSBI/PINNACLE evaluations.
"""

import os
import sys
import pickle
import random
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
    roc_curve,
    precision_recall_curve,
)

# ---- paths ----
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
CODE = os.path.join(BASE, "code")

EC_TSV = os.path.join(DATA, "EC_data.tsv")
ESM_PKL = os.path.join(DATA, "esm_uniprot_vec.pkl")
OUT_DIR = os.path.join(DATA, "function_pred_clean")

SEED = 42
PROJ_DIM = 128
EPOCHS = 100
LR = 5e-4
BATCH_SIZE = 256
TEMPERATURE = 0.1


# ---- reuse EC loading from eval_node_pred ----
sys.path.insert(0, CODE)
from eval_node_pred import (
    parse_uniprot_ec_tsv,
    build_label_index_level1,
    build_multilabel_vector_level1,
    split_data,
    load_embed_dict,
)


# ---- CLEAN projection head ----
class ProjectionHead(nn.Module):
    """MLP projection head: maps ESM embeddings to contrastive space."""

    def __init__(self, in_dim=1280, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=1)  # L2 normalize


# ---- SupCon-Hard loss (from CLEAN paper) ----
class SupConHardLoss(nn.Module):
    """
    Supervised contrastive loss with hard-negative mining.
    For each anchor, positives are proteins with the same EC label,
    negatives are proteins with different EC labels.
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: (N, D) L2-normalized embeddings
            labels: (N,) integer class labels (for single-label) or (N, C) multi-hot
        """
        device = features.device
        N = features.shape[0]

        # Compute pairwise similarity
        sim = torch.matmul(features, features.T) / self.temperature  # (N, N)

        # Build positive mask: same EC label
        if labels.dim() == 1:
            mask_pos = labels.unsqueeze(0) == labels.unsqueeze(1)
        else:
            # Multi-hot: positive if they share any label
            mask_pos = torch.matmul(labels.float(), labels.float().T) > 0

        mask_pos = mask_pos.float().to(device)
        # Remove self-loops
        mask_self = torch.eye(N, device=device)
        mask_pos = mask_pos * (1 - mask_self)

        # For numerical stability
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Log-sum-exp over all non-self entries (denominator)
        exp_sim = torch.exp(sim) * (1 - mask_self)
        log_sum_exp = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Mean of log(exp(sim) / sum_exp) over positive pairs
        log_prob = sim - log_sum_exp

        # Average over positive pairs per anchor
        n_pos = mask_pos.sum(dim=1)
        mean_log_prob = (mask_pos * log_prob).sum(dim=1) / (n_pos + 1e-8)

        # Only compute loss for anchors that have positives
        valid = n_pos > 0
        loss = -mean_log_prob[valid].mean()

        return loss


def train_contrastive(model, X_train, ec_labels_train, epochs=100, lr=5e-4,
                      batch_size=256, temperature=0.1, device="cpu"):
    """Train the projection head with SupCon-Hard loss."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = SupConHardLoss(temperature=temperature)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    L_t = torch.tensor(ec_labels_train, dtype=torch.long)

    dataset = TensorDataset(X_t, L_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for batch_x, batch_l in loader:
            batch_x = batch_x.to(device)
            batch_l = batch_l.to(device)

            z = model(batch_x)
            loss = criterion(z, batch_l)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:03d} | Loss: {total_loss/n_batches:.4f}")

    return model


def compute_ec_centers(model, X_train, full_ec_labels, device="cpu"):
    """
    Compute cluster centers for each unique full EC number in the training set.
    Returns dict: full_ec_string -> center_vector (in projected space).
    """
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        Z = model(X_t).cpu().numpy()

    centers = {}
    ec_to_indices = defaultdict(list)
    for i, ec_str in enumerate(full_ec_labels):
        ec_to_indices[ec_str].append(i)

    for ec_str, indices in ec_to_indices.items():
        center = Z[indices].mean(axis=0)
        center = center / (np.linalg.norm(center) + 1e-8)
        centers[ec_str] = center

    return centers


def predict_level1_scores(model, X_test, ec_centers, ec1_to_idx, device="cpu"):
    """
    For each test protein, compute distance to nearest EC center per level-1 class.
    Convert to scores: higher score = more likely to belong to that class.
    """
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        Z = model(X_t).cpu().numpy()

    n_test = len(X_test)
    n_classes = len(ec1_to_idx)
    scores = np.full((n_test, n_classes), -np.inf)

    # Group EC centers by level-1 class
    ec1_centers = defaultdict(list)
    for ec_str, center in ec_centers.items():
        if "." in ec_str:
            ec1 = ec_str.split(".")[0]
            if ec1 in ec1_to_idx:
                ec1_centers[ec1].append(center)

    for ec1, centers_list in ec1_centers.items():
        idx = ec1_to_idx[ec1]
        centers_arr = np.array(centers_list)  # (K, D)
        # Cosine similarity between each test protein and each center
        sims = Z @ centers_arr.T  # (N, K)
        # Take max similarity (min distance) as score for this class
        scores[:, idx] = sims.max(axis=1)

    # For classes with no centers, leave as -inf (will become 0 after sigmoid)
    # Apply sigmoid to convert to probability-like scores
    scores = 1 / (1 + np.exp(-scores * 5))  # scale factor for sharper probabilities

    return scores


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- load data ----
    prot_to_ec = parse_uniprot_ec_tsv(EC_TSV)
    emb = load_embed_dict(ESM_PKL)

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
    train_p, val_p, test_p = split_data(proteins, seed=SEED)
    print(f"Train: {len(train_p)}, Val: {len(val_p)}, Test: {len(test_p)}")

    # ---- prepare data ----
    in_dim = emb[proteins[0]].shape[0]

    X_train = np.array([emb[p] for p in train_p])
    X_val = np.array([emb[p] for p in val_p])
    X_test = np.array([emb[p] for p in test_p])

    y_train = np.array([build_multilabel_vector_level1(prot_to_ec[p], ec1_to_idx) for p in train_p])
    y_val = np.array([build_multilabel_vector_level1(prot_to_ec[p], ec1_to_idx) for p in val_p])
    y_test = np.array([build_multilabel_vector_level1(prot_to_ec[p], ec1_to_idx) for p in test_p])

    # For contrastive learning, assign each protein a single EC label
    # (use the first EC level-1 class; proteins with multiple are assigned to first)
    train_ec_labels = []
    train_full_ec = []  # full EC strings for center computation
    for p in train_p:
        ecs = prot_to_ec[p]
        # Get all full EC numbers
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
        epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE,
        temperature=TEMPERATURE, device=device
    )

    # ---- compute EC cluster centers from training set ----
    print("\nComputing EC cluster centers...")
    # Build full EC labels for all training proteins
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

    # ---- compute metrics (same as eval_node_pred.py) ----
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

    print(f"\nCLEAN TEST METRICS")
    print(f"AUC:     {macro_auc:.4f}")
    print(f"AUPRC:   {macro_auprc:.4f}")
    print(f"ACC:     {acc:.4f}")
    print(f"PREC:    {prec:.4f}")
    print(f"RECALL:  {rec:.4f}")
    print(f"F1:      {f1:.4f}")

    # ---- save results ----
    np.save(os.path.join(OUT_DIR, "test_probs.npy"), test_scores)
    np.save(os.path.join(OUT_DIR, "test_labels.npy"), y_test)
    np.save(os.path.join(OUT_DIR, "ec_classes.npy"), np.array(all_ec1))

    # Save test protein list for understudied evaluation
    with open(os.path.join(OUT_DIR, "test_proteins.pkl"), "wb") as f:
        pickle.dump(test_p, f)

    torch.save(model.state_dict(), os.path.join(OUT_DIR, "clean_projection.pt"))
    print(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
