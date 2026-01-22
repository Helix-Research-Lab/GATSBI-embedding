import os
import pickle
import random
import argparse

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1. ARGPARSE
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Interaction prediction using protein embeddings")

    parser.add_argument("--biogrid", type=str, required=True,
                        help="Path to BioGRID tab-delimited file")

    parser.add_argument("--embeddings", type=str, required=True,
                        help="Path to embedding dictionary (.pkl)")

    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory")

    parser.add_argument("--epochs", type=int, default=20,
                        help="Training epochs")

    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    return parser.parse_args()


# ---------------------------------------------------------
# 2. BIOGRID LOADING
# ---------------------------------------------------------
def first_swiss(x):
    if pd.isna(x):
        return None
    x = str(x).strip()
    if x in ("", "-", "nan", "None"):
        return None
    return x.split("|")[0]


def load_biogrid_data(file_path):
    df = pd.read_csv(file_path, sep="\t", low_memory=False)
    HUMAN = 9606

    human_ppi = df[
        (df["Organism ID Interactor A"] == HUMAN) &
        (df["Organism ID Interactor B"] == HUMAN)
    ].copy()

    print(f"Total interactions: {len(df)}")
    print(f"Human–human physical PPIs: {len(human_ppi)}")

    a = human_ppi["SWISS-PROT Accessions Interactor A"].map(first_swiss)
    b = human_ppi["SWISS-PROT Accessions Interactor B"].map(first_swiss)

    edges_df = pd.DataFrame({"uA": a, "uB": b}).dropna()
    edges_df = edges_df[edges_df["uA"] != edges_df["uB"]]

    edges_df["pair"] = edges_df.apply(
        lambda r: tuple(sorted((r["uA"], r["uB"]))), axis=1
    )
    edges_df = edges_df.drop_duplicates("pair").drop(columns=["pair"])

    edges = edges_df[["uA", "uB"]].values.tolist()

    print(f"Final Swiss-Prot human edges: {len(edges)}")
    print("Example:", edges[:5])

    return edges


# ---------------------------------------------------------
# 3. EMBEDDING LOADING
# ---------------------------------------------------------
def load_embed_dict(file_path):
    with open(file_path, "rb") as f:
        emb_by_name = pickle.load(f)
    print(f"Loaded {len(emb_by_name)} embeddings")
    return emb_by_name


# ---------------------------------------------------------
# 4. DATA SPLITTING + NEG SAMPLING
# ---------------------------------------------------------
def split_edges(edges, train_ratio=0.8, val_ratio=0.1, seed=42):
    random.seed(seed)
    edges = list(edges)
    random.shuffle(edges)

    n = len(edges)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    train_edges = edges[:n_train]
    val_edges = edges[n_train:n_train+n_val]
    test_edges = edges[n_train+n_val:]

    return train_edges, val_edges, test_edges


def negative_sample(edges, nodes, num_samples):
    edge_set = set((u, v) for u, v in edges)
    edge_set |= set((v, u) for u, v in edges)

    neg = []
    while len(neg) < num_samples:
        u = random.choice(nodes)
        v = random.choice(nodes)
        if u != v and (u, v) not in edge_set:
            neg.append((u, v))
    return neg


# ---------------------------------------------------------
# 5. FEATURE CONSTRUCTION
# ---------------------------------------------------------
def edge_to_feature(u, v, emb):
    return np.concatenate([
        emb[u] * emb[v],
        np.abs(emb[u] - emb[v]),
        (emb[u] - emb[v]) ** 2
    ])


def build_dataset(pos_edges, neg_edges, emb):
    X, y = [], []

    for u, v in pos_edges:
        if u in emb and v in emb:
            X.append(edge_to_feature(u, v, emb))
            y.append(1)

    for u, v in neg_edges:
        if u in emb and v in emb:
            X.append(edge_to_feature(u, v, emb))
            y.append(0)

    return np.array(X), np.array(y)


# ---------------------------------------------------------
# 6. MODEL
# ---------------------------------------------------------
class EdgeMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ---------------------------------------------------------
# 7. TRAINING LOOP
# ---------------------------------------------------------
def compute_metrics(logits, labels):
    probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
    labels = labels.detach().cpu().numpy().flatten()

    auc = roc_auc_score(labels, probs)
    auprc = average_precision_score(labels, probs)
    return auc, auprc


def save_curve_data(fpr, tpr, precision, recall, out_prefix):
    np.save(f"{out_prefix}_fpr.npy", fpr)
    np.save(f"{out_prefix}_tpr.npy", tpr)
    np.save(f"{out_prefix}_precision.npy", precision)
    np.save(f"{out_prefix}_recall.npy", recall)


def plot_roc_prc(fpr, tpr, precision, recall, out_prefix, title_suffix=""):
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve {title_suffix}")
    plt.savefig(f"{out_prefix}_roc.png")
    plt.close()

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve {title_suffix}")
    plt.savefig(f"{out_prefix}_prc.png")
    plt.close()


def train_dnn(model, X_train, y_train, X_val, y_val, out_dir, epochs=20, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        train_logits = model(X_train)
        train_loss = loss_fn(train_logits, y_train)
        train_loss.backward()
        opt.step()

        train_auc, train_auprc = compute_metrics(train_logits, y_train)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = loss_fn(val_logits, y_val)

        val_auc, val_auprc = compute_metrics(val_logits, y_val)

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train Loss: {train_loss.item():.4f} | "
            f"Val Loss: {val_loss.item():.4f} | "
            f"Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} | "
            f"Train AUPRC: {train_auprc:.4f} | Val AUPRC: {val_auprc:.4f}"
        )

    torch.save(model.state_dict(), f"{out_dir}/trained_edge_mlp.pt")
    print("Saved trained model.")


# ---------------------------------------------------------
# 8. MAIN
# ---------------------------------------------------------
def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    edges = load_biogrid_data(args.biogrid)
    emb = load_embed_dict(args.embeddings)

    nodes = list(emb.keys())

    train_pos, val_pos, test_pos = split_edges(edges, seed=args.seed)

    train_neg = negative_sample(train_pos, nodes, len(train_pos))
    val_neg   = negative_sample(val_pos, nodes, len(val_pos))
    test_neg  = negative_sample(test_pos, nodes, len(test_pos))

    X_train, y_train = build_dataset(train_pos, train_neg, emb)
    X_val,   y_val   = build_dataset(val_pos,   val_neg,   emb)
    X_test,  y_test  = build_dataset(test_pos,  test_neg,  emb)

    print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

    in_dim = X_train.shape[1]
    model = EdgeMLP(in_dim)

    train_dnn(model, X_train, y_train, X_val, y_val,
              out_dir=args.out_dir,
              epochs=args.epochs,
              lr=args.lr)

    print("Evaluating on test set...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        test_logits = model(X_test_t)

    test_probs = torch.sigmoid(test_logits).detach().cpu().numpy().flatten()
    test_labels = y_test_t.detach().cpu().numpy().flatten()

    test_auc = roc_auc_score(test_labels, test_probs)
    test_auprc = average_precision_score(test_labels, test_probs)

    print(f"TEST AUC: {test_auc:.4f} | TEST AUPRC: {test_auprc:.4f}")

    fpr_t, tpr_t, _ = roc_curve(test_labels, test_probs)
    prec_t, rec_t, _ = precision_recall_curve(test_labels, test_probs)

    save_curve_data(fpr_t, tpr_t, prec_t, rec_t, f"{args.out_dir}/test")
    plot_roc_prc(fpr_t, tpr_t, prec_t, rec_t, f"{args.out_dir}/test", "Test")

    np.save(f"{args.out_dir}/test_probs.npy", test_probs)
    np.save(f"{args.out_dir}/test_labels.npy", test_labels)


if __name__ == "__main__":
    main()
