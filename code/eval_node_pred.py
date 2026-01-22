import csv
import pickle
import random
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from typing import Dict, Set, List
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
# 1. ARGPARSE
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="EC level-1 function prediction using protein embeddings")

    parser.add_argument("--embeddings", type=str, required=True,
                        help="Path to embedding dictionary (.pkl)")

    parser.add_argument("--ec_tsv", type=str, required=True,
                        help="Path to EC annotation TSV file")

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
# 2. Load EC annotations
# ---------------------------------------------------------
def parse_uniprot_ec_tsv(path: str) -> Dict[str, Set[str]]:
    df = pd.read_csv(path, sep="\t", dtype=str)
    print("Loaded EC TSV rows:", len(df))

    if "Entry" not in df.columns or "EC number" not in df.columns:
        raise ValueError("TSV must contain 'Entry' and 'EC number' columns.")

    df["EC number"] = df["EC number"].fillna("")
    df["EC_list"] = df["EC number"].apply(
        lambda x: [ec.strip() for ec in x.split(";") if ec.strip()]
    )

    return {acc: set(ecs) for acc, ecs in zip(df["Entry"], df["EC_list"])}


def build_label_index_level1(prot_to_ec: Dict[str, Set[str]]):
    all_ec1 = sorted({ec.split(".")[0] for ecs in prot_to_ec.values() for ec in ecs if "." in ec})
    ec1_to_idx = {ec1: i for i, ec1 in enumerate(all_ec1)}
    return ec1_to_idx, all_ec1


def build_multilabel_vector_level1(ec_set: Set[str], ec1_to_idx: Dict[str, int]):
    y = np.zeros(len(ec1_to_idx), dtype=np.float32)
    for ec in ec_set:
        if "." in ec:
            ec1 = ec.split(".")[0]
            if ec1 in ec1_to_idx:
                y[ec1_to_idx[ec1]] = 1.0
    return y


def split_data(proteins: List[str], train_ratio=0.8, val_ratio=0.1, seed=42):
    random.seed(seed)
    random.shuffle(proteins)
    n = len(proteins)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    return proteins[:n_train], proteins[n_train:n_train+n_val], proteins[n_train+n_val:]


# ---------------------------------------------------------
# 3. Load embeddings
# ---------------------------------------------------------
def load_embed_dict(path):
    with open(path, "rb") as f:
        emb = pickle.load(f)
    print(f"Loaded {len(emb)} embeddings")
    return emb


# ---------------------------------------------------------
# 4. Model
# ---------------------------------------------------------
class FunctionMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ---------------------------------------------------------
# 5. Macro ROC/PRC utilities
# ---------------------------------------------------------
def compute_macro_roc_prc(y_true, y_prob):
    valid = np.where(y_true.sum(axis=0) > 0)[0]

    fprs, tprs = [], []
    precs, recs = [], []

    for c in valid:
        fpr, tpr, _ = roc_curve(y_true[:, c], y_prob[:, c])
        prec, rec, _ = precision_recall_curve(y_true[:, c], y_prob[:, c])
        fprs.append(fpr)
        tprs.append(tpr)
        precs.append(prec)
        recs.append(rec)

    grid = np.linspace(0, 1, 200)
    mean_tpr = np.zeros_like(grid)
    mean_prec = np.zeros_like(grid)

    for fpr, tpr in zip(fprs, tprs):
        mean_tpr += np.interp(grid, fpr, tpr)

    for rec, prec in zip(recs, precs):
        mean_prec += np.interp(grid, rec[::-1], prec[::-1])

    mean_tpr /= len(fprs)
    mean_prec /= len(precs)

    return grid, mean_tpr, grid, mean_prec


def save_macro_curves(fpr, tpr, rec, prec, out_prefix):
    np.save(f"{out_prefix}_fpr.npy", fpr)
    np.save(f"{out_prefix}_tpr.npy", tpr)
    np.save(f"{out_prefix}_rec.npy", rec)
    np.save(f"{out_prefix}_prec.npy", prec)


def plot_macro_curves(fpr, tpr, rec, prec, auc, auprc, out_prefix, title):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC={auc:.3f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title} ROC")
    plt.legend()
    plt.savefig(f"{out_prefix}_roc.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(rec, prec, linewidth=2, label=f"AUPRC={auprc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title} PRC")
    plt.legend()
    plt.savefig(f"{out_prefix}_prc.png", dpi=200, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------
# 6. Training
# ---------------------------------------------------------
def train_model(model, X_train, y_train, X_val, y_val, out_dir, epochs=20, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pos_weight = torch.tensor((y_train.shape[0] - y_train.sum(axis=0)) / y_train.sum(axis=0)).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(X_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        opt.step()

        train_probs = torch.sigmoid(logits).detach().cpu().numpy()
        train_true = y_train.cpu().numpy()

        valid = np.where(train_true.sum(axis=0) > 0)[0]
        train_auc = np.mean([roc_auc_score(train_true[:, c], train_probs[:, c]) for c in valid])
        train_auprc = np.mean([average_precision_score(train_true[:, c], train_probs[:, c]) for c in valid])

        fpr_tr, tpr_tr, rec_tr, prec_tr = compute_macro_roc_prc(train_true, train_probs)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = loss_fn(val_logits, y_val)

        val_probs = torch.sigmoid(val_logits).detach().cpu().numpy()
        val_true = y_val.cpu().numpy()

        valid_val = np.where(val_true.sum(axis=0) > 0)[0]
        val_auc = np.mean([roc_auc_score(val_true[:, c], val_probs[:, c]) for c in valid_val])
        val_auprc = np.mean([average_precision_score(val_true[:, c], val_probs[:, c]) for c in valid_val])

        fpr_val, tpr_val, rec_val, prec_val = compute_macro_roc_prc(val_true, val_probs)

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train Loss={loss.item():.4f} | Val Loss={val_loss.item():.4f} | "
            f"Train AUC={train_auc:.4f} | Val AUC={val_auc:.4f} | "
            f"Train AUPRC={train_auprc:.4f} | Val AUPRC={val_auprc:.4f}"
        )

        prefix = f"{out_dir}/epoch{epoch+1}"
        save_macro_curves(fpr_tr, tpr_tr, rec_tr, prec_tr, prefix + "_train")
        save_macro_curves(fpr_val, tpr_val, rec_val, prec_val, prefix + "_val")

        plot_macro_curves(fpr_tr, tpr_tr, rec_tr, prec_tr, train_auc, train_auprc, prefix + "_train", "Train")
        plot_macro_curves(fpr_val, tpr_val, rec_val, prec_val, val_auc, val_auprc, prefix + "_val", "Val")

    torch.save(model.state_dict(), f"{out_dir}/function_mlp.pt")
    print("Saved trained model.")
    return model


# ---------------------------------------------------------
# 7. Evaluation
# ---------------------------------------------------------
def evaluate(model, X, y):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X)
        probs = torch.sigmoid(logits).cpu().numpy()
        y_true = y.cpu().numpy()

    valid = np.where(y_true.sum(axis=0) > 0)[0]

    aucs = [roc_auc_score(y_true[:, c], probs[:, c]) for c in valid]
    auprcs = [average_precision_score(y_true[:, c], probs[:, c]) for c in valid]

    macro_auc = np.mean(aucs)
    macro_auprc = np.mean(auprcs)

    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, average="micro", zero_division=0)
    rec = recall_score(y_true, preds, average="micro", zero_division=0)
    f1 = f1_score(y_true, preds, average="micro", zero_division=0)

    return macro_auc, macro_auprc, acc, prec, rec, f1, probs


# ---------------------------------------------------------
# 8. MAIN
# ---------------------------------------------------------
def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    prot_to_ec = parse_uniprot_ec_tsv(args.ec_tsv)
    emb = load_embed_dict(args.embeddings)

    proteins = [p for p, ecs in prot_to_ec.items() if len(ecs) > 0 and p in emb]
    print("Proteins with EC + embeddings:", len(proteins))

    ec1_to_idx, all_ec1 = build_label_index_level1(prot_to_ec)
    num_classes = len(all_ec1)
    print("EC level-1 classes:", all_ec1)

    X = np.array([emb[p] for p in proteins])
    Y = np.array([build_multilabel_vector_level1(prot_to_ec[p], ec1_to_idx) for p in proteins])

    train_p, val_p, test_p = split_data(proteins, seed=args.seed)

    X_train = np.array([emb[p] for p in train_p])
    y_train = np.array([build_multilabel_vector_level1(prot_to_ec[p], ec1_to_idx) for p in train_p])

    X_val = np.array([emb[p] for p in val_p])
    y_val = np.array([build_multilabel_vector_level1(prot_to_ec[p], ec1_to_idx) for p in val_p])

    X_test = np.array([emb[p] for p in test_p])
    y_test = np.array([build_multilabel_vector_level1(prot_to_ec[p], ec1_to_idx) for p in test_p])

    model = FunctionMLP(X_train.shape[1], num_classes)
    model = train_model(model, X_train, y_train, X_val, y_val,
                        out_dir=args.out_dir,
                        epochs=args.epochs,
                        lr=args.lr)

    auc, auprc, acc, prec, rec, f1, probs = evaluate(model, X_test, y_test)

    print("\nTEST METRICS")
    print(f"AUC:     {auc:.4f}")
    print(f"AUPRC:   {auprc:.4f}")
    print(f"ACC:     {acc:.4f}")
    print(f"PREC:    {prec:.4f}")
    print(f"RECALL:  {rec:.4f}")
    print(f"F1:      {f1:.4f}")

    torch.save(model.state_dict(), f"{args.out_dir}/function_mlp.pt")
    np.save(f"{args.out_dir}/test_probs.npy", probs)
    np.save(f"{args.out_dir}/test_labels.npy", y_test)
    np.save(f"{args.out_dir}/ec_classes.npy", np.array(all_ec1))


if __name__ == "__main__":
    main()
