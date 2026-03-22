#!/usr/bin/env python3
"""
D-SCRIPT baseline for interaction prediction.
Uses the same BioGRID edge split (seed=42, 80/10/10) as GATSBI/PINNACLE evaluations.

D-SCRIPT (Sledzieski et al., Cell Systems 2021) predicts protein-protein interactions
directly from amino acid sequences using a learned contact module on top of the
Bepler & Berger protein language model.

Workflow:
  1. Load BioGRID edges and split (reusing eval_interaction_pred logic)
  2. Write train/val/test TSV files for D-SCRIPT
  3. Run D-SCRIPT embedding, training, and prediction via subprocess CLI
  4. Parse predictions and save test_probs.npy / test_labels.npy
"""

import argparse
import os
import sys
import pickle
import random
import subprocess
import tempfile

import numpy as np
import pandas as pd

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

BIOGRID = os.path.join(DATA, "biogrid.txt")
FASTA = os.path.join(DATA, "all_proteins.fasta")
OUT_DIR = os.path.join(DATA, "interaction_pred_dscript")
DSCRIPT_DIR = os.path.join(DATA, "dscript_work")

SEED = 42


# ---- reuse BioGRID loading from eval_interaction_pred ----
sys.path.insert(0, CODE)
from eval_interaction_pred import load_biogrid_data, split_edges, negative_sample


def load_fasta_mapping(fasta_path):
    """Return (set of accession IDs, dict mapping accession -> full FASTA header name).

    D-SCRIPT stores embeddings keyed by the full FASTA header (e.g. 'sp|P12345|NAME_HUMAN'),
    so we need to map our accession IDs to those full names for the TSV files.
    """
    acc_to_full = {}
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                full_name = line[1:].strip().split()[0]  # e.g. sp|P12345|NAME_HUMAN
                parts = full_name.split("|")
                if len(parts) >= 2:
                    acc = parts[1]
                else:
                    acc = full_name
                acc_to_full[acc] = full_name
    return set(acc_to_full.keys()), acc_to_full


def write_pairs_tsv(pairs, labels, path, acc_to_full=None):
    """Write protein pairs + labels as TSV for D-SCRIPT.
    If acc_to_full is provided, convert accessions to full FASTA header names.
    """
    with open(path, "w") as f:
        for (u, v), lab in zip(pairs, labels):
            u_name = acc_to_full.get(u, u) if acc_to_full else u
            v_name = acc_to_full.get(v, v) if acc_to_full else v
            f.write(f"{u_name}\t{v_name}\t{lab}\n")
    print(f"  Wrote {len(pairs)} pairs to {path}")


def run_cmd(cmd, desc=""):
    """Run a shell command and check for errors."""
    print(f"  Running: {desc or cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDOUT: {result.stdout[:3000]}")
        print(f"  STDERR: {result.stderr[:3000]}")
        raise RuntimeError(f"Command failed (exit {result.returncode}): {cmd}")
    return result.stdout


def parse_args():
    parser = argparse.ArgumentParser(description="D-SCRIPT interaction prediction baseline")
    parser.add_argument("--max-train-pairs", type=int, default=None,
                        help="Limit positive training pairs (negatives matched). Default: use all.")
    parser.add_argument("--num-epochs", type=int, default=10,
                        help="Number of D-SCRIPT training epochs (default: 10)")
    parser.add_argument("--work-dir", type=str, default=None,
                        help="D-SCRIPT working directory (default: data/dscript_work)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory for results (default: data/interaction_pred_dscript)")
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    out_dir = args.out_dir or OUT_DIR
    dscript_dir = args.work_dir or DSCRIPT_DIR

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(dscript_dir, exist_ok=True)

    # ---- check FASTA exists ----
    if not os.path.exists(FASTA):
        print(f"ERROR: {FASTA} not found. Run download_sequences.py first.")
        sys.exit(1)

    fasta_ids, acc_to_full = load_fasta_mapping(FASTA)
    print(f"FASTA contains {len(fasta_ids)} protein sequences")

    # ---- load edges and split (same as GATSBI/PINNACLE) ----
    edges = load_biogrid_data(BIOGRID)

    # Filter to proteins with sequences
    edges = [(u, v) for u, v in edges if u in fasta_ids and v in fasta_ids]
    print(f"Edges with sequences: {len(edges)}")

    nodes = list(fasta_ids)
    train_pos, val_pos, test_pos = split_edges(edges, seed=SEED)

    # ---- subsample training pairs if requested ----
    if args.max_train_pairs and args.max_train_pairs < len(train_pos):
        print(f"Subsampling training pairs: {len(train_pos)} -> {args.max_train_pairs}")
        idx = np.random.choice(len(train_pos), args.max_train_pairs, replace=False)
        train_pos = [train_pos[i] for i in idx]

    train_neg = negative_sample(train_pos, nodes, len(train_pos))
    val_neg = negative_sample(val_pos, nodes, len(val_pos))
    test_neg = negative_sample(test_pos, nodes, len(test_pos))

    print(f"Train: {len(train_pos)} pos + {len(train_neg)} neg")
    print(f"Val:   {len(val_pos)} pos + {len(val_neg)} neg")
    print(f"Test:  {len(test_pos)} pos + {len(test_neg)} neg")

    # ---- write TSV files (using full FASTA header names for D-SCRIPT) ----
    train_tsv = os.path.join(dscript_dir, "train.tsv")
    val_tsv = os.path.join(dscript_dir, "val.tsv")
    test_tsv = os.path.join(dscript_dir, "test.tsv")

    train_pairs = train_pos + train_neg
    train_labels = [1] * len(train_pos) + [0] * len(train_neg)
    write_pairs_tsv(train_pairs, train_labels, train_tsv, acc_to_full)

    val_pairs = val_pos + val_neg
    val_labels = [1] * len(val_pos) + [0] * len(val_neg)
    write_pairs_tsv(val_pairs, val_labels, val_tsv, acc_to_full)

    test_pairs = test_pos + test_neg
    test_labels = [1] * len(test_pos) + [0] * len(test_neg)
    write_pairs_tsv(test_pairs, test_labels, test_tsv, acc_to_full)

    # ---- D-SCRIPT embedding ----
    emb_h5 = os.path.join(dscript_dir, "embeddings.h5")
    if not os.path.exists(emb_h5):
        run_cmd(
            f"dscript embed --seqs {FASTA} --outfile {emb_h5}",
            "D-SCRIPT sequence embedding"
        )
    else:
        print(f"  Embeddings already exist at {emb_h5}")

    # ---- D-SCRIPT training ----
    model_prefix = os.path.join(dscript_dir, "dscript_model")
    model_file = f"{model_prefix}.sav"
    if not os.path.exists(model_file):
        run_cmd(
            f"dscript train --train {train_tsv} --test {val_tsv} "
            f"--embedding {emb_h5} --save-prefix {model_prefix} "
            f"-d 0 --num-epochs {args.num_epochs}",
            f"D-SCRIPT model training ({args.num_epochs} epochs)"
        )
    else:
        print(f"  Model already exists at {model_file}")

    # ---- D-SCRIPT prediction ----
    pred_out = os.path.join(dscript_dir, "test_predictions.tsv")
    # Write test pairs without labels for prediction (using full FASTA names)
    test_pairs_only = os.path.join(dscript_dir, "test_pairs.tsv")
    with open(test_pairs_only, "w") as f:
        for u, v in test_pairs:
            f.write(f"{acc_to_full.get(u, u)}\t{acc_to_full.get(v, v)}\n")

    run_cmd(
        f"dscript predict --pairs {test_pairs_only} "
        f"--embedding {emb_h5} --model {model_file} "
        f"--outfile {pred_out} -d 0",
        "D-SCRIPT prediction on test set"
    )

    # ---- parse predictions ----
    pred_df = pd.read_csv(pred_out, sep="\t", header=None,
                          names=["protA", "protB", "score"])

    # Build reverse mapping: full_name -> accession
    full_to_acc = {v: k for k, v in acc_to_full.items()}

    # Map predictions back to test labels (D-SCRIPT outputs full names)
    pair_to_score = {}
    for _, row in pred_df.iterrows():
        a = full_to_acc.get(row["protA"], row["protA"])
        b = full_to_acc.get(row["protB"], row["protB"])
        pair_to_score[(a, b)] = row["score"]

    test_probs = []
    test_labels_final = []
    for (u, v), lab in zip(test_pairs, test_labels):
        if (u, v) in pair_to_score:
            test_probs.append(pair_to_score[(u, v)])
            test_labels_final.append(lab)
        elif (v, u) in pair_to_score:
            test_probs.append(pair_to_score[(v, u)])
            test_labels_final.append(lab)

    test_probs = np.array(test_probs)
    test_labels_final = np.array(test_labels_final)

    print(f"\nPredictions matched: {len(test_probs)}/{len(test_pairs)}")

    # ---- compute metrics ----
    auc = roc_auc_score(test_labels_final, test_probs)
    auprc = average_precision_score(test_labels_final, test_probs)
    preds_binary = (test_probs >= 0.5).astype(int)
    acc = accuracy_score(test_labels_final, preds_binary)
    prec = precision_score(test_labels_final, preds_binary, zero_division=0)
    rec = recall_score(test_labels_final, preds_binary, zero_division=0)
    f1 = f1_score(test_labels_final, preds_binary, zero_division=0)

    print(f"\nD-SCRIPT TEST METRICS")
    print(f"AUC:     {auc:.4f}")
    print(f"AUPRC:   {auprc:.4f}")
    print(f"ACC:     {acc:.4f}")
    print(f"PREC:    {prec:.4f}")
    print(f"RECALL:  {rec:.4f}")
    print(f"F1:      {f1:.4f}")

    # ---- save results ----
    np.save(os.path.join(out_dir, "test_probs.npy"), test_probs)
    np.save(os.path.join(out_dir, "test_labels.npy"), test_labels_final)

    # Save test pairs for understudied evaluation
    with open(os.path.join(out_dir, "test_pairs.pkl"), "wb") as f:
        pickle.dump(list(zip(
            [p[0] for p in test_pairs[:len(test_probs)]],
            [p[1] for p in test_pairs[:len(test_probs)]]
        )), f)

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
