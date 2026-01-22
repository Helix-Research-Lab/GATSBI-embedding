# GATSBI-embedding

Associated Zenodo data files: 

<img width="3359" height="1062" alt="figure_overview" src="https://github.com/user-attachments/assets/f9fa2252-cbc4-4c81-8ba6-8b1a3c4054fe" />

This repository contains code for GATSBI, a framework for learning protein embeddings using Graph Attention Networks (GATs) and biologically informed data splits.
The learned embeddings are evaluated across multiple downstream biological tasks, including interaction prediction, function prediction, and pathway-level inference.

Overview

GATSBI learns node embeddings on protein–protein interaction (PPI) graphs using:

- Graph Attention Networks (GATs) for message passing
- ESM protein language model embeddings for initialization
- Degree-matched negative sampling for robust link prediction
- Leakage-aware node and edge splits, including sequence-similarity constraints
The pipeline supports both node-split and edge-split training, followed by task-specific evaluation.

Repository Structure
```.
├── GATSBI_data_split.py        # Node splitting with sequence-similarity constraints
├── GATSBI_node_embed.py        # Node-split GAT training
├── GATSBI_edge_embed.py        # Edge-split GAT training
├── eval_node_pred.py           # Protein function (EC) prediction
├── eval_interaction_pred.py    # Protein–protein interaction prediction
├── eval_set_prediction.py      # Pathway / protein-set prediction
├── pinnacle.py                 # PINNACLE embedding post-processing
├── requirements.yml            # Conda environment specification
├── README.md
└── temp/                       # Intermediate outputs (optional)
```

---

## Installation
We recommend using **Conda**.

```bash
conda env create -f requirements.yml
conda activate <env_name>
```
---

## Data Requirements

You will need the following inputs:

1.Protein–protein interaction graph
- NetworkX .gpickle format (for node splitting)
2. Protein sequence embeddings
- Pickle file mapping UniProt ID → ESM embedding (1280-dim)
3. Sequence similarity matrix
- NumPy .npy matrix for similarity-aware node splits
4. Downstream task annotations
- Enzyme Code annotations (TSV)
- BioGRID interaction file (TSV)
- Reactome Pathway → protein-set mappings (Pickle)
---
## Step 1: Graph Splitting (Node and Edge Splits)
The script `GATSBI_data_split.py` performs leakage-aware graph splitting and produces both node splits and edge splits from a single input graph.

Key features:
- Sequence-similarity–aware node splitting
- Deterministic hashing for reproducibility
- Induced subgraph generation
- Standard train/val/test edge splits

Running `GATSBI_data_split.py` generates:

- Node split
```node_split_nodes.pkl
node_split_train_induced.edgelist.gz
node_split_val_induced.edgelist.gz
node_split_test_induced.edgelist.gz
```

- Edge split
```edge_split_train.edgelist.gz
edge_split_val.edgelist.gz
edge_split_test.edgelist.gz
```
Running the Split Script
```python GATSBI_data_split.py \
  --graph_path data/ppi_graph.gpickle \
  --seq_matrix data/seq_similarity.npy \
  --protein_list data/protein_ids.npy \
  --out_dir splits/ \
  --similarity_threshold 0.30
```
---
## Step 2: GAT Embedding Training

Choose one of the following training modes.

Node-Split Training (Inductive)
```python GATSBI_node_embed.py \
  --split_dir splits/ \
  --esm_path data/esm_uniprot_vec.pkl \
  --out_dir outputs/node_split
```

Edge-Split Training (Transductive)
```python GATSBI_edge_embed.py \
  --split_dir splits/ \
  --esm_path data/esm_uniprot_vec.pkl \
  --out_dir outputs/edge_split
```
---
## Step 3: Downstream Evaluation Tasks
### Protein Function Prediction (EC Level-1)
```python eval_node_pred.py \
  --embeddings outputs/node_split/gat_node_embeddings.pkl \
  --ec_tsv data/uniprot_ec.tsv \
  --out_dir results/function_pred
```
Multi-label classification

Macro ROC-AUC and AUPRC

### Protein–Protein Interaction Prediction
```python eval_interaction_pred.py \
  --biogrid data/BIOGRID-ALL.tsv \
  --embeddings outputs/node_split/gat_node_embeddings.pkl \
  --out_dir results/ppi_pred
```
Binary edge classification

ROC and precision–recall curves

### Functional Set Prediction
```python eval_set_prediction.py \
  --embeddings outputs/node_split/gat_node_embeddings.pkl \
  --pathways data/pathway_to_proteins.pkl \
  --out_dir results/pathway_pred
```
Attention-based pooling over protein sets

Corrupted-positive negative sampling

---
#### Key Methodological Features

- Multi-head Graph Attention Networks
- Degree-matched negative sampling
- Label smoothing for link prediction
- Sequence-similarity–aware splitting
- Centroid-normalized ESM initialization

#### License
Released under the MIT License.


