import ast
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
import pickle

# ----------------------------------------------------
# Paths
# ----------------------------------------------------
EMBED_PATH  = "../data/pinnacle_protein_embed.pth"
LABEL_PATH  = "../data/pinnacle_protein_labels_dict.txt"
MAP_PATH    = "../data/uniprot_genename.tsv"
OUT_PATH    = "../data/pinnacle_protein_emb_uniprot.pkl"

# ----------------------------------------------------
# 1) Load embedding blocks
# ----------------------------------------------------
emb_obj = torch.load(EMBED_PATH, map_location="cpu")

blocks = []
for k in sorted(emb_obj.keys()):
    v = emb_obj[k]
    if torch.is_tensor(v) and v.ndim == 2 and v.shape[1] == 128:
        blocks.append(v)

E = torch.cat(blocks, dim=0).numpy()   # [394760, 128]
print("Total embedding rows:", E.shape)

# ----------------------------------------------------
# 2) Load labels dict (python-literal txt)
# ----------------------------------------------------
with open(LABEL_PATH, "r") as f:
    txt = f.read()

labels = ast.literal_eval(txt)

names = labels["Name"]
cell_types = set(labels["Cell Type"])

# Keep only protein rows (Name not equal to a cell type)
prot_names = [n for n in names if n not in cell_types]

print("Total label rows:", len(names))
print("Protein rows:", len(prot_names))

assert len(prot_names) == E.shape[0], "Row mismatch between labels and embeddings!"

# ----------------------------------------------------
# 3) Aggregate per protein (mean across contexts)
# ----------------------------------------------------
prot_sum = defaultdict(lambda: np.zeros(128, dtype=np.float64))
prot_cnt = defaultdict(int)

for i, pname in enumerate(prot_names):
    prot_sum[pname] += E[i]
    prot_cnt[pname] += 1

prot_emb_gene = {
    p: (prot_sum[p] / prot_cnt[p]).astype(np.float32)
    for p in prot_sum
}

print("Unique gene/protein names before mapping:", len(prot_emb_gene))

# ----------------------------------------------------
# 4) Load gene → UniProt mapping
# ----------------------------------------------------
df = pd.read_csv(MAP_PATH, sep="\t", dtype=str)

# Build mapping from any gene symbol to Entry
gene2uniprot = {}

for _, row in df.iterrows():
    entry = row["Entry"]
    genes = row.get("Gene Names", "")

    if isinstance(genes, str):
        for g in genes.split():
            gene2uniprot[g] = entry

print("Mapping entries loaded:", len(gene2uniprot))

# ----------------------------------------------------
# 5) Convert to UniProt IDs and drop missing
# ----------------------------------------------------
prot_emb_uniprot = {}

for gene, vec in prot_emb_gene.items():
    if gene in gene2uniprot:
        uniprot = gene2uniprot[gene]
        prot_emb_uniprot[uniprot] = vec

print("Final nodes with UniProt IDs:", len(prot_emb_uniprot))

# ----------------------------------------------------
# 6) Save dictionary
# ----------------------------------------------------
with open(OUT_PATH, "wb") as f:
    pickle.dump(prot_emb_uniprot, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Saved to:", OUT_PATH)
