# new_arch_v2.py (highly abridged, focusing on the major changes)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

import chemprop
import pkgutil
# from chemprop.models import MoleculeModel  # chemprop’s D-MPNN
try:
    from chemprop.models import MoleculeModel
except ImportError:
    from chemprop.models.model import MoleculeModel

import inspect
print(inspect.signature(MoleculeModel.__init__))

from chemprop.data import MoleculeDataset, MoleculeDataLoader
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
import numpy as np

# -----------------------------------------------------------------------------
# 1) Diagnostics & Weighted Loss
# -----------------------------------------------------------------------------
class WeightedBCE(nn.Module):
    def __init__(self, pos_weights: torch.Tensor):
        super().__init__()
        self.pos_weights = pos_weights  # shape = (num_labels,)
    def forward(self, logits, targets):
        # logits: (B, L), targets: (B, L)
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weights.to(logits.device), reduction="mean"
        )
        return loss

# -----------------------------------------------------------------------------
# 2) Enhanced Node + Edge Featurization
# -----------------------------------------------------------------------------
def featurize_smiles_enhanced(smiles: str, radius3d: float):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    # 2A) 3D embedding
    coords = None
    try:
        if radius3d > 0:
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            if mol.GetNumConformers() > 0:
                coords = torch.tensor(mol.GetConformer(0).GetPositions(), dtype=torch.float32)
    except:
        coords = None

    # 2B) Node features: [one-hot Z (100)] ⊕ [one-hot Hybrid (5)] ⊕ [formal_charge (1)] ⊕ [numHs (1)] ⊕ [aromatic (1)]
    node_feats = []
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        one_hot_z = [0]*100
        one_hot_z[z-1 if 1<=z<=100 else 0] = 1

        hyb = atom.GetHybridization()
        hyb_map = {
            Chem.rdchem.HybridizationType.SP: 0,
            Chem.rdchem.HybridizationType.SP2: 1,
            Chem.rdchem.HybridizationType.SP3: 2,
            Chem.rdchem.HybridizationType.SP3D: 3,
            Chem.rdchem.HybridizationType.SP3D2: 4
        }
        one_hot_hyb = [0]*5
        one_hot_hyb[hyb_map.get(hyb, 0)] = 1

        charge = atom.GetFormalCharge()
        num_hs = atom.GetTotalNumHs()
        aro = 1 if atom.GetIsAromatic() else 0

        node_feats.append(one_hot_z + one_hot_hyb + [charge, num_hs, aro])

    x = torch.tensor(node_feats, dtype=torch.float32)  # (num_atoms, 100+5+1+1+1 = 108)

    # 2C) Edge features: bond type one-hot
    num_atoms = mol.GetNumAtoms()
    E_dim = 4  # single, double, triple, aromatic
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        btype = bond.GetBondType()
        if btype == Chem.rdchem.BondType.SINGLE:
            bo = [1,0,0,0]
        elif btype == Chem.rdchem.BondType.DOUBLE:
            bo = [0,1,0,0]
        elif btype == Chem.rdchem.BondType.TRIPLE:
            bo = [0,0,1,0]
        elif btype == Chem.rdchem.BondType.AROMATIC:
            bo = [0,0,0,1]
        else:
            bo = [0,0,0,0]

        edge_index.append((i,j)); edge_attr.append(bo)
        edge_index.append((j,i)); edge_attr.append(bo)  # undirected

    edge_index = torch.tensor(edge_index, dtype=torch.long).t()  # (2, num_edges)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)   # (num_edges, 4)

    mask = torch.ones(x.shape[0], dtype=torch.bool)
    return {"x": x, "pos": coords, "mask": mask, "edge_index": edge_index, "edge_attr": edge_attr}

# -----------------------------------------------------------------------------
# 3) Dataset with Chemprop D-MPNN + Extra Descriptors + Transformer Embedding
# -----------------------------------------------------------------------------
class OdourDatasetEnhanced(Dataset):
    def __init__(self, df: pd.DataFrame, cfg):
        self.df = df.reset_index(drop=True)
        self.smiles_col = cfg.data.smiles_column
        self.label_cols = cfg.data.label_columns
        self.radius3d = cfg.data.radius
        self.use_fp = cfg.model.use_fingerprint
        self.fp_dim = cfg.model.fp_dim
        # Compute label frequencies for weighting
        label_counts = (df[self.label_cols] > 0).sum(axis=0).values
        neg_counts = len(df) - label_counts
        pos_weights = neg_counts / np.clip(label_counts, a_min=1, a_max=None)
        self.pos_weights = torch.tensor(pos_weights, dtype=torch.float32)
        # Precompute ChemBERTa embeddings (offline) if desired
        if cfg.model.use_transformer:
            self.transformer_embs = torch.load(cfg.embedding.precomputed_path)
        else:
            self.transformer_embs = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = row[self.smiles_col]
        fea = featurize_smiles_enhanced(smiles, self.radius3d)
        x = fea["x"]; mask = fea["mask"]
        edge_index = fea["edge_index"]; edge_attr = fea["edge_attr"]
        pos = fea["pos"]  # may be None

        # Per-label targets (float32)
        y = torch.tensor(row[self.label_cols].values.astype("float32"), dtype=torch.float32)

        # Extra global descriptors
        mol = Chem.MolFromSmiles(smiles)
        descs = torch.tensor([
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.MolWt(mol),
            Descriptors.NumRotatableBonds(mol),
        ], dtype=torch.float32)  # (4,)

        # Morgan fingerprint if desired
        if self.use_fp:
            rv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=self.fp_dim)
            fp = torch.tensor([int(c) for c in rv.ToBitString()], dtype=torch.float32)
        else:
            fp = None

        # Transformer embedding if provided
        if self.transformer_embs is not None:
            trans_emb = self.transformer_embs[idx]  # (768,)
        else:
            trans_emb = None

        return {
            "x": x,                 # (num_atoms, node_dim)
            "pos": pos,             # (num_atoms, 3) or None
            "mask": mask,           # (num_atoms,)
            "edge_index": edge_index,  # (2, num_edges)
            "edge_attr": edge_attr,    # (num_edges, 4)
            "y": y,                 # (num_labels,)
            "descs": descs,         # (4,)
            "fp": fp,               # (fp_dim,) or None
            "trans_emb": trans_emb  # (768,) or None
        }

# -----------------------------------------------------------------------------
# 4) Collate_fn for D-MPNN Variant
# -----------------------------------------------------------------------------
def collate_mpnn(batch: list):
    # We’ll use chemprop’s MoleculeDataLoader, which handles batching for us.
    # But if we choose to stick with PyTorch Geometric, we’d write a custom collate.
    raise NotImplementedError("Use chemprop’s MoleculeDataLoader instead")

# -----------------------------------------------------------------------------
# 5) Model: D-MPNN Backbone + Fusion Heads
# -----------------------------------------------------------------------------
class OdourPredictorEnhanced(nn.Module):
    def __init__(self, cfg, num_labels, pos_weights: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.num_labels = num_labels
        self.pos_weights = pos_weights

        ### 5A) D-MPNN (from chemprop) as backbone
        # self.mpnn = MoleculeModel(
        #     num_tasks=num_labels,  # Chemprop allows multi‐task, but we’ll override final head
        #     hidden_size=cfg.model.hidden_dim,  # typically 300
        #     depth=cfg.model.mpnn_layers,       # e.g. 3–4
        #     dropout=cfg.model.dropout,
        #     ff_hidden_size=cfg.model.ff_hidden, # feed‐forward head size for D-MPNN
        #     #-------------------------------------------------------------------------
        #     # **We don’t actually use chemprop’s final head since we want to fuse extra features.**
        #     # We will do: MPNN → pooling → [fusion with descs, fp, trans_emb] → our own MLP.
        #     #-------------------------------------------------------------------------
        # )
        self.mpnn = MoleculeModel(
            hidden_size=cfg.model.hidden_dim,     # e.g. 128
            depth=cfg.model.mpnn_layers,           # e.g. 3
            dropout=cfg.model.dropout,             # e.g. 0.2
            ff_hidden_size=cfg.model.ff_hidden,    # e.g. 256
            bias=True,                             # or False, depending on your preference
            atom_fdim=39,                          # default Chemprop atom feature dim
            bond_fdim=10                           # default Chemprop bond feature dim
        )

        ### 5B) Extra heads for global descriptors, fingerprint, transformer
        feat_dim = cfg.model.hidden_dim               # MPNN pool → hidden_dim
        if cfg.model.use_fingerprint:
            self.fp_mlp = nn.Sequential(
                nn.Linear(cfg.model.fp_dim, cfg.model.fp_hidden),
                nn.ReLU(),
                nn.Dropout(cfg.model.dropout)
            )
            feat_dim += cfg.model.fp_hidden

        if cfg.model.use_descs:
            self.descs_mlp = nn.Sequential(
                nn.Linear(4, cfg.model.descs_hidden),
                nn.ReLU(),
                nn.Dropout(cfg.model.dropout)
            )
            feat_dim += cfg.model.descs_hidden

        if cfg.model.use_transformer:
            self.trans_mlp = nn.Sequential(
                nn.Linear(768, cfg.model.trans_hidden),
                nn.ReLU(),
                nn.Dropout(cfg.model.dropout)
            )
            feat_dim += cfg.model.trans_hidden

        ### 5C) Final MLP head
        layers = []
        in_dim = feat_dim
        for h in cfg.model.mlp_hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(cfg.model.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, num_labels))
        self.head = nn.Sequential(*layers)

        # Weighted loss will be applied externally in training loop

    def forward(self, mol_batch, descs=None, fp=None, trans_emb=None):
        """
        mol_batch: the MoleculeBatch object from chemprop, which internally handles
                   atom/bond indices, etc. The chemprop model returns a (B, hidden_dim)
                   graph‐pooled vector if you call model(mol_batch, features_only=True).
        descs:     (B, 4) optional  
        fp:        (B, fp_dim) optional  
        trans_emb: (B, 768) optional
        """
        # 1) Get MPNN embedding
        mpnn_out = self.mpnn(mol_batch, features_only=True)  # (B, hidden_dim)

        h = mpnn_out
        if fp is not None:
            h_fp = self.fp_mlp(fp)  # (B, fp_hidden)
            h = torch.cat([h, h_fp], dim=1)

        if descs is not None:
            h_descs = self.descs_mlp(descs)  # (B, descs_hidden)
            h = torch.cat([h, h_descs], dim=1)

        if trans_emb is not None:
            h_trans = self.trans_mlp(trans_emb)  # (B, trans_hidden)
            h = torch.cat([h, h_trans], dim=1)

        out = self.head(h)  # (B, num_labels)
        return out

# -----------------------------------------------------------------------------
# 6) Training Loop with Weighted Loss, Early Stopping
# -----------------------------------------------------------------------------
def train_loop(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.train.use_cuda else "cpu")
    df = pd.read_csv(cfg.data.input_csv)
    X = df[[cfg.data.smiles_column] + cfg.data.label_columns].copy()

    # Multi‐label stratified split
    X_np = X[cfg.data.smiles_column].values.reshape(-1, 1)
    y_np = X[cfg.data.label_columns].values.astype(np.int32)
    X_train, y_train, X_val, y_val = iterative_train_test_split(X_np, y_np, test_size=1-cfg.data.train_test_split_ratio)

    # Reconstruct DataFrames
    df_train = pd.DataFrame(
        data=np.hstack([X_train, y_train]), 
        columns=[cfg.data.smiles_column] + cfg.data.label_columns
    )
    df_val = pd.DataFrame(
        data=np.hstack([X_val, y_val]),
        columns=[cfg.data.smiles_column] + cfg.data.label_columns
    )

    # Datasets
    ds_train = OdourDatasetEnhanced(df_train, cfg)
    ds_val   = OdourDatasetEnhanced(df_val, cfg)

    # Compute pos_weights from training set only
    pos_weights = ds_train.pos_weights  # (num_labels,)

    # DataLoaders (using chemprop’s MoleculeDataLoader)
    # mol_data_train = MoleculeDataset.from_smiles_list(df_train[cfg.data.smiles_column].tolist(),
    #                                                  labels=df_train[cfg.data.label_columns].values.tolist())
    # mol_data_val   = MoleculeDataset.from_smiles_list(df_val[cfg.data.smiles_column].tolist(),
    #                                                  labels=df_val[cfg.data.label_columns].values.tolist())
    from chemprop.data import MoleculeDatapoint, MoleculeDataset

    # Build a list of MoleculeDatapoint instances for the training set:
    train_datapoints = []
    for smiles, *label_vals in zip(
            df_train[cfg.data.smiles_column],
            *[df_train[col] for col in cfg.data.label_columns]
        ):
        # `label_vals` is a tuple of as many elements as there are labels; convert to floats
        targets = [float(v) for v in label_vals]
        train_datapoints.append(
            MoleculeDatapoint(smiles=smiles, targets=targets)
        )

    mol_data_train = MoleculeDataset(train_datapoints)

    # Likewise for validation:
    val_datapoints = []
    for smiles, *label_vals in zip(
            df_val[cfg.data.smiles_column],
            *[df_val[col] for col in cfg.data.label_columns]
        ):
        targets = [float(v) for v in label_vals]
        val_datapoints.append(
            MoleculeDatapoint(smiles=smiles, targets=targets)
        )

    mol_data_val = MoleculeDataset(val_datapoints)



    loader_train = MoleculeDataLoader(mol_data_train, batch_size=cfg.train.batch_size, shuffle=True)
    loader_val   = MoleculeDataLoader(mol_data_val, batch_size=cfg.train.batch_size, shuffle=False)

    # Model + optimizer + scheduler
    model = OdourPredictorEnhanced(cfg, num_labels=len(cfg.data.label_columns), pos_weights=pos_weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    criterion = WeightedBCE(pos_weights)

    best_val_auc = 0.0
    no_improve = 0

    for epoch in range(cfg.train.epochs):
        model.train()
        total_loss = 0.0
        for batch in loader_train:
            smiles_list, y_list = batch.smiles(), batch.labels()
            # Construct the PyTorch Geometric or chemprop batch objects as needed
            optimizer.zero_grad()
            # Forward pass
            logits = model(batch, 
                           descs=len(batch)*[None],  # or pre-extracted, shaped (B,4)
                           fp=len(batch)*[None], 
                           trans_emb=len(batch)*[None])  
            y_true = torch.tensor(np.array(y_list), dtype=torch.float32, device=device)
            loss = criterion(logits, y_true)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)

        # Validation
        model.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for batch in loader_val:
                logits = model(batch, descs=None, fp=None, trans_emb=None)
                all_logits.append(logits.cpu())
                all_labels.append(torch.tensor(np.array(batch.labels()), dtype=torch.float32))
        all_logits = torch.cat(all_logits, dim=0).numpy()  # (N_val, L)
        all_labels = torch.cat(all_labels, dim=0).numpy()  # (N_val, L)

        # Compute per-label AUC
        from sklearn.metrics import roc_auc_score, average_precision_score
        aucs = []
        aps = []
        for j in range(all_labels.shape[1]):
            if len(np.unique(all_labels[:, j])) > 1:
                aucs.append(roc_auc_score(all_labels[:, j], all_logits[:, j]))
                aps.append(average_precision_score(all_labels[:, j], all_logits[:, j]))
        mean_auc = float(np.mean(aucs))
        mean_ap  = float(np.mean(aps))

        scheduler.step(1 - mean_auc)  # we want to maximize AUC, so step on (1−AUC)

        if mean_auc > best_val_auc:
            best_val_auc = mean_auc
            no_improve = 0
            torch.save(model.state_dict(), cfg.output.model_path)
        else:
            no_improve += 1

        print(f"Epoch {epoch+1:02d} | Train Loss: {total_loss/len(ds_train):.4f}  Val AUC: {mean_auc:.4f}  Val AP: {mean_ap:.4f}")

        if no_improve >= cfg.train.early_stopping_patience:
            print("Early stopping triggered.")
            break

    print(f"Best validation AUC: {best_val_auc:.4f}")

if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig
    @hydra.main(config_path="conf", config_name="config3", version_base=None)
    def main(cfg: DictConfig):
        train_loop(cfg)
    main()
