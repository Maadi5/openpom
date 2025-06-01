# new_arch_v5.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import numpy as np

import hydra
from omegaconf import DictConfig

# ─────────────────────────────────────────────────────────────────────────────
# For stratified multi-label splitting:
from skmultilearn.model_selection import iterative_train_test_split
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# EGNN backbone:
from egnn_pytorch import EGNN
# ─────────────────────────────────────────────────────────────────────────────


# -----------------------------------------------------------------------------
# 1) Weighted BCE for imbalanced multi-label
# -----------------------------------------------------------------------------
class WeightedBCE(nn.Module):
    def __init__(self, pos_weights: torch.Tensor):
        super().__init__()
        self.pos_weights = pos_weights

    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weights.to(logits.device),
            reduction="mean"
        )


# -----------------------------------------------------------------------------
# 2) SMILES → rich node/edge features
# -----------------------------------------------------------------------------
def featurize_smiles_enhanced(smiles: str, radius3d: float):
    """
    Returns dict with:
      'x':         (num_atoms, 108) FloatTensor
      'pos':       (num_atoms,  3) FloatTensor or None
      'mask':      (num_atoms,)     BoolTensor
      'edge_index':(2, num_edges)   LongTensor
      'edge_attr': (num_edges, 4)   FloatTensor (bond type one-hot)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)

    # 2A) 3D embedding via ETKDG
    coords = None
    try:
        if radius3d > 0:
            res = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            if res == 0 and mol.GetNumConformers() > 0:
                coords_np = mol.GetConformer(0).GetPositions()
                coords = torch.tensor(coords_np, dtype=torch.float32)
    except Exception:
        coords = None

    # 2B) Node features: one-hot Z (100) ⊕ one-hot hybrid (5) ⊕ charge (1) ⊕ numHs (1) ⊕ aromatic (1)
    node_feats = []
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        one_hot_z = [0] * 100
        if 1 <= z <= 100:
            one_hot_z[z - 1] = 1
        else:
            one_hot_z[0] = 1

        hyb = atom.GetHybridization()
        hyb_map = {
            Chem.rdchem.HybridizationType.SP: 0,
            Chem.rdchem.HybridizationType.SP2: 1,
            Chem.rdchem.HybridizationType.SP3: 2,
            Chem.rdchem.HybridizationType.SP3D: 3,
            Chem.rdchem.HybridizationType.SP3D2: 4
        }
        one_hot_hyb = [0] * 5
        one_hot_hyb[hyb_map.get(hyb, 0)] = 1

        charge = atom.GetFormalCharge()
        num_hs = atom.GetTotalNumHs()
        aro = 1 if atom.GetIsAromatic() else 0

        node_feats.append(one_hot_z + one_hot_hyb + [charge, num_hs, aro])

    x = torch.tensor(node_feats, dtype=torch.float32)  # (num_atoms, 108)

    # 2C) Edge features: bond type one-hot
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        btype = bond.GetBondType()
        if btype == Chem.rdchem.BondType.SINGLE:
            bo = [1, 0, 0, 0]
        elif btype == Chem.rdchem.BondType.DOUBLE:
            bo = [0, 1, 0, 0]
        elif btype == Chem.rdchem.BondType.TRIPLE:
            bo = [0, 0, 1, 0]
        elif btype == Chem.rdchem.BondType.AROMATIC:
            bo = [0, 0, 0, 1]
        else:
            bo = [0, 0, 0, 0]

        edge_index.append((i, j))
        edge_attr.append(bo)
        edge_index.append((j, i))
        edge_attr.append(bo)  # undirected

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()  # (2, num_edges)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)    # (num_edges, 4)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 4), dtype=torch.float32)

    mask = torch.ones(x.shape[0], dtype=torch.bool)
    return {"x": x, "pos": coords, "mask": mask, "edge_index": edge_index, "edge_attr": edge_attr}


# -----------------------------------------------------------------------------
# 3) Dataset: SMILES → per-molecule graph, targets, descriptors, fingerprint
# -----------------------------------------------------------------------------
class OdourDatasetEnhanced(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: DictConfig):
        """
        df must contain:
          - cfg.data.smiles_column (SMILES)
          - each column in cfg.data.label_columns (numeric or convertible to numeric)
        """
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.smiles_col = cfg.data.smiles_column
        self.label_cols = cfg.data.label_columns
        self.radius3d = cfg.data.radius
        self.use_fp = cfg.model.use_fingerprint
        self.fp_dim = cfg.model.fp_dim

        # Pre-coerce labels to float32, compute pos_weights = (neg_count/pos_count)
        labels = (
            self.df[self.label_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .astype("float32")
        )
        label_counts = (labels > 0).sum(axis=0).values  # (# positives per label)
        neg_counts = len(self.df) - label_counts
        pos_weights = neg_counts / np.clip(label_counts, a_min=1, a_max=None)
        self.pos_weights = torch.tensor(pos_weights, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        smiles = row[self.smiles_col]

        fea = featurize_smiles_enhanced(smiles, self.radius3d)
        x = fea["x"]
        pos = fea["pos"]
        mask = fea["mask"]
        edge_index = fea["edge_index"]
        edge_attr = fea["edge_attr"]

        # Multi-label targets
        y = torch.tensor(row[self.label_cols].values.astype("float32"), dtype=torch.float32)

        # RDKit descriptors: logP, TPSA, MolWt, NumRotatableBonds
        mol = Chem.MolFromSmiles(smiles)
        descs = torch.tensor([
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.MolWt(mol),
            Descriptors.NumRotatableBonds(mol)
        ], dtype=torch.float32)  # (4,)

        # Morgan fingerprint if requested
        if self.use_fp:
            mol_nohs = Chem.MolFromSmiles(smiles)
            rv = AllChem.GetMorganFingerprintAsBitVect(mol_nohs, radius=2, nBits=self.fp_dim)
            fp = torch.tensor([int(c) for c in rv.ToBitString()], dtype=torch.float32)
        else:
            fp = None

        return {
            "x": x,                   # (num_atoms, 108)
            "pos": pos,               # (num_atoms, 3) or None
            "mask": mask,             # (num_atoms,)
            "edge_index": edge_index, # (2, num_edges)
            "edge_attr": edge_attr,   # (num_edges, 4)
            "y": y,                   # (num_labels,)
            "descs": descs,           # (4,)
            "fp": fp                  # (fp_dim,) or None
        }


# -----------------------------------------------------------------------------
# 4) collate_fn to pad variable-sized graphs for EGNN
# -----------------------------------------------------------------------------
def collate_egnn(batch: list):
    """
    Input: list of dicts, each with:
      - x:         (n_i, 108)
      - pos:       (n_i, 3) or None
      - mask:      (n_i,)
      - edge_index:(2, e_i)   [not used by EGNN directly]
      - edge_attr: (e_i, 4)   [not used by EGNN directly]
      - y:         (num_labels,)
      - descs:     (4,)
      - fp:        (fp_dim,) or None

    Output dict:
      - x:    (B, N_max, 108)
      - pos:  (B, N_max, 3)
      - mask: (B, N_max)
      - y:    (B, num_labels)
      - descs:(B, 4)
      - fp:   (B, fp_dim) or None
    """
    B = len(batch)
    num_labels = batch[0]["y"].shape[0]
    use_fp = batch[0]["fp"] is not None
    fp_dim = batch[0]["fp"].shape[0] if use_fp else 0

    # find max number of atoms
    N_max = max(item["x"].shape[0] for item in batch)

    # allocate tensors
    x_batch = torch.zeros((B, N_max, 108), dtype=torch.float32)
    pos_batch = torch.zeros((B, N_max, 3), dtype=torch.float32)
    mask_batch = torch.zeros((B, N_max), dtype=torch.bool)
    y_batch = torch.zeros((B, num_labels), dtype=torch.float32)
    descs_batch = torch.zeros((B, 4), dtype=torch.float32)
    if use_fp:
        fp_batch = torch.zeros((B, fp_dim), dtype=torch.float32)
    else:
        fp_batch = None

    for i, item in enumerate(batch):
        n_i = item["x"].shape[0]
        x_batch[i, :n_i, :] = item["x"]
        mask_batch[i, :n_i] = item["mask"]
        if item["pos"] is not None:
            pos_batch[i, :n_i, :] = item["pos"]
        y_batch[i] = item["y"]
        descs_batch[i] = item["descs"]
        if use_fp:
            fp_batch[i] = item["fp"]

    return {
        "x": x_batch,       # (B, N_max, 108)
        "pos": pos_batch,   # (B, N_max, 3)
        "mask": mask_batch, # (B, N_max)
        "y": y_batch,       # (B, num_labels)
        "descs": descs_batch,#(B, 4)
        "fp": fp_batch      # (B, fp_dim) or None
    }


# -----------------------------------------------------------------------------
# 5) EGNN + Fusion MLP
# -----------------------------------------------------------------------------
class OdourPredictorEGNN(nn.Module):
    def __init__(self, cfg: DictConfig, num_labels: int):
        super().__init__()
        hidden_dim = cfg.model.hidden_dim
        fp_dim = cfg.model.fp_dim if cfg.model.use_fingerprint else 0
        fp_hidden = cfg.model.fp_hidden if cfg.model.use_fingerprint else 0
        descs_hidden = cfg.model.descs_hidden if cfg.model.use_descs else 0

        # 5.1) Node-input MLP: 108 → hidden_dim
        self.lin_in = nn.Linear(108, hidden_dim)

        # 5.2) EGNN layer (one layer here; you can stack by manually repeating or adjust `EGNN` internals)
        self.egnn = EGNN(
            dim=hidden_dim,
            edge_dim=0,  # we’re not using explicit edge_attr; EGNN uses distances
            m_dim=hidden_dim,
            num_nearest_neighbors=cfg.model.num_nearest_neighbors,
            dropout=cfg.model.egnn_dropout,
            norm_feats=cfg.model.norm_feats,
            norm_coors=cfg.model.norm_coors,
            update_feats=True,
            update_coors=True,
            only_sparse_neighbors=False,
            valid_radius=cfg.model.valid_radius,
            m_pool_method=cfg.model.m_pool_method,
            fourier_features=0,
            soft_edges=False,
            coor_weights_clamp_value=None
        )

        # 5.3) Fingerprint encoder (if used)
        if cfg.model.use_fingerprint:
            self.fp_encoder = nn.Sequential(
                nn.Linear(fp_dim, fp_hidden),
                nn.ReLU(),
                nn.Dropout(cfg.model.dropout)
            )
            fusion_dim = hidden_dim + fp_hidden
        else:
            self.fp_encoder = None
            fusion_dim = hidden_dim

        # 5.4) Descriptor encoder (if used)
        if cfg.model.use_descs:
            self.descs_encoder = nn.Sequential(
                nn.Linear(4, descs_hidden),
                nn.ReLU(),
                nn.Dropout(cfg.model.dropout)
            )
            fusion_dim += descs_hidden
        else:
            self.descs_encoder = None

        # 5.5) Final MLP head
        layers = []
        in_dim = fusion_dim
        for h in cfg.model.mlp_hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(cfg.model.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, num_labels))
        self.head = nn.Sequential(*layers)

    def forward(self, x, pos, mask, fp=None, descs=None):
        """
        - x:      (B, N_max, 108)
        - pos:    (B, N_max,  3)
        - mask:   (B, N_max)
        - fp:     (B, fp_dim) or None
        - descs:  (B, 4) or None
        """
        # 1) Node MLP
        h = self.lin_in(x)  # (B, N_max, hidden_dim)

        # 2) EGNN message passing
        fea_out, coor_out = self.egnn(h, pos, mask=mask)  # (B, N_max, hidden_dim), (B, N_max, 3)

        # 3) Masked sum-pool → mean
        mask_f = mask.unsqueeze(-1).float()          # (B, N_max, 1)
        fea_masked = fea_out * mask_f                 # (B, N_max, hidden_dim)
        graph_sum = fea_masked.sum(dim=1)             # (B, hidden_dim)
        num_nodes = mask_f.sum(dim=1)                 # (B, 1)
        graph_mean = graph_sum / torch.clamp(num_nodes, min=1.0)  # (B, hidden_dim)

        h_graph = graph_mean  # (B, hidden_dim)

        # 4) Fingerprint fusion
        if fp is not None and self.fp_encoder is not None:
            fp_emb = self.fp_encoder(fp)             # (B, fp_hidden)
            h_graph = torch.cat([h_graph, fp_emb], dim=1)

        # 5) Descriptor fusion
        if descs is not None and self.descs_encoder is not None:
            descs_emb = self.descs_encoder(descs)     # (B, descs_hidden)
            h_graph = torch.cat([h_graph, descs_emb], dim=1)

        # 6) Final MLP
        out = self.head(h_graph)  # (B, num_labels)
        return out


# -----------------------------------------------------------------------------
# 6) Training Loop
# -----------------------------------------------------------------------------
def train_loop(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.train.use_cuda else "cpu")
    df = pd.read_csv(cfg.data.input_csv)

    # Build X = [SMILES + labels], then stratify
    X_full = df[[cfg.data.smiles_column] + cfg.data.label_columns].copy()
    X_np = X_full[cfg.data.smiles_column].values.reshape(-1, 1)
    y_np = X_full[cfg.data.label_columns].values.astype(np.int32)

    X_train, y_train, X_val, y_val = iterative_train_test_split(
        X_np, y_np, test_size=1 - cfg.data.train_test_split_ratio
    )

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
    ds_val = OdourDatasetEnhanced(df_val, cfg)

    # pos_weights for WeightedBCE
    pos_weights = ds_train.pos_weights.to(device)

    train_loader = DataLoader(
        ds_train,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=collate_egnn
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        collate_fn=collate_egnn
    )

    # Model, optimizer, scheduler, loss
    model = OdourPredictorEGNN(cfg, num_labels=len(cfg.data.label_columns)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=3,
        factor=0.5
    )
    criterion = WeightedBCE(pos_weights)

    best_val_auc = 0.0
    no_improve = 0

    for epoch in range(cfg.train.epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            x = batch["x"].to(device)
            pos = batch["pos"].to(device)
            mask = batch["mask"].to(device)
            y = batch["y"].to(device)
            descs = batch["descs"].to(device)
            fp = batch["fp"].to(device) if batch["fp"] is not None else None

            optimizer.zero_grad()
            logits = model(x, pos, mask, fp=fp, descs=descs)  # (B, num_labels)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.shape[0]

        # Validation
        model.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                pos = batch["pos"].to(device)
                mask = batch["mask"].to(device)
                y = batch["y"].to(device)
                descs = batch["descs"].to(device)
                fp = batch["fp"].to(device) if batch["fp"] is not None else None

                logits = model(x, pos, mask, fp=fp, descs=descs)
                all_logits.append(logits.cpu())
                all_labels.append(y.cpu())

        all_logits = torch.cat(all_logits, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        from sklearn.metrics import roc_auc_score, average_precision_score

        aucs = []
        aps = []
        for j in range(all_labels.shape[1]):
            if len(np.unique(all_labels[:, j])) > 1:
                aucs.append(roc_auc_score(all_labels[:, j], all_logits[:, j]))
                aps.append(average_precision_score(all_labels[:, j], all_logits[:, j]))

        mean_auc = float(np.mean(aucs)) if aucs else 0.0
        mean_ap = float(np.mean(aps)) if aps else 0.0

        scheduler.step(1.0 - mean_auc)

        if mean_auc > best_val_auc:
            best_val_auc = mean_auc
            no_improve = 0
            if cfg.output.save_model:
                os.makedirs(os.path.dirname(cfg.output.model_path), exist_ok=True)
                torch.save(model.state_dict(), cfg.output.model_path)
        else:
            no_improve += 1

        print(
            f"Epoch {epoch+1}/{cfg.train.epochs}  "
            f"Train Loss: {total_loss/len(ds_train):.4f}  "
            f"Val AUC: {mean_auc:.4f}  Val AP: {mean_ap:.4f}"
        )

        if no_improve >= cfg.train.early_stopping_patience:
            print("Early stopping triggered.")
            break

    print(f"Best validation AUC: {best_val_auc:.4f}")


# -----------------------------------------------------------------------------
# 7) Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    @hydra.main(config_path="conf", config_name="config4", version_base=None)
    def main(cfg: DictConfig):
        train_loop(cfg)
    main()
