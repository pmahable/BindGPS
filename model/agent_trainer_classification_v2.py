# train_gnn.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

# Project modules
from src.data import DatasetLoader, GraphParamBuilder
from src.models import GCN

# PyG
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

# Optional: Weights & Biases
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False


# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    # Paths
    base_path: str = "/home/jchc/Documents/larschan_laboratory/BindGPS/data/datasets"

    # Data
    p_value: str = "0_1"
    resolution: str = "1kb"
    exclude_features: Tuple[str, ...] = ("clamp", "gaf", "psq")
    features_of_interest: Tuple[str, ...] = (
        "clamp","gaf","psq","h3k27ac","h3k27me3","h3k36me3",
        "h3k4me1","h3k4me2","h3k4me3","h3k9me3","h4k16ac"
    )
    target_column: str = "mre_labels"
    train_size: float = 0.8
    seed: int = 42

    # Model
    hidden_gnn_size: int = 128
    num_gnn_layers: int = 3
    hidden_linear_size: int = 128
    num_linear_layers: int = 3
    dropout: float = 0.5
    normalize: bool = True

    # Optimization
    lr: float = 5e-4
    weight_decay: float = 5e-4
    epochs: int = 100

    # NeighborLoader
    num_neighbors: Tuple[int, int, int] = (20, 20, 20)
    batch_size: int = 256
    num_workers: int = 8

    # Device / perf
    use_cuda_if_available: bool = True

    # Logging
    use_wandb: bool = False
    wandb_project: str = "gnn-mre"
    wandb_entity: Optional[str] = None  # or your entity string

# ----------------------------
# Utilities
# ----------------------------
def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For more determinism (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(use_cuda_if_available: bool = True) -> torch.device:
    if use_cuda_if_available and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# ----------------------------
# Trainer
# ----------------------------
class GNNTrainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        set_global_seed(cfg.seed)
        self.device = get_device(cfg.use_cuda_if_available)

        # Will be populated later
        self.node_df: Optional[pd.DataFrame] = None
        self.edge_df: Optional[pd.DataFrame] = None
        self.data: Optional[Data] = None
        self.model: Optional[torch.nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion = torch.nn.CrossEntropyLoss()

        # W&B
        self.wandb_run = None
        if self.cfg.use_wandb and WANDB_AVAILABLE:
            self._init_wandb()

    # ----- Logging -----
    def _init_wandb(self) -> None:
        try:
            wandb.login()
            self.wandb_run = wandb.init(
                project=self.cfg.wandb_project,
                entity=self.cfg.wandb_entity,
                config=asdict(self.cfg),
            )
        except Exception as e:
            print(f"[WARN] W&B init failed: {e}")
            self.wandb_run = None

    def _log(self, metrics: dict, step: Optional[int] = None) -> None:
        if self.wandb_run is not None:
            wandb.log(metrics, step=step)

    # ----- Data -----
    def build_dataset(self) -> None:
        # Load raw dataframes
        loader = DatasetLoader(base_path=self.cfg.base_path)
        node_df, edge_df = loader.load(p_value=self.cfg.p_value, resolution=self.cfg.resolution)

        self.node_df = node_df.copy()
        self.edge_df = edge_df.copy()

        # Feature selection
        feats = [f for f in self.cfg.features_of_interest if f not in self.cfg.exclude_features]
        input_features = self.node_df.loc[:, feats].copy()

        # Target and mask (mre > 0 considered labeled)
        target = self.node_df[self.cfg.target_column].copy()
        self.node_df["mre_mask"] = self.node_df[self.cfg.target_column].apply(lambda x: True if x > 0 else False)
        mask = self.node_df["mre_mask"].astype(bool)

        # Stratified split on masked subset
        X_train, X_test, y_train, y_test = train_test_split(
            input_features[mask],
            target[mask],
            train_size=self.cfg.train_size,
            stratify=target[mask],
            random_state=self.cfg.seed,
        )

        # Build boolean masks over the FULL index
        train_mask = pd.Series(False, index=input_features.index)
        test_mask = pd.Series(False, index=input_features.index)
        train_mask.loc[X_train.index] = True
        test_mask.loc[X_test.index] = True

        # Build tensors via your helper
        builder = GraphParamBuilder(
            node_df=self.node_df,
            edge_df=self.edge_df,
            target=target,
            mask=mask,
            input_features=input_features,
            seed=self.cfg.seed,
        )
        tensors = builder.convert_to_tensors()

        # Ensure dtypes
        X = tensors["X"]                          # [N, F] float
        y = tensors["y"].to(torch.long)           # [N] long for CE loss
        edge_index = tensors["edge_index"]        # [2, E]
        edge_weight = tensors["edge_pvalue_transformed"]      # [E] (optional)

        pyg_data = Data(
            x=X,
            y=y,
            edge_index=edge_index,
            edge_weight=edge_weight,
            train_mask=torch.tensor(train_mask.to_numpy(), dtype=torch.bool),
            test_mask=torch.tensor(test_mask.to_numpy(), dtype=torch.bool),
        )
        self.data = pyg_data

    def build_loaders(self) -> NeighborLoader:
        assert self.data is not None, "Call build_dataset() first."
        loader = NeighborLoader(
            self.data,
            input_nodes=self.data.train_mask,
            num_neighbors=list(self.cfg.num_neighbors),
            batch_size=self.cfg.batch_size,
            weight_attr="edge_weight",
            num_workers=self.cfg.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        return loader

    # ----- Model / Optim -----
    def build_model(self) -> None:
        assert self.data is not None, "Call build_dataset() first."

        num_classes = len(self.data.y.unique()) # total classes, instead of those solely unmasked

        self.model = GCN(
            in_channels=self.data.x.size(1),
            out_channels=num_classes,
            hidden_gnn_size=self.cfg.hidden_gnn_size,
            num_gnn_layers=self.cfg.num_gnn_layers,
            hidden_linear_size=self.cfg.hidden_linear_size,
            num_linear_layers=self.cfg.num_linear_layers,
            dropout=self.cfg.dropout,
            normalize=self.cfg.normalize,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

    # ----- Train / Eval -----
    def _train_one_epoch(self, train_loader: NeighborLoader, epoch: int) -> Tuple[float, float]:
        assert self.model is not None and self.optimizer is not None

        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            out = self.model(batch.x, batch.edge_index)  # [N_batch, C]
            mask = batch.train_mask.bool()
            targets = batch.y.to(torch.long)

            loss = self.criterion(out[mask], targets[mask])
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            preds = out[mask].argmax(dim=1)
            correct += int((preds == targets[mask]).sum().item())
            total += int(mask.sum().item())

        epoch_loss = total_loss / max(len(train_loader), 1)
        epoch_acc = correct / max(total, 1)

        self._log({"train/loss": epoch_loss, "train/acc": epoch_acc, "epoch": epoch}, step=epoch)
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def evaluate(self) -> float:
        assert self.model is not None and self.data is not None

        self.model.eval()
        d = self.data.to(self.device)

        logits = self.model(d.x, d.edge_index)
        mask = d.test_mask.bool()
        preds = logits[mask].argmax(dim=1).detach().cpu()
        targets = d.y[mask].to(torch.long).detach().cpu()

        acc = accuracy_score(targets.numpy(), preds.numpy())
        self._log({"test/acc": acc})
        return float(acc)

    def fit(self) -> None:
        self.build_dataset()
        train_loader = self.build_loaders()
        self.build_model()

        print(self.device)
        for epoch in range(self.cfg.epochs):
            loss, acc = self._train_one_epoch(train_loader, epoch)
            print(f"Epoch {epoch:03d} | loss={loss:.4f} | acc={acc:.4f}")

        test_acc = self.evaluate()
        print(f"Test Accuracy: {test_acc:.4f}")

    # Convenience single entry
    def run(self) -> None:
        self.fit()
        if self.wandb_run is not None:
            self.wandb_run.finish()

# ----------------------------
# Script entry
# ----------------------------
def main():
    cfg = Config(
        # toggle this on to log to W&B (requires `wandb login`)
        use_wandb=False,
    )
    trainer = GNNTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()