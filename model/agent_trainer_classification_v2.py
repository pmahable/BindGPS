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
from src.models import GCN, GATModel

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
    # base_path: str = "/home/jchc/Documents/larschan_laboratory/BindGPS/data/datasets"
    base_path: str = "/gpfs/data/larschan/shared_data/BindGPS/data/datasets/"

    # Data
    p_value: str = "0_1"
    resolution: str = "1kb"
    exclude_features: Tuple[str, ...] = ("clamp", "gaf", "psq")
    features_of_interest: Tuple[str, ...] = (
        "clamp","gaf","psq","h3k27ac","h3k27me3","h3k36me3",
        "h3k4me1","h3k4me2","h3k4me3","h3k9me3","h4k16ac"
    )
    target_column: str = "mre_labels"
    train_size: float = 0.7
    non_mre_size: float = 0.3
    seed: int = 42

    # Model
    model_type: str = "gcn"  # "gcn" or "gat"
    hidden_gnn_size: int = 128
    num_gnn_layers: int = 3
    hidden_linear_size: int = 128
    num_linear_layers: int = 3
    dropout: float = 0.5
    normalize: bool = True
    
    # GAT-specific parameters
    gat_heads: int = 4
    gat_negative_slope: float = 0.2
    gat_concat: bool = True
    gat_edge_dim: int = 2  # contactCount + loop_size_transformed

    # Optimization
    lr: float = 5e-4
    weight_decay: float = 5e-4
    epochs: int = 25

    # NeighborLoader
    num_neighbors: Tuple[int, int, int] = (20, 20, 20)
    batch_size: int = 256
    num_workers: int = 1
    
    # Batch validation (processes all edges)
    val_batch_size: Optional[int] = 1024  # If None, uses full graph evaluation

    # Device / perf
    use_cuda_if_available: bool = True

    # Logging
    use_wandb: bool = False
    wandb_project: str = "basic-gnn"
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
        mre_mask = self.node_df["mre_mask"].astype(bool)

        total_mre_samples = mre_mask.sum()
        self.node_df["non_mre_mask"] = self.node_df[self.cfg.target_column].apply(lambda x: True if x == 0 else False)
        non_mre_mask = self.node_df["non_mre_mask"].astype(bool)


        # Randomly select non-MRE samples based on defined size. 
        non_mre_samples = total_mre_samples * self.cfg.non_mre_size
        indices = np.arange(len(non_mre_mask))
        non_mre_indices = indices[non_mre_mask]
        selected_non_mres = np.random.choice(non_mre_indices, size=int(non_mre_samples), replace=False)
        non_mre_mask = np.zeros(len(non_mre_mask), dtype=bool)
        non_mre_mask[selected_non_mres] = True
        self.node_df["non_mre_mask"] = non_mre_mask

        # Combine MRE and non-MRE Samples
        mask = mre_mask | non_mre_mask

        # Stratified split on masked subset
        X_train, X_evaluation, y_train, y_evaluation = train_test_split(
            input_features[mask],
            target[mask],
            train_size=self.cfg.train_size,
            stratify=target[mask],
            random_state=self.cfg.seed,
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_evaluation,
            y_evaluation,
            train_size=0.3,
            stratify=y_evaluation,
            random_state=self.cfg.seed,
        )

        # Build boolean masks over the FULL index
        train_mask = pd.Series(False, index=input_features.index)
        val_mask = pd.Series(False, index=input_features.index)
        test_mask = pd.Series(False, index=input_features.index)
        train_mask.loc[X_train.index] = True
        val_mask.loc[X_val.index] = True
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
        
        # Edge weights for NeighborSampler (use p-value transformed)
        edge_weight = tensors["edge_pvalue_transformed"]  # [E]
        
        # Edge features (contact count + loop size transformed)
        edge_attr = torch.stack([
            tensors["edge_contactCount"],
            tensors["edge_loop_size_transformed"]
        ], dim=1)  # [E, 2] - 2 edge features

        pyg_data = Data(
            x=X,
            y=y,
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_attr=edge_attr,
            train_mask=torch.tensor(train_mask.to_numpy(), dtype=torch.bool),
            val_mask=torch.tensor(val_mask.to_numpy(), dtype=torch.bool),
            test_mask=torch.tensor(test_mask.to_numpy(), dtype=torch.bool),
        )
        self.data = pyg_data

    def build_loaders(self) -> Tuple[NeighborLoader, NeighborLoader, NeighborLoader]:
        assert self.data is not None, "Call build_dataset() first."
        
        # Training loader with neighbor sampling
        train_loader = NeighborLoader(
            self.data,
            input_nodes=self.data.train_mask,
            num_neighbors=list(self.cfg.num_neighbors),
            batch_size=self.cfg.batch_size,
            weight_attr="edge_weight",
            num_workers=self.cfg.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        
        # Validation loader with ALL neighbors (-1 means no sampling limit)
        val_loader = NeighborLoader(
            self.data,
            input_nodes=self.data.val_mask,
            num_neighbors=[-1] * len(self.cfg.num_neighbors),  # Sample ALL neighbors
            batch_size=self.cfg.val_batch_size if self.cfg.val_batch_size else self.cfg.batch_size,
            weight_attr="edge_weight",
            num_workers=self.cfg.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        
        # Test loader with ALL neighbors
        test_loader = NeighborLoader(
            self.data,
            input_nodes=self.data.test_mask,
            num_neighbors=[-1] * len(self.cfg.num_neighbors),  # Sample ALL neighbors
            batch_size=self.cfg.val_batch_size if self.cfg.val_batch_size else self.cfg.batch_size,
            weight_attr="edge_weight",
            num_workers=self.cfg.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        
        return train_loader, val_loader, test_loader

    # ----- Model / Optim -----
    def build_model(self) -> None:
        assert self.data is not None, "Call build_dataset() first."

        # Safer class count (works even if labels aren't 0..C-1)
        train_labels = self.data.y[self.data.train_mask]
        unique_labels = torch.unique(train_labels)
        num_classes = int(unique_labels.numel())
        
        print(f"Unique train labels: {unique_labels}")
        print(f"Training with num_classes: {num_classes}")
        print(f"Train Samples: {self.data.train_mask.sum()}")
        print(f"Val Samples: {self.data.val_mask.sum()}")
        print(f"Test Samples: {self.data.test_mask.sum()}")
        
        # Check if labels need remapping to 0..C-1 range, will happen when we exclude non-mres
        if unique_labels.min() != 0 or unique_labels.max() != (num_classes - 1):
            print(f"Warning: Labels not in 0..{num_classes-1} range, remapping...")
            # Create mapping from original labels to 0..C-1
            label_mapping = {int(old_label): new_label for new_label, old_label in enumerate(unique_labels)}
            print(f"Label mapping: {label_mapping}")
            
            # Remap all labels in the dataset
            for old_label, new_label in label_mapping.items():
                self.data.y[self.data.y == old_label] = new_label


        if self.cfg.model_type.lower() == "gcn":
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
        elif self.cfg.model_type.lower() == "gat":
            self.model = GATModel(
                in_channels=self.data.x.size(1),
                out_channels=num_classes,
                hidden_gnn_size=self.cfg.hidden_gnn_size,
                num_gnn_layers=self.cfg.num_gnn_layers,
                hidden_linear_size=self.cfg.hidden_linear_size,
                num_linear_layers=self.cfg.num_linear_layers,
                heads=self.cfg.gat_heads,
                concat=self.cfg.gat_concat,
                negative_slope=self.cfg.gat_negative_slope,
                dropout=self.cfg.dropout,
                edge_dim=self.cfg.gat_edge_dim,
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported model_type: {self.cfg.model_type}. Use 'gcn' or 'gat'.")

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

    # ----- Train / Eval -----
    def _train_one_epoch(self, train_loader: NeighborLoader, val_loader: NeighborLoader, epoch: int) -> Tuple[float, float, float, float]:
        assert self.model is not None and self.optimizer is not None

        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            # Handle different model types
            if self.cfg.model_type.lower() == "gat":
                # GAT models can use edge attributes if available
                edge_attr = getattr(batch, 'edge_attr', None)
                out = self.model(batch.x, batch.edge_index, edge_attr=edge_attr)  # [N_batch, C]
            else:
                out = self.model(batch.x, batch.edge_index)  # [N_batch, C]
            mask = batch.train_mask.bool()
            targets = batch.y.to(torch.long)

            loss = self.criterion(out[mask], targets[mask])
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.detach().item())
            preds = out[mask].detach().argmax(dim=1)
            correct += int((preds == targets[mask]).sum().item())
            total += int(mask.sum().item())
            
            # Clear intermediate variables to free GPU memory
            del out, loss, preds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        epoch_loss = total_loss / max(len(train_loader), 1)
        epoch_acc = correct / max(total, 1)

        self._log({"train/loss": epoch_loss, "train/acc": epoch_acc, "epoch": epoch}, step=epoch)

        # evaluate on validation dataset using validation loader
        val_loss, val_acc = self.evaluate(split="val", loader=val_loader)
        # self._log({"val/loss": val_loss, "val/acc": val_acc, "epoch": epoch}, step=epoch)
        
        return epoch_loss, epoch_acc, val_loss, val_acc

    @torch.no_grad()
    def evaluate(self, split="test", loader=None) -> Tuple[float, float]:
        """Evaluate model using NeighborLoader with all neighbors.
        
        Args:
            split: "test" or "val" to specify which nodes to evaluate
            loader: NeighborLoader to use for evaluation (required)
            
        Returns:
            Tuple of (loss, accuracy)
        """
        assert self.model is not None and self.data is not None
        assert loader is not None, "NeighborLoader is required for evaluation"
        
        self.model.eval()
        return self._evaluate_with_loader(loader, split)
    
    
    def _evaluate_with_loader(self, loader: NeighborLoader, split: str) -> Tuple[float, float]:
        """Evaluate using NeighborLoader with all neighbors (-1 sampling)."""
        all_preds = []
        all_targets = []
        all_losses = []
        
        for batch in loader:
            batch = batch.to(self.device)
            
            if self.cfg.model_type.lower() == "gat":
                edge_attr = getattr(batch, 'edge_attr', None)
                out = self.model(batch.x, batch.edge_index, edge_attr=edge_attr)
            else:
                out = self.model(batch.x, batch.edge_index)
            
            if split == "val":
                mask = batch.val_mask.bool()
            else:
                mask = batch.test_mask.bool()
            
            if mask.sum() == 0:
                continue
            
            batch_logits = out[mask]
            batch_targets = batch.y[mask].to(torch.long)
            
            batch_loss = self.criterion(batch_logits, batch_targets)
            batch_preds = batch_logits.argmax(dim=1).detach().cpu()
            batch_targets_cpu = batch_targets.detach().cpu()
            
            all_preds.append(batch_preds)
            all_targets.append(batch_targets_cpu)
            all_losses.append(batch_loss.item())
            
            # Memory cleanup
            del out, batch_logits, batch_targets
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if not all_preds:
            print(f"Warning: No {split} predictions generated")
            return 0.0, 0.0
        
        final_preds = torch.cat(all_preds, dim=0)
        final_targets = torch.cat(all_targets, dim=0)
        avg_loss = sum(all_losses) / len(all_losses)
        
        acc = accuracy_score(final_targets.numpy(), final_preds.numpy())
        self._log({f"{split}/acc": acc, f"{split}/loss": avg_loss})
        
        return float(avg_loss), float(acc)

    def fit(self) -> None:
        self.build_dataset()
        train_loader, val_loader, test_loader = self.build_loaders()
        self.build_model()

        print(self.device)
        for epoch in range(self.cfg.epochs):
            loss, acc, val_loss, val_acc = self._train_one_epoch(train_loader, val_loader, epoch)
            print(f"Epoch {epoch:03d} | loss={loss:.4f} | train_acc={acc:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        test_loss, test_acc = self.evaluate(split="test", loader=test_loader)
        print(f"Final Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

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
        use_wandb=True, 
        seed=42,  # Using default seed for reproducibility
        val_batch_size=1024  # Enable batched validation
    )
    trainer = GNNTrainer(cfg)
    trainer.run()

if __name__ == "__main__":
    main()