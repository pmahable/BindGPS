import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, Optional, Tuple
import json

from src.wandb_utils import WandbLogger
from src.metrics import compute_metrics
from src.models import KmerSequenceModel

from sklearn.metrics import roc_auc_score, average_precision_score
from torcheval.metrics import MulticlassAUROC, MulticlassAUPRC, MulticlassAccuracy

class SeqSVMTrainer:
    def __init__(self,
                 dataset: DNASequenceDataset,
                 resolution: str,
                 svd_dim: int,
                 input_dim: int = 128, 
                 hidden_dims: Tuple[int, ...] = (256, 128),
                 num_classes: int = 3, 
                 dropout: float = 0.3,
                 batch_size: int = 128, 
                 lr: float = 1e-3, 
                 weighted: bool = False,
                 task: str = "mre",
                 target_metric: str = "MulticlassAccuracy",
                 wandb_logger: Optional[WandbLogger] = None,
                 device: Optional[torch.device] = None):
               
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wandb_logger = wandb_logger
        self.batch_size = batch_size
        self.weighted = weighted
        self.task = task
        self.target_metric = target_metric
        self.resolution = resolution
        self.svd_dim = svd_dim
        
        
        self.run_name = f"{task}_res{resolution}_svd{svd_dim}_lr{lr}_bs{batch_size}_weighted{weighted}"
               
        #Making training and validation data loaders
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

        #Instantiating the model
        self.model = KmerSequenceModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout=dropout,
        ).to(self.device)    

        if weighted:
            # Compute class weights
            if task == "mre":
                _, class_counts = np.unique(dataset.mre_labels, return_counts=True)
            else:
                _, class_counts = np.unique(dataset.gene_labels, return_counts=True)
            weights = 1 / (class_counts / class_counts.sum())
            weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
               

    def train(self, val_metrics=None, epochs: int=10):
        if val_metrics is None:
            val_metrics = [
            MulticlassAUROC(num_classes=3),
            MulticlassAUPRC(num_classes=3),
            MulticlassAccuracy(),
        ]
        best_state_dict = None
        
        device = self.device
        patience = 0
        best_metric = 0
        train_acc = MulticlassAccuracy().to(device)
        best_metrics = {}
        
        for epoch in range(epochs):
            if patience >= 5:
                print(f"Early Stopping at epoch {epoch}")
                break
            self.model.train()
            train_loss = 0.0
            
            for gene, mre, vecs in self.train_loader:
                if self.task == "mre":
                    labels = mre
                elif self.task == "gene":
                    labels = gene
                labels = labels.long().to(self.device)
                vecs = vecs.to(self.device)
                
                self.optimizer.zero_grad()
                outs = self.model(vecs)
                loss = self.criterion(outs, labels)
                loss.backward()
                self.optimizer.step()
                
                logits = F.softmax(outs, dim=1)
                train_loss += loss.detach().item()
                train_acc.update(logits, labels)
                
            train_accuracy = train_acc.compute().item()
            print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f}")
            train_acc.reset()
                
            val_loss, metrics = self.evaluate(val_metrics=val_metrics)
            print(f"[Epoch {epoch}] Val Loss: {val_loss:.4f}, Val Metrics: {metrics}")
            
            
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                **{f"val_{k}": v for k, v in metrics.items()}
            })
                
            target_metric = metrics.get(self.target_metric, 0.0)
            if target_metric > best_metric:
                patience = 0
                best_metric = target_metric
                best_metrics = metrics
                #torch.save(self.model.state_dict(), f"{self.run_name}_best.pt")
                best_state_dict = self.model.state_dict()
            else:
                patience +=1
                
        #best_model = torch.load(f"{self.run_name}_best.pt", map_location=device)
        #self.model.load_state_dict(best_model)
        
        if best_state_dict is not None:
            torch.save(best_state_dict, f"{self.run_name}_best.pt")
            self.model.load_state_dict(best_state_dict)

        
        
        self.model.to(device)

        return self.model, best_metrics
               

    def evaluate(self, val_metrics=None, return_full: bool=False):
        device = self.device
        self.model.eval()
        
        for m in val_metrics:
            m.to(device)
            m.reset()
        val_loss = 0
        lab_pred_log = ([], [], [])
        
        with torch.no_grad():
            for gene, mre, vecs in self.val_loader:
                labels = mre if self.task == "mre" else gene
                labels = labels.long().to(device)
                vecs = vecs.to(device)
                
                outs = self.model(vecs)
                loss = self.criterion(outs, labels)
                val_loss += loss.detach().item()
                logits = F.softmax(outs, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                if return_full:
                    for append, ls in zip([labels, preds, logits], lab_pred_log):
                        ls.append(append.detach().cpu())
                    
                for metric in val_metrics:
                    metric.update(logits, labels)
        
        lab_pred_log = [torch.cat(ls) for ls in lab_pred_log] if return_full else None
        metrics = {m.__class__.__name__: m.compute().item() for m in val_metrics}

        return val_loss, metrics if not return_full else (val_loss, metrics, lab_pred_log)
      

    def input_gradients(self):
        device = self.device
        self.model.eval()
        all_loss_grads = []
        all_class_grads = []
        
        for gene, mre, vecs in self.val_loader:
            labels = mre if self.task == "mre" else gene
            labels = labels.long().to(device)

            vecs = vecs.to(device)
            vecs.requires_grad_(True)

            outputs = self.model(vecs)

            # per-class gradients: d(outputs[:,c].sum())/d(vecs)
            class_grads = []
            C = outputs.shape[1]
            for c in range(C):
                grad_c = torch.autograd.grad(
                    outputs[:, c].sum(), vecs, retain_graph=True
                )[0]
                class_grads.append(grad_c.detach().cpu())
            class_grads = torch.stack(class_grads, dim=1)  # (B, C, D)

            loss = self.criterion(outputs, labels)
            loss_grads = torch.autograd.grad(loss, vecs)[0].detach().cpu()

            all_loss_grads.append(loss_grads)
            all_class_grads.append(class_grads)

        loss_grads = torch.cat(all_loss_grads, dim=0)   # (N, D)
        class_grads = torch.cat(all_class_grads, dim=0) # (N, C, D)
        return loss_grads, class_grads
        