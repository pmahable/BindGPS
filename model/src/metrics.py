from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn.functional as F
import numpy as np

def compute_metrics(outputs, labels, num_classes):
    """
    Compute metrics (AUROC, AUPRC, accuracy) for the model.

    Args:
        outputs: outputs from the model (before softmax)
        labels: true labels
    """

    outputs = F.softmax(torch.Tensor(outputs), dim=-1)

    if type(outputs) == torch.Tensor:
        outputs = outputs.detach().cpu().numpy()
    if type(labels) == torch.Tensor:
        labels = labels.detach().cpu().numpy()

    predictions = np.argmax(outputs, axis=1)
    accuracy = np.mean(predictions == labels)
    if num_classes == 2:
        auroc = roc_auc_score(labels, outputs[:, 1])
        auprc = average_precision_score(labels, outputs[:, 1])
    elif num_classes > 2:
        auroc = roc_auc_score(labels, outputs, multi_class='ovr', average='macro')
        ohe_labels = np.eye(num_classes)[labels]
        auprc = average_precision_score(ohe_labels, outputs)
    else:
        raise ValueError("num_classes must be 2 or greater")
    
    return {
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": accuracy,
    }

def compute_regression_metrics(outputs, labels):
    """
    Compute metrics (MAE, MSE) for the model.

    Args:
        outputs: outputs from the model
        labels: true labels
    """

    outputs = torch.Tensor(outputs)
    labels = torch.Tensor(labels)

    mae = F.l1_loss(outputs, labels)
    mse = F.mse_loss(outputs, labels)
    
    return {
        "mae": mae,
        "mse": mse,
    }
        