import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torch_geometric.nn as geom_nn
# from transformers import AutoModel, AutoTokenizer


class KmerSequenceModel(nn.Module):
    """
    Sequence model that takes input DNA sequences, represents
    sequences as k-mer counts and then uses a neural network to 
    predict functional associations or other regulatory properties.

    Input: DNA Sequence
    Output: Classification tasks
    """
    
    def __init__(self, input_dim, hidden_dims=(256, 128), dropout=0.3, num_classes=1):
        super().__init__()
        layers = []
        dims = [input_dim, *hidden_dims]
        for in_dim, out_dim in zip(dims, dims[1:]):
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
        # final classification/regression head
        layers += [nn.Linear(dims[-1], num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)