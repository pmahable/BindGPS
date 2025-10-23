import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch_geometric.nn import SAGEConv, GATConv
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




class GATModel(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_gnn_size,
            num_gnn_layers,
            hidden_linear_size,
            num_linear_layers,
            out_channels,
            heads = 4,
            negative_slope = 0.2,
            dropout=0.5,
            edge_dim = None,
            concat = True,
            ):
        super().__init__()
        
        assert num_gnn_layers >= 1, "num_gnn_layers must be >= 1"
        assert num_linear_layers >= 1, "num_linear_layers counts the output layer; must be >= 1"

        # GAT convolutional layers - mirrors GCN structure
        if num_gnn_layers == 1:
            # Single layer case
            self.gat_convs = nn.ModuleList([
                GATConv(
                    in_channels, 
                    hidden_gnn_size, 
                    heads=heads, 
                    concat=False,  # Don't concat for single layer
                    negative_slope=negative_slope,
                    dropout=dropout,
                    edge_dim=edge_dim
                )
            ])
        else:
            # Multi-layer case
            convs = []
            
            # First layer
            convs.append(
                GATConv(
                    in_channels, 
                    hidden_gnn_size, 
                    heads=heads, 
                    concat=concat,
                    negative_slope=negative_slope,
                    dropout=dropout,
                    edge_dim=edge_dim
                )
            )
            
            # Middle layers
            for _ in range(num_gnn_layers - 2):
                in_dim = hidden_gnn_size * heads if concat else hidden_gnn_size
                convs.append(
                    GATConv(
                        in_dim, 
                        hidden_gnn_size, 
                        heads=heads, 
                        concat=concat,
                        negative_slope=negative_slope,
                        dropout=dropout,
                        edge_dim=edge_dim
                    )
                )
            
            # Last layer (no concat to keep output size consistent)
            in_dim = hidden_gnn_size * heads if concat else hidden_gnn_size
            convs.append(
                GATConv(
                    in_dim, 
                    hidden_gnn_size, 
                    heads=heads, 
                    concat=False,  # No concat for last layer
                    negative_slope=negative_slope,
                    dropout=dropout,
                    edge_dim=edge_dim
                )
            )
            
            self.gat_convs = nn.ModuleList(convs)

        # Linear layers - same as GCN
        lin_sizes = [hidden_gnn_size] + [hidden_linear_size] * max(0, num_linear_layers - 1) + [out_channels]
        self.linear = nn.ModuleList([
            nn.Linear(lin_sizes[i], lin_sizes[i+1]) for i in range(len(lin_sizes) - 1)
        ])

        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        # GAT convolutional layers - standard approach without ReLU between layers
        for i, conv in enumerate(self.gat_convs):
            x = conv(x, edge_index, edge_attr=edge_attr)
            # Only apply dropout between GAT layers (not after the last one)
            if i < len(self.gat_convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
            
        # Linear layers - same as GCN
        for i, lin in enumerate(self.linear):
            x = lin(x)
            if i < (len(self.linear) - 1):
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class GCN(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_gnn_size,
            num_gnn_layers,
            hidden_linear_size,
            num_linear_layers,
            out_channels,
            dropout=0.5,
            normalize=True
            ):
        super().__init__()
        
        assert num_gnn_layers >= 1, "num_gnn_layers must be >= 1"
        assert num_linear_layers >= 1, "num_linear_layers counts the output layer; must be >= 1"

        # convolutional layers
        self.convs  = nn.ModuleList(
            [SAGEConv(in_channels, hidden_gnn_size,normalize=normalize)] +
            [SAGEConv(hidden_gnn_size, hidden_gnn_size,normalize=normalize) for _ in range(max(0, num_gnn_layers - 1))]
        )

        # linear layers
        lin_sizes = [hidden_gnn_size] + [hidden_linear_size] * max(0, num_linear_layers - 1) + [out_channels]
        self.linear = nn.ModuleList([
            nn.Linear(lin_sizes[i], lin_sizes[i+1]) for i in range(len(lin_sizes) - 1)
        ])

        self.dropout = dropout

    def forward(self,x,edge_index):
        # No trim

        # convolutional layers
        for conv in self.convs:
            x = conv(x,edge_index)
            x = F.relu(x)
            x = F.dropout(x,p=self.dropout,training=self.training)

        # linear layers
        for i,lin in enumerate(self.linear):
            x = lin(x)
            if i < (len(self.linear) - 1):
                x = F.relu(x)
                x = F.dropout(x,p=self.dropout,training=self.training)
        
        return x