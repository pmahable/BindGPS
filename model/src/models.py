import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch_geometric.nn import SAGEConv
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


from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax

class TopKGATConv(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        concat,
        heads = 4,
        negative_slope = 0.2,
        dropout = 0.0,
        edge_dim = None,
        bias = True,
        use_topk = False,
        k = 10,
        # removed add_self_loops
    ):
        super().__init__(aggr='add', node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.edge_dim = edge_dim
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.bias = bias
        self.use_topk = use_topk
        self.k = k

        # linear projection prior to attention
        self.lin = nn.Linear(in_channels, heads * out_channels,  bias=False)

        self.att_source = nn.Parameter(torch.empty(1, heads, out_channels))
        self.att_dest = nn.Parameter(torch.empty(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
            self.att_edge = nn.Parameter(torch.empty(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.att_edge = self.register_parameter('att_edge', None)

        if bias:
            if self.concat:
                self.bias = nn.Parameter(torch.empty(heads * out_channels))
            else:
                self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_source)
        nn.init.xavier_uniform_(self.att_dest)

        if self.lin_edge is not None:
            nn.init.xavier_uniform_(self.lin_edge.weight)
            nn.init.xavier_uniform_(self.att_edge)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr=None, return_attention=False):
        H, C = self.heads, self.out_channels
        num_nodes = x.size(0)

        # x: [E, input_dim]
        # project all nodes using initial linear proj.
        x = self.lin(x) # [E, H * C]
        x = x.view(-1, H, C)
        x_src = x
        x_dst = x
        dst_idx = edge_index[1]

        # compute attention scores - x: [E, H, C]
        alpha_src = (x_src * self.att_source).sum(dim=-1) # (Num Edges, Num Heads) --> (E, H)
        alpha_dst = (x_dst * self.att_dest).sum(dim=-1) # (E, H)
        alpha = (alpha_src, alpha_dst) # (E, H)
        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr, size=None)
        
        # without Top K, functions as standard GATConv layer.
        if self.use_topk:
            alpha_mask = torch.zeros_like(alpha, dtype=torch.bool)
            for h in range(H):
                with torch.no_grad():
                    uniq_dst = torch.unique(dst_idx)
                    for node in uniq_dst.tolist():
                        idx = (dst_idx == node).nonzero(as_tuple=False).view(-1)
                        if idx.numel() == 0:
                            continue
                        a_j = alpha[idx, h]
                        k_local = min(self.k, a_j.numel())
                        topk_vals, topk_pos = torch.topk(a_j, k=k_local)
                        alpha_mask[idx[topk_pos], h] = True
            alpha = alpha.masked_fill(~alpha_mask, float('-inf'))

        alpha = softmax(alpha, dst_idx)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = self.propagate(edge_index, x=x, alpha=alpha)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return out, alpha

    def edge_update(self, alpha_j, alpha_i, edge_attr, index, ptr, dim_size):
        alpha = alpha_j + alpha_i # (E, H)
        
        # if no edges, return alpha
        if index.numel() == 0:
            return alpha

        # if we have edge features, add them to alpha logits
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1: # (E, ) --> single counts at the moment
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr) # (E, H * C)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels) # (E, H, C)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1) # (E, H)
            alpha = alpha + alpha_edge # (E, H)

        # postprocessing and return
        alpha = F.leaky_relu(alpha, self.negative_slope) # (E, H)

        return alpha

    def message(self, x_j, alpha):
        # x_j is in shape [E, H, C], alpha is in shape [E, H]
        # we want to broadcast alpha across the C dimension --> [E, H, 1]
        return alpha.unsqueeze(-1) * x_j
    
    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class GATModel(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_gnn_size,
            num_gnn_layers,
            hidden_linear_size,
            num_linear_layers,
            out_channels,
            bias = True,
            edge_dim = None,
            concat = True,
            heads = 4,
            negative_slope = 0.2,
            dropout=0.5,
            normalize=True,
            use_topk = False,
            k = 10,
            ):
        super().__init__()
        
        assert num_gnn_layers >= 1, "num_gnn_layers must be >= 1"
        assert num_linear_layers >= 1, "num_linear_layers counts the output layer; must be >= 1"

        if num_gnn_layers == 1:
            self.gat_convs = nn.ModuleList([TopKGATConv(in_channels, hidden_gnn_size, concat, heads, negative_slope, dropout, edge_dim, bias, use_topk, k)])
        else:
            first = [TopKGATConv(in_channels, hidden_gnn_size, concat, heads, negative_slope, dropout, edge_dim, bias, use_topk, k)]
            middle = [TopKGATConv(hidden_gnn_size*heads, hidden_gnn_size, concat, heads, negative_slope, dropout, edge_dim, bias, use_topk, k) for _ in range(num_gnn_layers - 2)]
            last = [TopKGATConv(hidden_gnn_size*heads, hidden_gnn_size, False, heads, negative_slope, dropout, edge_dim, bias, use_topk, k)]
            self.gat_convs = nn.ModuleList(first + middle + last)

        # linear layers
        lin_sizes = [hidden_gnn_size] + [hidden_linear_size] * max(0, num_linear_layers - 1) + [out_channels]
        self.linear = nn.ModuleList([
            nn.Linear(lin_sizes[i], lin_sizes[i+1]) for i in range(len(lin_sizes) - 1)
        ])

        self.dropout = dropout

    def forward(self,x,edge_index, edge_attr=None, return_attention=False):

        # convolutional layers
        for conv in self.gat_convs:
            x, alpha = conv(x,edge_index, edge_attr=edge_attr, return_attention=return_attention)
            
        # linear layers
        for i,lin in enumerate(self.linear):
            x = lin(x)
            if i < (len(self.linear) - 1):
                x = F.relu(x)
                x = F.dropout(x,p=self.dropout,training=self.training)
        
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