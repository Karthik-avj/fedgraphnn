import torch
from torch_geometric.nn import GINConv
from torch.nn import ReLU, Sequential, Linear, BatchNorm1d

class GINLinkPred(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels):
        super(GINLinkPred, self).__init__()
        self.conv1 = GINConv(
            Sequential(Linear(in_channels, hidden_dim), ReLU(),
                       Linear(hidden_dim, out_channels)))

    def encode(self, x, edge_index):
        return self.conv1(x, edge_index)

    def decode(self, z, pos_edge_index):
        edge_index = pos_edge_index
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()
