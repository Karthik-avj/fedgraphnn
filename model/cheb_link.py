import torch
from torch_geometric.nn import ChebConv

class ChebLinkPred(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels):
        super(ChebLinkPred, self).__init__()
        self.conv1 = ChebConv(in_channels, hidden_dim, K=1)
        self.conv2 = ChebConv(hidden_dim, out_channels, K=1)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        return self.conv2(x, edge_index)

    def decode(self, z, pos_edge_index):
        edge_index = pos_edge_index
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()
