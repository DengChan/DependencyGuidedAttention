import torch.nn as nn
import torch.nn.functional as F

def gelu(x):
    # return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GCN(nn.Module):
    """ A GCN/Contextualized GCN module operated on dependency graphs. """
    def __init__(self, in_dim, out_dim, num_layers, in_drop_prob, gcn_drop_prob):
        super(GCN, self).__init__()
        self.W = nn.ModuleList()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = num_layers
        self.in_drop = nn.Dropout(in_drop_prob)
        self.gcn_drop = nn.Dropout(gcn_drop_prob)
        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.out_dim
            self.W.append(nn.Linear(input_dim, self.out_dim))

    def forward(self, adj, gcn_inputs):

        gcn_inputs = self.in_drop(gcn_inputs)

        denom = adj.sum(2).unsqueeze(2) + 1

        for l in range(self.layers):
            Ax = adj.bmm(gcn_inputs)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](gcn_inputs)  # self loop
            AxW = AxW / denom

            # gAxW = F.relu(AxW)
            gAxW = gelu(AxW)
            gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW

        return gcn_inputs
