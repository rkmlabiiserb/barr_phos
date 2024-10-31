import numpy as np
import torch
from torch.nn import Linear, Module, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data
from torch.optim import Adam

class Message_Passing_Class(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        out = self.propagate(edge_index, x=x, norm=norm)
        out = out + self.bias
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class EdgeAggregationLayer(Module):
    def __init__(self, in_channels, edge_in_channels, out_channels):
        super().__init__()
        self.node_to_edge_lin = Linear(in_channels, edge_in_channels, bias=False)
        self.edge_lin = Linear(edge_in_channels, out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.node_to_edge_lin.reset_parameters()
        self.edge_lin.reset_parameters()

    def forward(self, x, edge_index):
        row, col = edge_index
        edge_features = self.node_to_edge_lin(x[row])
        edge_features = self.edge_lin(edge_features)
        return edge_features

class EdgeToNodeAggregationLayer(Module):
    def __init__(self, in_channels, edge_in_channels):
        super().__init__()
        self.edge_to_node_lin = Linear(edge_in_channels, in_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.edge_to_node_lin.reset_parameters()

    def forward(self, edge_features, edge_index, num_nodes):
        row, col = edge_index
        edge_features = self.edge_to_node_lin(edge_features)
        node_features = torch.zeros((num_nodes, edge_features.size(1)), device=edge_features.device)
        node_features.index_add_(0, row, edge_features)
        return node_features

class GNNAutoencoder(Module):
    def __init__(self, in_channels, hidden_channels, edge_in_channels, out_channels):
        super().__init__()
        self.message_passing = Message_Passing_Class(in_channels, hidden_channels)
        self.edge_agg = EdgeAggregationLayer(hidden_channels, edge_in_channels, out_channels)
        self.edge_to_node_agg = EdgeToNodeAggregationLayer(hidden_channels, out_channels)
        self.node_recon = Linear(hidden_channels, in_channels, bias=False)

    def forward(self, x, edge_index):
        x = self.message_passing(x, edge_index)
        edge_features = self.edge_agg(x, edge_index)
        num_nodes = x.size(0)
        x_recon = self.edge_to_node_agg(edge_features, edge_index, num_nodes)
        x_recon = self.node_recon(x_recon)
        return x_recon, edge_features

nodes = np.loadtxt('nodes.dat', dtype=int)
num_nodes = len(nodes)
all_features = np.load('features_3d_nodes.npy')
edges = np.loadtxt('edges.out', dtype=int)
edge_index_raw = edges - 1
max_frames = all_features.shape[0]

allevalloss = []
all_latent_space = np.zeros(len(edges))

for frame in range(max_frames):
    frame_node_features = all_features[frame]
    x = torch.tensor(frame_node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index_raw, dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    num_features = x.size(1)
    hidden_channels = 2
    edge_in_channels = 4
    out_channels = 1
    model = GNNAutoencoder(num_features, hidden_channels, edge_in_channels, out_channels).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    num_epochs = 500
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        x_recon, _ = model(data.x, data.edge_index)
        loss = criterion(x_recon, data.x)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        x_recon, edge_features = model(data.x, data.edge_index)
        eval_loss = criterion(x_recon, data.x)
        allevalloss.append([eval_loss.item()])
        np.savetxt('eval_loss.txt', allevalloss, fmt='%.4f')
        all_latent_space += np.transpose(edge_features.detach().cpu().numpy())[0]

    torch.cuda.empty_cache()

np.savetxt('edge_weight.txt', all_latent_space/max_frames, fmt='%.4f', delimiter=' ')
