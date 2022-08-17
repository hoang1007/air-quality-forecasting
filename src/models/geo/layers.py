import torch
from torch import nn
from torch_scatter import scatter_mean


class GraphNode(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, gnn_dim: int):
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(node_dim + edge_dim, gnn_dim),
            nn.ReLU(inplace=True),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(node_dim + gnn_dim, gnn_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, edge_ids, edge_attr):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]

        row, col = edge_ids
        out = torch.cat((x[row], edge_attr), dim=1)
        out = self.fc1(out)

        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat((x, out), dim=1)

        out = self.fc2(out)

        return out


class DecoderModule(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_gnn_layers = config["num_dec_gnn_layers"]
        self.input_embedding_dim = config["input_embedding_dim"]
        self.edge_dim = config["edge_dim"]
        self.gnn_dim = config["gnn_dim"]

        self.input_embedding = nn.Linear(self.gnn_dim, self.input_embedding_dim)

        self.group_gnn = nn.ModuleList([GraphNode(self.input_embedding_dim, self.edge_dim, self.gnn_dim)])
        for _ in range(self.num_gnn_layers - 1):
            self.group_gnn.append(GraphNode(self.gnn_dim, self.edge_dim, self.gnn_dim))

        self.global_gnn = nn.ModuleList([GraphNode(self.input_embedding_dim + self.gnn_dim, 1, self.gnn_dim)])
        for _ in range(self.num_gnn_layers - 1):
            self.global_gnn.append([GraphNode(self.gnn_dim, 1, self.gnn_dim)])

    def forward(
        self,
        input,
        tar_group_weights,
        enc_weights,
        group_edge_ids,
        group_edge_weights,
        edge_ids,
        edge_weights
    ):
        input = self.input_embedding(input)
        