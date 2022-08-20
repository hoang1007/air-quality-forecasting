import torch
from torch import nn
from torch.nn.parameter import Parameter
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
        self.num_groups = config["num_groups"]

        self.input_embedding = nn.Linear(self.gnn_dim, self.input_embedding_dim)

        self.group_gnn = nn.ModuleList([GraphNode(self.input_embedding_dim, self.edge_dim, self.gnn_dim)])
        for _ in range(self.num_gnn_layers - 1):
            self.group_gnn.append(GraphNode(self.gnn_dim, self.edge_dim, self.gnn_dim))

        self.global_gnn = nn.ModuleList([GraphNode(self.input_embedding_dim + self.gnn_dim, 1, self.gnn_dim)])
        for _ in range(self.num_gnn_layers - 1):
            self.global_gnn.append(GraphNode(self.gnn_dim, 1, self.gnn_dim))

        self.fc = nn.Linear(self.gnn_dim, self.gnn_dim)

    def forward(
        self,
        x: torch.Tensor,
        src_len: int,
        enc_weights: torch.Tensor,
        dec_weights: torch.Tensor,
        group_edge_ids: torch.Tensor,
        group_edge_weights: torch.Tensor,
        edge_ids: torch.Tensor,
        edge_weights: torch.Tensor
    ):
        # assert enc_weights.requires_grad == False
        # assert dec_weights.requires_grad == False

        # enc_weights = torch.softmax(enc_weights, dim=-1)
        # dec_weights = torch.softmax(dec_weights, dim=-1)
        x = self.input_embedding(x)
        x = x.view(-1, src_len, x.size(-1))
        
        enc_weights = nn.parameter.Parameter(enc_weights, requires_grad=False)
        enc_w1 = enc_weights.transpose(0, 1).unsqueeze(0).repeat_interleave(x.size(0), 0)

        gr_feats = torch.bmm(enc_w1, x)
        gr_feats = gr_feats.view(-1, gr_feats.size(-1))

        for i in range(self.num_gnn_layers):
            gr_feats = self.group_gnn[i](gr_feats, group_edge_ids, group_edge_weights)
        gr_feats = gr_feats.reshape(-1, self.num_groups, gr_feats.size(-1))

        enc_w2 = enc_weights.unsqueeze(0).repeat_interleave(gr_feats.size(0), 0)

        new_x = torch.bmm(enc_w2, gr_feats)
        new_x = torch.cat((x, new_x), dim=-1)
        new_x = new_x.reshape(-1, new_x.size(-1))

        for i in range(self.num_gnn_layers):
            new_x = self.global_gnn[i](new_x, edge_ids, edge_weights)

        return new_x
        # dec_out = torch.bmm(dec_weights, gr_feats)
        # dec_out = self.fc(dec_out)

        # return dec_out