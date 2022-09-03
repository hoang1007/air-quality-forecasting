import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_scatter import scatter_mean
from .utils import batch_gnn_input


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


class GAGNNEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_stations = config["num_stations"]
        self.n_features = config["n_features"]
        self.input_embedding_dim = config["input_embedding_dim"]
        self.time_embedding_dim = config["time_embedding_dim"]
        self.loc_embedding_dim = config["loc_embedding_dim"]
        self.num_gnn_layers = config["num_enc_gnn_layers"]
        self.num_attn_heads = config["num_attn_heads"]
        self.attn_ff_dim = config["attn_feedforward_dim"]
        self.edge_dim = config["edge_dim"]
        self.gnn_dim = config["gnn_dim"]
        self.num_groups = config["num_groups"]
        self.inseq_len = config["inseq_len"]
        self.dropout = config["dropout"]

        self.enc_layer = nn.TransformerEncoderLayer(
            self.n_features,
            nhead=self.num_attn_heads,
            dim_feedforward=self.attn_ff_dim,
            dropout=self.dropout,
            batch_first=True
        )
        # self.enc_layer = nn.LSTM(
        #     input_size=self.n_features,
        #     hidden_size=self.input_embedding_dim,
        #     num_layers=1,
        #     batch_first=True,
        #     dropout=self.dropout
        # )

        self.register_parameter("w", self._init_weights(self.num_stations, self.num_groups))

        self.input_embedding = nn.Linear(self.inseq_len * self.n_features, self.input_embedding_dim)
        self.loc_embedding = nn.Linear(2, self.loc_embedding_dim)

        self.edge_inf = nn.Sequential(
            nn.Linear(self.input_embedding_dim * 2 + self.loc_embedding_dim * 2 + self.time_embedding_dim * 3, self.edge_dim),
            nn.Dropout(p=self.dropout),
            nn.ReLU(inplace=True),
        )

        self.group_gnn = nn.ModuleList([GraphNode(self.input_embedding_dim + self.loc_embedding_dim, self.edge_dim, self.gnn_dim)])
        for _ in range(self.num_gnn_layers - 1):
            self.group_gnn.append(GraphNode(self.gnn_dim, self.edge_dim, self.gnn_dim))

        self.global_gnn = nn.ModuleList([GraphNode(self.input_embedding_dim + self.gnn_dim, 1, self.gnn_dim)])
        for _ in range(self.num_gnn_layers - 1):
            self.global_gnn.append(GraphNode(self.gnn_dim, 1, self.gnn_dim))

    def _init_weights(self, num_stations: int, num_groups: int):
        x = torch.zeros(num_stations, num_groups)
        nn.init.xavier_uniform_(x)

        return nn.parameter.Parameter(x, requires_grad=True)

    @property
    def weights(self):
        return self.w.detach()

    def forward(
        self,
        x: torch.Tensor,
        locs: torch.Tensor,
        time_embed: torch.Tensor,
        edge_weights: torch.Tensor,
        edge_ids: torch.Tensor
    ):
        batch_size = x.size(0)

        x = x.reshape(-1, x.size(-2), x.size(-1))
        x = self.enc_layer(x)
        x = x.reshape(-1, self.num_stations, x.size(-2) * x.size(-1))
        x = self.input_embedding(x)

        w = F.softmax(self.w, dim=-1)
        w1 = w.transpose(0, 1).unsqueeze(0)
        w1 = w1.repeat_interleave(batch_size, dim=0)

        loc_embed = self.loc_embedding(locs)

        in_locs = torch.cat((x, loc_embed), dim=-1)
        # group_feats.shape == (batch_size, n_groups, feat_size)
        group_feats = torch.bmm(w1, in_locs)

        gr_edge_weights = []
        gr_edge_ids = []

        for i in range(self.num_groups):
            for j in range(self.num_groups):
                if i == j:
                    continue
                gr_edge_in = torch.cat((group_feats[:, i], group_feats[:, j], time_embed), dim=-1)
                gr_edge_attr = self.edge_inf(gr_edge_in)
                gr_edge_idx = x.new_tensor([i, j], dtype=torch.long)

                gr_edge_weights.append(gr_edge_attr)
                gr_edge_ids.append(gr_edge_idx)

        gr_edge_weights = torch.stack(gr_edge_weights, dim=1)
        gr_edge_ids = torch.stack(gr_edge_ids, dim=1).unsqueeze(0)
        gr_edge_ids = gr_edge_ids.repeat_interleave(batch_size, dim=0)

        group_feats, gr_edge_weights, gr_edge_ids = batch_gnn_input(group_feats, gr_edge_weights, gr_edge_ids)
        for i in range(self.num_gnn_layers):
            group_feats = self.group_gnn[i](group_feats, gr_edge_ids, gr_edge_weights)

        group_feats = group_feats.reshape(batch_size, self.num_groups, -1)

        w2 = w.unsqueeze(dim=0)
        w2 = w2.repeat_interleave(batch_size, dim=0)

        new_x = torch.bmm(w2, group_feats)
        new_x = torch.cat((x, new_x), dim=-1)

        # new_x.shape == (batch_size', gnn_dim)
        new_x, edge_weights, edge_ids = batch_gnn_input(new_x, edge_weights.unsqueeze(-1), edge_ids)
        for i in range(self.num_gnn_layers):
            new_x = self.global_gnn[i](new_x, edge_ids, edge_weights)

        new_x = new_x.view(batch_size, self.num_stations, -1)
        return new_x, gr_edge_weights, gr_edge_ids


class GAGNNDecoder(nn.Module):
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
        x = self.input_embedding(x)
        x = x.view(-1, src_len, x.size(-1))
        
        enc_weights = torch.softmax(enc_weights, dim=-1)

        w1 = enc_weights.transpose(0, 1).unsqueeze(0).repeat_interleave(x.size(0), 0)

        gr_feats = torch.bmm(w1, x)
        gr_feats = gr_feats.view(-1, gr_feats.size(-1))

        for i in range(self.num_gnn_layers):
            gr_feats = self.group_gnn[i](gr_feats, group_edge_ids, group_edge_weights)
        gr_feats = gr_feats.reshape(-1, self.num_groups, gr_feats.size(-1))

        dec_weights = torch.softmax(dec_weights, dim=-1)
        w2 = dec_weights.unsqueeze(0).repeat_interleave(gr_feats.size(0), 0)

        new_x = torch.bmm(w2, gr_feats)
        new_x = torch.cat((x, new_x), dim=-1)
        
        new_x, edge_weights, edge_ids = batch_gnn_input(new_x, edge_weights.unsqueeze(-1), edge_ids)

        for i in range(self.num_gnn_layers):
            new_x = self.global_gnn[i](new_x, edge_ids, edge_weights)

        return new_x