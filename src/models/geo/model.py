import torch
from torch import nn
import torch.nn.functional as F
from utils.functional import euclidean_distances
from .layers import GraphNode


class SpatialCorrelation(nn.Module):
    def __init__(self, config):
        """
        Args:
            inputs: Tensor (batch_size, n_stations, inseq_len, n_features)
            src_locs: Tensor (batch_size, n_stations, 2)
            tar_loc: Tensor (batch_size, 2)
            time: Tensor (batch_size, inseq_len, 4) (hour, day, month, solar_term)
        """
        super().__init__()

        self.n_features = config["n_features"]
        self.input_embedding_dim = config["input_embedding_dim"]
        self.loc_embedding_dim = config["loc_embedding_dim"]
        self.time_embedding_dim = config["time_embedding_dim"]
        self.num_gnn_layers = config["num_gnn_layers"]
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

        self.input_embedding = nn.Linear(self.inseq_len * self.n_features, self.input_embedding_dim)
        self.loc_embedding = nn.Linear(2, self.loc_embedding_dim)
        self.hour_embedding = nn.Embedding(24, self.time_embedding_dim)
        self.solar_term_embedding = nn.Embedding(24, self.time_embedding_dim)
        self.day_embedding = nn.Embedding(31, self.time_embedding_dim)
        self.month_embedding = nn.Embedding(12, self.time_embedding_dim)

        self.group_mapping = nn.Linear(2, self.num_groups)

        self.edge_inf = nn.Sequential(
            nn.Linear(self.input_embedding_dim * 2 + self.loc_embedding_dim * 2, self.edge_dim),
            nn.ReLU(inplace=True)
        )

        self.group_gnn = nn.ModuleList([GraphNode(self.input_embedding_dim + self.loc_embedding_dim, self.edge_dim, self.gnn_dim)])
        for _ in range(self.num_gnn_layers - 1):
            self.group_gnn.append(GraphNode(self.gnn_dim, self.edge_dim, self.gnn_dim))

        self.global_gnn = nn.ModuleList([GraphNode(self.input_embedding_dim + self.gnn_dim, 1, self.gnn_dim)])
        for _ in range(self.num_gnn_layers - 1):
            self.global_gnn.append(GraphNode(self.gnn_dim, 1, self.gnn_dim))

        self.decoder = None

    def _get_edge_attrs(self, locs: torch.Tensor):
        """
        Args:
            locs: (batch_size, num_stations, 2)
        
        Returns:
            weights: (batch_size, num_)
            ids: (batch_size, 2, num_)
        """

        dists, ids = [], []
        num_nodes = locs.size(0)

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    dist = (locs[:, i] - locs[:, j]).pow(2).sum(-1).sqrt()

                    dists.append(dist)
                    ids.append(locs.new_tensor([i, j], dtype=torch.long))

        dists = torch.stack(dists, dim=-1)
        ids = torch.stack(ids, dim=-1).unsqueeze(0).repeat_interleave(locs.size(0), dim=0)

        return dists, ids
    
    def batch_input(self, x, edge_weights, edge_ids):
        num_groups = x.size(1)
        x = x.reshape(-1, x.size(-1))
        edge_weights = edge_weights.reshape(-1, edge_weights.size(-1))

        for i in range(edge_ids.size(0)):
            edge_ids[i] = torch.add(edge_ids[i], i * num_groups)
        
        edge_ids = edge_ids.transpose(0, 1).reshape(2, -1)

        return x, edge_weights, edge_ids

    def forward(
        self,
        x: torch.Tensor,
        src_locs: torch.Tensor,
        tar_loc: torch.Tensor,
        time: torch.Tensor
    ):
        batch_size = x.size(0)
        n_stations = x.size(1)

        x = x.reshape(-1, x.size(-2), x.size(-1))
        x = self.enc_layer(x)
        x = x.reshape(-1, n_stations, x.size(-2) * x.size(-1))
        x = self.input_embedding(x)

        gr_weights = F.softmax(self.group_mapping(src_locs), dim=-1)
        loc_embed = self.loc_embedding(src_locs)

        in_locs = torch.cat((x, loc_embed), dim=-1)
        # group_feats.shape == (batch_size, n_groups, feat_size)
        group_feats = torch.bmm(gr_weights.transpose(1, 2), in_locs)

        # group gnn
        # hour_embed = self.hour_embedding(time[:, 0])

        gr_edge_weights = []
        gr_edge_ids = []

        for i in range(self.num_groups):
            for j in range(self.num_groups):
                if i == j:
                    continue
                gr_edge_in = torch.cat((group_feats[:, i], group_feats[:, j]), dim=-1)
                gr_edge_attr = self.edge_inf(gr_edge_in)
                gr_edge_idx = x.new_tensor([i, j], dtype=torch.long)

                gr_edge_weights.append(gr_edge_attr)
                gr_edge_ids.append(gr_edge_idx)        

        # gr_edge_weights.shape == (batch_size, n_gr * (n_gr - 1), edge_dim)
        gr_edge_weights = torch.stack(gr_edge_weights, dim=1)
        gr_edge_ids = torch.stack(gr_edge_ids, dim=1).unsqueeze(0)
        gr_edge_ids = gr_edge_ids.repeat_interleave(batch_size, dim=0)

        group_feats, gr_edge_weights, gr_edge_ids = self.batch_input(group_feats, gr_edge_weights, gr_edge_ids)
        for i in range(self.num_gnn_layers):
            group_feats = self.group_gnn[i](group_feats, gr_edge_ids, gr_edge_weights)

        group_feats = group_feats.reshape(batch_size, self.num_groups, -1)

        new_x = torch.bmm(gr_weights, group_feats)
        new_x = torch.cat((x, new_x), dim=-1)
        edge_weights, edge_ids = self._get_edge_attrs(src_locs)
        edge_weights = edge_weights.unsqueeze(-1)

        # new_x.shape == (batch_size', gnn_dim)
        new_x, edge_weights, edge_ids = self.batch_input(new_x, edge_weights, edge_ids)
        for i in range(self.num_gnn_layers):
            new_x = self.global_gnn[i](new_x, edge_ids, edge_weights)