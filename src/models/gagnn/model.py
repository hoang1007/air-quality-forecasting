import torch
from torch import nn
import torch.nn.functional as F
from .layers import DecoderModule, GraphNode
from models import BaseAQFModel


class GAGNNModel(BaseAQFModel):
    def __init__(self, config, target_normalize_mean: float, target_normalize_std: float):
        """
        Args:
            inputs: Tensor (batch_size, n_stations, inseq_len, n_features)
            src_locs: Tensor (batch_size, n_stations, 2)
            tar_loc: Tensor (batch_size, 2)
            time: Tensor (batch_size, inseq_len, 4) (hour, day, month, solar_term)
        """
        super().__init__(config, target_normalize_mean, target_normalize_std)

        self.n_features = config["n_features"] + 4 * config["time_embedding_dim"]
        self.input_embedding_dim = config["input_embedding_dim"]
        self.loc_embedding_dim = config["loc_embedding_dim"]
        self.time_embedding_dim = config["time_embedding_dim"]
        self.num_gnn_layers = config["num_enc_gnn_layers"]
        self.num_attn_heads = config["num_attn_heads"]
        self.attn_ff_dim = config["attn_feedforward_dim"]
        self.edge_dim = config["edge_dim"]
        self.gnn_dim = config["gnn_dim"]
        self.num_groups = config["num_groups"]
        self.inseq_len = config["inseq_len"]
        self.outseq_len = config["outseq_len"]
        self.dropout = config["dropout"]


        self.enc_layer = nn.TransformerEncoderLayer(
            self.n_features,
            nhead=5,
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

        self.input_embedding = nn.Linear(self.inseq_len * self.n_features, self.input_embedding_dim)
        self.loc_embedding = nn.Linear(2, self.loc_embedding_dim)
        self.hour_embedding = nn.Embedding(24, self.time_embedding_dim)
        self.solar_term_embedding = nn.Embedding(24, self.time_embedding_dim)
        self.day_embedding = nn.Embedding(31, self.time_embedding_dim)
        self.month_embedding = nn.Embedding(12, self.time_embedding_dim)

        self.w = nn.parameter.Parameter(torch.randn((11, self.num_groups), device=self.device), requires_grad=True)

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

        self.decoder = DecoderModule(config)
        self.predictor = nn.Sequential(
            nn.Linear(self.gnn_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, self.outseq_len),
            # nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        src_locs: torch.Tensor,
        tar_locs: torch.Tensor,
        time: torch.Tensor
    ):  
        # loc_mean_ = x.new_tensor([105.8010,  21.0294])
        # loc_std_ = x.new_tensor([0.0553, 0.0131])
        
        # src_locs = (src_locs - loc_mean_) / loc_std_
        # tar_locs = (tar_locs - loc_mean_) / loc_std_

        batch_size = x.size(0)
        n_stations = x.size(1)

        # time.shape == (batch_size, n_stations, seq_len, n_timefeats)
        time = time.reshape(batch_size * n_stations, -1, time.size(-1))
        hour_embed = self.hour_embedding(time[..., 0]).view(batch_size, n_stations, -1, self.time_embedding_dim)
        day_embed = self.day_embedding(time[..., 1] - 1).view(batch_size, n_stations, -1, self.time_embedding_dim)
        month_embed = self.month_embedding(time[..., 2] - 1).view(batch_size, n_stations, -1, self.time_embedding_dim)
        solar_term_embed = self.solar_term_embedding(time[..., 3]).view(batch_size, n_stations, -1, self.time_embedding_dim)

        x = torch.cat((x, hour_embed, day_embed, month_embed, solar_term_embed), dim=-1).float()

        x = x.reshape(-1, x.size(-2), x.size(-1))
        x = self.enc_layer(x)
        x = x.reshape(-1, n_stations, x.size(-2) * x.size(-1))
        x = self.input_embedding(x)
        # x, _ = self.enc_layer(x)
        # x = x[:, -1].contiguous().view(batch_size, n_stations, -1)

        w = F.softmax(self.w, dim=-1)
        w1 = w.transpose(0, 1).unsqueeze(0)
        w1 = w1.repeat_interleave(batch_size, dim=0)
        # enc_weights = self.group_mapping(src_locs)

        loc_embed = self.loc_embedding(src_locs)

        in_locs = torch.cat((x, loc_embed), dim=-1)
        # group_feats.shape == (batch_size, n_groups, feat_size)
        group_feats = torch.bmm(w1, in_locs)

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

        w2 = w.unsqueeze(dim=0)
        w2 = w2.repeat_interleave(batch_size, dim=0)

        new_x = torch.bmm(w2, group_feats)
        new_x = torch.cat((x, new_x), dim=-1)
        edge_weights, edge_ids = self._get_edge_attrs(src_locs)
        edge_weights = edge_weights.unsqueeze(-1)

        # new_x.shape == (batch_size', gnn_dim)
        new_x, edge_weights, edge_ids = self.batch_input(new_x, edge_weights, edge_ids)
        for i in range(self.num_gnn_layers):
            new_x = self.global_gnn[i](new_x, edge_ids, edge_weights)
        
        # dec_weights = self.group_mapping(tar_locs)
        dec_weights = None
        # dec_outs.shape == (batch_size, n_stations, gnn_dim)
        dec_outs = self.decoder(
            new_x,
            n_stations,
            # enc_weights,
            self.w,
            dec_weights,
            gr_edge_ids,
            gr_edge_weights,
            edge_ids,
            edge_weights
        )

        preds = self.predictor(dec_outs).view(batch_size, -1, self.outseq_len)

        return preds * self.target_normalize_std + self.target_normalize_mean

    def _get_edge_attrs(self, locs: torch.Tensor):
        """
        Args:
            locs: (batch_size, num_stations, 2)
        
        Returns:
            weights: (batch_size, num_)
            ids: (batch_size, 2, num_)
        """

        weights, ids = [], []
        num_nodes = locs.size(1)

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    dist = (locs[:, i] - locs[:, j]).pow(2).sum(-1).sqrt()

                    weights.append(1 / dist)
                    ids.append(locs.new_tensor([i, j], dtype=torch.long))

        weights = torch.stack(weights, dim=-1)
        ids = torch.stack(ids, dim=-1).unsqueeze(0).repeat_interleave(locs.size(0), dim=0)

        return weights, ids
    
    def batch_input(self, x, edge_weights, edge_ids):
        num_groups = x.size(1)
        x = x.reshape(-1, x.size(-1))
        edge_weights = edge_weights.reshape(-1, edge_weights.size(-1))

        for i in range(edge_ids.size(0)):
            edge_ids[i] = torch.add(edge_ids[i], i * num_groups)
        
        edge_ids = edge_ids.transpose(0, 1).reshape(2, -1)

        return x, edge_weights, edge_ids

    def compute_loss(self, input: torch.Tensor, target: torch.Tensor, reduction="mean"):
        return F.l1_loss(input, target, reduction=reduction)

    def training_step(self, batch, batch_idx):
        # outs = self(batch["features"], batch["src_locs"], batch["tar_locs"], None)
        outs = self(batch["features"], batch["src_locs"], None, batch["time"])

        # loss = self.compute_loss(outs, batch["target"])
        loss = self.compute_loss(outs, batch["src_nexts"], reduction="mean")

        self.log("loss", loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        # pres.shape == (batch_size, n_src_stations, output_size)
        # outs = self(batch["features"], batch["src_locs"], batch["tar_locs"], None)
        outs = self(batch["features"], batch["src_locs"], None, batch["time"])

        # mae = self.compute_loss(outs, batch["gt_target"])
        mae = self.compute_loss(outs, batch["src_nexts"], reduction="sum")

        return {"mae": mae}
        # return self.metric_fns(outs.squeeze(), batch["gt_target"].squeeze())

    def predict(self, dt):
        st = self.training
        self.train(False)
        with torch.no_grad():
            # add batch dim
            features = dt["features"].unsqueeze(0).to(self.device)
            src_locs = dt["src_locs"].unsqueeze(0).to(self.device)
            # tar_loc = dt["tar_locs"].unsqueeze(0).to(self.device)
            time = dt["time"].unsqueeze(0).to(self.device)

            # output = self(features, src_locs, tar_loc, None).squeeze(0)
            output = self(features, src_locs, None, time).squeeze(0)
            # inverse transforms
            # output = output * self.target_normalize_std + self.target_normalize_mean

        self.train(st)
        return output