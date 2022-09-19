import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, GATv2Conv
from models import BaseAQFModel
from .utils import inverse_distance_weighting


class AQFModel(BaseAQFModel):
    def __init__(self, config, target_normalize_mean: float, target_normalize_std: float):
        super().__init__(config, target_normalize_mean, target_normalize_std)

        self.loc_dim = 2
        self.dist_thresh_air = config["dist_thresh_air"]
        self.dist_thresh_meteo = config["dist_thresh_meteo"]
        self.dist_type = config["dist_type"]

        if config["extractor"] == "gru":
            self.air_extractor = nn.GRU(
                input_size=config["num_air_features"],
                hidden_size=config["hidden_size"],
                num_layers=config["num_extractor_layers"],
                batch_first=True,
                dropout=config["dropout"],
                bidirectional=config["bidirectional"]
            )

            self.meteo_extractor = nn.GRU(
                input_size=config["num_meteo_features"],
                hidden_size=config["hidden_size"],
                num_layers=config["num_extractor_layers"],
                batch_first=True,
                dropout=config["dropout"],
                bidirectional=config["bidirectional"]
            )
        elif config["extractor"] == "lstm":
            self.air_extractor = nn.LSTM(
                input_size=config["num_air_features"],
                hidden_size=config["hidden_size"],
                num_layers=config["num_extractor_layers"],
                batch_first=True,
                dropout=config["dropout"],
                bidirectional=config["bidirectional"]
            )

            self.meteo_extractor = nn.LSTM(
                input_size=config["num_meteo_features"],
                hidden_size=config["hidden_size"],
                num_layers=config["num_extractor_layers"],
                batch_first=True,
                dropout=config["dropout"],
                bidirectional=config["bidirectional"]
            )
        else:
            raise ValueError(f"Unknown extractor: {config['extractor']}")

        self.gconv1 = GraphConv(
            (config["hidden_size"] * (2 if config["bidirectional"] else 1), self.loc_dim),
            config["gcn_dim"],
            aggr="mean"
        )

        self.gconv2 = GraphConv(
            (config["hidden_size"] * (2 if config["bidirectional"] else 1), self.loc_dim),
            config["gcn_dim"],
            aggr="mean"
        )

        self.fc = nn.Sequential(
            nn.Linear(config["gcn_dim"] * 2, config["gcn_dim"]),
            nn.ReLU(),
            nn.Linear(config["gcn_dim"], config["outseq_len"])
        )

    def forward(
        self,
        air: torch.Tensor,
        meteo: torch.Tensor,
        air_locs: torch.Tensor,
        meteo_locs: torch.Tensor,
        tar_locs: torch.Tensor
    ):
        """
        Args:
            air: (batch_size, num_air_nodes, seq_len1, n_air_feats)
            meteo: (batch_size, num_meteo_nodes, seq_len2, n_meteo_feats)
            air_locs: (num_air_nodes, 2)
            meteo_locs: (num_meteo_nodes, 2)
        """

        batch_size, num_air_nodes = air.shape[0:2]
        num_meteo_nodes = meteo.size(1)

        air = air.view(-1, air.size(2), air.size(3))
        # air.shape == (batch_size, num_nodes, hidden_size)
        air, _ = self.air_extractor(air)
        air = air[:, -1].view(batch_size, num_air_nodes, -1)

        meteo = meteo.view(-1, meteo.size(2), meteo.size(3))
        # meteo.shape == (batch_size, num_nodes, hidden_size)
        meteo, _ = self.meteo_extractor(meteo)
        meteo = meteo[:, -1].view(batch_size, num_meteo_nodes, -1)

        tar_loc_embed = self._forward_loc_em(tar_locs)

        gconv1_out = self._forward_gconv1(air, tar_loc_embed, air_locs, tar_locs)
        gconv2_out = self._forward_gconv2(meteo, tar_loc_embed, meteo_locs, tar_locs)

        # gconv_out.shape == (batch_size, num_air_nodes, 2 * gcn_dim)
        gconv_out = torch.cat((gconv1_out, gconv2_out), dim=-1)

        pred = self.fc(gconv_out)

        return pred
    
    def _forward_loc_em(self, locs: torch.Tensor):
        # (tensor([105.7804,  21.0309]), tensor([0.1088, 0.0793]))
        # pre-computed
        loc_mean_ = locs.new_tensor([105.7804,  21.0309])
        loc_std_ = locs.new_tensor([0.1088, 0.0793])

        loc_embed = (locs - loc_mean_) / loc_std_
        # loc_embed = self.loc_embedding(loc_embed)

        return loc_embed

    def _forward_gconv1(self, src_x: torch.Tensor, tar_x: torch.Tensor, src_locs: torch.Tensor, tar_locs: torch.Tensor):
        batch_size = src_x.size(0)
        edge_weights, edge_ids = [], []

        for batch_idx in range(batch_size):
            temp_w, temp_idx = inverse_distance_weighting(
                src_locs[batch_idx],
                tar_locs[batch_idx],
                dist_thresh=self.dist_thresh_air,
                dist_type=self.dist_type,
                norm=True
            )

            edge_weights.append(temp_w)
            edge_ids.append(temp_idx)

        src_x, tar_x, edge_ids, edge_weights = self._batch_gconv_input(src_x, tar_x, edge_ids, edge_weights)

        out = self.gconv1(
            (src_x, tar_x), edge_ids, edge_weights)
        out = out.view(batch_size, -1, out.size(-1))
        return out

    def _forward_gconv2(self, src_x: torch.Tensor, tar_x: torch.Tensor, src_locs: torch.Tensor, tar_locs: torch.Tensor):
        batch_size = src_x.size(0)
        edge_weights, edge_ids = [], []

        for batch_idx in range(src_x.size(0)):
            temp_w, temp_idx = inverse_distance_weighting(
                src_locs[batch_idx],
                tar_locs[batch_idx],
                dist_thresh=self.dist_thresh_meteo,
                dist_type=self.dist_type,
                norm=True
            )

            edge_weights.append(temp_w)
            edge_ids.append(temp_idx)

        src_x, tar_x, edge_ids, edge_weights = self._batch_gconv_input(src_x, tar_x, edge_ids, edge_weights)

        out = self.gconv2(
            (src_x, tar_x), edge_ids, edge_weights)
        out = out.view(batch_size, -1, out.size(-1))

        return out

    def _batch_gconv_input(self, src_x: torch.Tensor, tar_x: torch.Tensor, edge_ids: torch.Tensor, edge_weights: torch.Tensor):
        """
        Args:
            src_x: Tensor (batch_size, n1, num_features)
            tar_x: Tensor (batch_size, n2, num_features)
            edge_ids: List of (2, num_edges)
            edge_weights: List of (num_edges)
        """

        num_nodes = torch.tensor((src_x.size(1), tar_x.size(1)), dtype=torch.long, device=src_x.device).unsqueeze(-1)

        src_x = src_x.view(-1, src_x.size(-1))
        tar_x = tar_x.view(-1, tar_x.size(-1))

        for i in range(len(edge_ids)): # batch
            edge_ids[i] = edge_ids[i] + i * num_nodes

        edge_weights = torch.hstack(edge_weights)
        edge_ids = torch.cat(edge_ids, dim=1)

        return src_x, tar_x, edge_ids, edge_weights

    def compute_loss(self, input: torch.Tensor, target: torch.Tensor):
        return F.l1_loss(input, target)

    def training_step(self, batch, batch_idx):
        pred = self(batch["air"], batch["meteo"],
                    batch["air_locs"], batch["meteo_locs"], batch["tar_locs"])

        target = (batch["targets"] - self.target_normalize_mean) / \
            self.target_normalize_std
        loss = self.compute_loss(pred, target)

        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch["air"], batch["meteo"],
                    batch["air_locs"], batch["meteo_locs"], batch["tar_locs"])

        pred = pred * self.target_normalize_std + self.target_normalize_mean
        return self.metric_fns(pred, batch["targets"])

    def predict(self, dt):
        st = self.training
        self.train(False)
        with torch.no_grad():
            air = dt["air"].to(self.device).unsqueeze(0)
            meteo = dt["meteo"].to(self.device).unsqueeze(0)
            air_locs = dt["air_locs"].to(self.device).unsqueeze(0)
            meteo_locs = dt["meteo_locs"].to(self.device).unsqueeze(0)
            tar_locs = dt["tar_locs"].to(self.device).unsqueeze(0)

            preds = self(air, meteo, air_locs, meteo_locs, tar_locs).squeeze(0)
            preds = preds * self.target_normalize_std + self.target_normalize_mean
        self.train(st)

        return preds
