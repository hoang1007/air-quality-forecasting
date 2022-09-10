import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv
from models import BaseAQFModel
from .utils import inverse_distance_weighting


class AQFModel(BaseAQFModel):
    def __init__(self, config, target_normalize_mean: float, target_normalize_std: float):
        super().__init__(config, target_normalize_mean, target_normalize_std)

        self.air_extractor = nn.GRU(
            input_size=config["num_air_features"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_extractor_layers"],
            batch_first=True,
            dropout=config["dropout"],
            bidirectional=True
        )

        self.meteo_extractor = nn.GRU(
            input_size=config["num_meteo_features"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_extractor_layers"],
            batch_first=True,
            dropout=config["dropout"],
            bidirectional=True
        )

        self.gconv1 = GCNConv(
            config["hidden_size"] * 2,
            config["gcn_dim"]
        )

        self.gconv2 = GraphConv(
            (config["hidden_size"] * 2, config["hidden_size"] * 2),
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
        meteo_locs: torch.Tensor
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

        gconv1_out = self._forward_gconv1(air, air_locs)
        gconv2_out = self._forward_gconv2(air, meteo, air_locs, meteo_locs)

        # gconv_out.shape == (batch_size, num_air_nodes, 2 * gcn_dim)
        gconv_out = torch.cat((gconv1_out, gconv2_out), dim=-1)

        pred = self.fc(gconv_out)

        return pred

    def _forward_gconv1(self, x: torch.Tensor, locs: torch.Tensor):
        edge_weights, edge_ids = inverse_distance_weighting(locs, dist_thresh=0.3)

        out = []
        for batch_idx in range(x.size(0)):
            temp = self.gconv1(x[batch_idx], edge_ids, edge_weights)

            out.append(temp)

        return torch.stack(out, dim=0)

    def _forward_gconv2(self, air: torch.Tensor, meteo: torch.Tensor, air_locs: torch.Tensor, meteo_locs: torch.Tensor):
        edge_weights, edge_ids = inverse_distance_weighting(meteo_locs, air_locs, dist_thresh=0.8)

        out = []
        for batch_idx in range(air.size(0)):
            temp = self.gconv2((meteo[batch_idx], air[batch_idx]), edge_ids, edge_weights)

            out.append(temp)

        return torch.stack(out, dim=0)

    def compute_loss(self, input: torch.Tensor, target: torch.Tensor):
        return F.l1_loss(input, target)

    def training_step(self, batch, batch_idx):
        pred = self(batch["air"], batch["meteo"], batch["air_locs"][0], batch["meteo_locs"][0])

        target = (batch["targets"] - self.target_normalize_mean) / self.target_normalize_std
        loss = self.compute_loss(pred, target)

        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch["air"], batch["meteo"], batch["air_locs"][0], batch["meteo_locs"][0])

        pred = pred * self.target_normalize_std + self.target_normalize_mean
        return self.metric_fns(pred, batch["targets"])

    def predict(self, dt):
        st = self.training
        self.train(False)
        with torch.no_grad():
            air = dt["air"].to(self.device).unsqueeze(0)
            meteo = dt["meteo"].to(self.device).unsqueeze(0)
            air_locs = dt["air_locs"].to(self.device)
            meteo_locs = dt["meteo_locs"].to(self.device)

            preds = self(air, meteo, air_locs, meteo_locs).squeeze(0)
            preds = preds * self.target_normalize_std + self.target_normalize_mean
        self.train(st)

        return preds
