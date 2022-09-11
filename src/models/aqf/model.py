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

        self.gconv1 = GraphConv(
            (config["hidden_size"] * 2 + self.loc_dim, self.loc_dim),
            config["gcn_dim"],
            aggr="mean"
        )

        self.gconv2 = GraphConv(
            (config["hidden_size"] * 2 + self.loc_dim, self.loc_dim),
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

        new_air = torch.cat((air, self._forward_loc_em(air_locs)), dim=-1)
        new_meteo = torch.cat((meteo, self._forward_loc_em(meteo_locs)), dim=-1)

        with torch.no_grad():
            tar_loc_embed = self._forward_loc_em(tar_locs)

        gconv1_out = self._forward_gconv1(new_air, tar_loc_embed, air_locs, tar_locs)
        gconv2_out = self._forward_gconv2(new_meteo, tar_loc_embed, meteo_locs, tar_locs)

        # gconv_out.shape == (batch_size, num_air_nodes, 2 * gcn_dim)
        gconv_out = torch.cat((gconv1_out, gconv2_out), dim=-1)

        pred = self.fc(gconv_out)

        return pred
    
    def _forward_loc_em(self, locs: torch.Tensor):
        # (tensor([105.7804,  21.0309]), tensor([0.1088, 0.0793]))
        loc_mean_ = locs.new_tensor([105.7804,  21.0309])
        loc_std_ = locs.new_tensor([0.1088, 0.0793])

        loc_embed = (locs - loc_mean_) / loc_std_
        # loc_embed = self.loc_embedding(loc_embed)

        return loc_embed

    def _forward_gconv1(self, src_x: torch.Tensor, tar_x: torch.Tensor, src_locs: torch.Tensor, tar_locs: torch.Tensor):
        out = []
        for batch_idx in range(src_x.size(0)):
            edge_weights, edge_ids = inverse_distance_weighting(
                src_locs[batch_idx], tar_locs[batch_idx], dist_thresh=0.3, norm=False)

            temp = self.gconv1(
                (src_x[batch_idx], tar_x[batch_idx]), edge_ids, edge_weights)

            out.append(temp)

        return torch.stack(out, dim=0)

    def _forward_gconv2(self, src_x: torch.Tensor, tar_x: torch.Tensor, src_locs: torch.Tensor, tar_locs: torch.Tensor):
        out = []
        for batch_idx in range(src_x.size(0)):
            edge_weights, edge_ids = inverse_distance_weighting(
                src_locs[batch_idx], tar_locs[batch_idx], dist_thresh=0.8, norm=False)

            temp = self.gconv2(
                (src_x[batch_idx], tar_x[batch_idx]), edge_ids, edge_weights)

            out.append(temp)

        return torch.stack(out, dim=0)

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
